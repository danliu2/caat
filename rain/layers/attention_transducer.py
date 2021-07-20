from dataclasses import dataclass, field
from enum import auto
from typing import Any, Dict, List, Optional, Tuple
import torch
import os
import math
from torch import Tensor,autograd
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils, checkpoint_utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    transformer,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
    FairseqEncoder, FairseqIncrementalDecoder,
    BaseFairseqModel,FairseqEncoderDecoderModel
)
from fairseq.data import Dictionary
from fairseq.models.transformer import (
    Embedding, TransformerDecoder,
    TransformerEncoder,
    TransformerModel
)
from argparse import Namespace
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,Fp32LayerNorm,
    LayerDropModuleList,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    MultiheadAttention
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import II
from .rand_pos import PositionalEmbedding
from .audio_convs import lengths_to_padding_mask
from .audio_encoder import AudioTransformerEncoder
from fairseq.incremental_decoding_utils import with_incremental_state,FairseqIncrementalState
from warprnnt_pytorch import DelayTLoss
from torch.autograd import backward, grad
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from .multihead_attention_patched import MultiheadAttentionPatched

def patch_mha(mh_attn:MultiheadAttention):
    """
        build RelAttn based on given MultiheadAttention, 
        modules before works with fairseq.modules.MultiheadAttention,
    """
    rel_attn = MultiheadAttentionPatched(
        mh_attn.embed_dim, mh_attn.num_heads, mh_attn.kdim, mh_attn.vdim,
        mh_attn.dropout_module.p, bias=True,
        add_bias_kv=False, add_zero_attn=False, self_attention=False,
        encoder_decoder_attention=mh_attn.encoder_decoder_attention,
    )
    return rel_attn

class IsolatedDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens,):
        super().__init__(
            args,dictionary, embed_tokens, no_encoder_attn=True
        )
        for layer in self.layers:
            layer.self_attn = patch_mha(layer.self_attn)
        # only output h for fuse
        self.output_projection= None
        # for layer in self.layers:
        #     layer.self_attn.self_attention=False
    
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
    
    def forward(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
            for transducer, prev_output_tokens should be [bos] concat target
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=None,
            incremental_state=incremental_state,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
        )
        return x
    
    def buffered_future_mask_length(self, tensor, dim):
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
    
    def convert_cache_pad(self, incremental_state,new_idx):
        head_num = self.layers[0].self_attn.num_heads
        head_dim = self.layers[0].self_attn.head_dim
        B,T = new_idx.shape
        expand_idx= new_idx.view(B,1,T,1).expand(B,head_num, T, head_dim)
        for layer in self.layers:
            input_buffer = layer.self_attn._get_input_buffer(incremental_state)
            input_buffer["prev_key"]= input_buffer["prev_key"].gather(2, expand_idx)
            input_buffer["prev_value"]= input_buffer["prev_value"].gather(2, expand_idx)
            #prev_tokens.gather(1, index)
            input_buffer["prev_key_padding_mask"] = input_buffer["prev_key_padding_mask"].gather(1, new_idx)
            layer.self_attn._set_input_buffer(incremental_state, input_buffer)
        pass
    
    def recalc_h(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        processed_length =0
    ):
        full_prev_tokens= prev_output_tokens
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=None
            )
            if self.embed_positions is not None
            else None
        )
        if processed_length >0:
            
            prev_output_tokens = prev_output_tokens[:,processed_length:]
            if positions is not None:
                positions = positions[:,processed_length:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            self_attn_mask = self.buffered_future_mask_length(x, full_prev_tokens.shape[1])
            self_attn_mask= self_attn_mask[processed_length:]
            #import pdb;pdb.set_trace()
            x, layer_attn, _ = layer(
                x,
                None,
                None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )
            inner_states.append(x)

        if attn is not None:
            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

class TransducerOut(nn.Module):
    """
        this module is special, it will 1)fwd-bwd itself, for rnnt grad to input and weight,
        2) backward grad of input to previous network
    """
    def __init__(
        self, output_proj:nn.Module,
        delay_scale = 1.0,
        tokens_per_step=20000,
        blank=0,
        smoothing=0.,
        label_smoothing=0.1,
        delay_func="zero",
        pad= 1,
        ce_scale=1.0,
        temperature=1.0
    ):
        super().__init__()
        self.rnnt_loss= DelayTLoss(
            blank=blank,delay_scale= delay_scale, temperature=temperature,
            reduction= "sum", delay_func= delay_func
        )
        print(f"transducer temperature= {temperature}")
        self.output_proj= output_proj
        self.vocab_size= output_proj.weight.shape[0]
        self.delay_scale= delay_scale
        self.tokens_per_step=tokens_per_step
        self.smoothing= smoothing
        self.pad= pad
        self.label_smoothing=label_smoothing
        self.ce_scale= ce_scale
        #self.step =0
    
    def forward(self, x:Tensor):
        return self.output_proj(x)
    
    def _calc_entropy(self, logits, src_len, tgt_len):
        """
            Args:
                logits: BxSxTxV, float32
        """
        B,S,T,V= logits.shape
        with torch.no_grad():
            mask1= torch.arange(S).repeat(B,1).to(logits.device) >=src_len.unsqueeze(1)
            mask2 = torch.arange(T).repeat(B,1).to(logits.device) >= tgt_len.unsqueeze(1)
            mask = mask1.unsqueeze(2) | mask2.unsqueeze(1)
        logits= logits[mask]
        lprobs=F.log_softmax(logits, dim=-1)
        neg_entropy = (torch.exp(lprobs)*lprobs).sum()/lprobs.shape[0]
        loss= neg_entropy*B
        return loss
    
    def cross_entropy(self, x:Tensor, targets:Tensor,src_lengths:Tensor, tgt_lengths:Tensor,):
        """
            Args:
                x: B*G*T*D
        """
        B,G,T,D= x.shape
        
        idx_pos= (src_lengths-1).view(-1,1,1,1).expand(B,1,T,D)
        last_h = torch.gather(x, 1, idx_pos).squeeze(1).contiguous()
        last_h= last_h[:,:-1]
        logits= self.output_proj(last_h)
        lprobs= F.log_softmax(logits.float(),dim= -1)
        loss, nll_loss= label_smoothed_nll_loss(
            lprobs,
            targets,
            self.label_smoothing,
            ignore_index = self.pad,
            reduce = True
        )
        ntokens= targets.ne(self.pad).sum().item()
        #loss = loss*(B/ntokens)
        #nll_loss= nll_loss*(B/ntokens)
        return loss, nll_loss
        
    def train_step(
        self, x:Tensor, targets:Tensor,
        src_lengths:Tensor, tgt_lengths:Tensor,
        scaler=None
    ):
        """
        Args:
            x: BxTx(U+1)xd
            target: BxU
        """    
        B,T,U,d = x.shape
        bsz_per_step = max(self.tokens_per_step // (T*U),1)
        
        input_list= x.split(bsz_per_step)
        tgt_list= targets.split(bsz_per_step)
        slen_list= src_lengths.split(bsz_per_step)
        tlen_list= tgt_lengths.split(bsz_per_step)
        losses=[0,0,0,0]
        input_grads=[]
        
        for input_, tgt,slen, tlen in zip(input_list, tgt_list, slen_list, tlen_list):
            dup_ = input_.detach()
            dup_.requires_grad= True
           
            logits= self.forward(dup_)
            logits= logits.float()
            loss,loss_prob, loss_delay = self.rnnt_loss(
                logits, tgt.int(),
                slen.int(),tlen.int()
            )
            #import pdb;pdb.set_trace()
            #loss_me= self._calc_entropy(logits, slen,tlen)
            loss_smoothed, nll_loss = self.cross_entropy(dup_, tgt, slen, tlen)
            loss=loss+self.ce_scale*loss_smoothed
            losses=[l+l2 for l,l2 in zip(losses, \
                (loss.detach(), loss_prob.detach(), loss_delay.detach(), nll_loss.detach()))]
            
            if scaler is not None:
                loss= scaler.scale(loss)
           
            loss.backward()
            input_grads.append(dup_.grad.detach())
        input_grads= torch.cat(input_grads, dim= 0)
        # x_dummy = x.detach()
        # x_dummy.requires_grad=True
        # loss_smoothed, nll_loss= self.cross_entropy(x_dummy, targets,src_lengths, tgt_lengths)
        # loss_ce = self.ce_scale*loss_smoothed
        # loss_ce.backward()
        # input_grads += x_dummy.grad.detach()
        autograd.backward(x, input_grads)
        ntokens= targets.ne(self.pad).sum().item()
        
        # loss= losses[0] + loss_ce
        return {
            "loss":losses[0], "loss_prob":losses[1], "loss_delay":losses[2], "nll_loss":losses[3], "sample_size": ntokens,
        }
    
    def eval_step(
        self, x:Tensor, targets:Tensor,
        src_lengths:Tensor, tgt_lengths:Tensor
    ):
        B,T,U,d = x.shape
        bsz_per_step = max(self.tokens_per_step // (T*U),1)
        input_list= x.split(bsz_per_step)
        tgt_list= targets.split(bsz_per_step)
        slen_list= src_lengths.split(bsz_per_step)
        tlen_list= tgt_lengths.split(bsz_per_step)
        losses=[0,0,0, 0]

        for input_, tgt,slen, tlen in zip(input_list, tgt_list, slen_list, tlen_list):
            dup_ = input_.detach()
            dup_.requires_grad= True
           
            logits= self.forward(dup_)
            logits= logits.float()
            loss,loss_prob, loss_delay = self.rnnt_loss(
                logits, tgt.int(),
                slen.int(),tlen.int()
            )
            #import pdb;pdb.set_trace()
            #loss_me= self._calc_entropy(logits, slen,tlen)
            loss_smoothed, nll_loss = self.cross_entropy(dup_, tgt, slen, tlen)
            loss=loss+self.ce_scale*loss_smoothed
            losses=[l+l2 for l,l2 in zip(losses, \
                (loss.detach(), loss_prob.detach(), loss_delay.detach(), nll_loss.detach()))]
        ntokens= targets.ne(self.pad).sum().item()
        return {
            "loss":losses[0], "loss_prob":losses[1], "loss_delay":losses[2], "nll_loss":losses[3], "sample_size": ntokens,
        }
        


class ConcatJointNet(nn.Module):
    def __init__(
        self, encoder_dim, decoder_dim, 
        hid_dim, 
        activation="tanh",
        downsample=1,
    ):
        super().__init__()
        self.downsample=downsample
        self.encoder_proj = nn.Linear(encoder_dim, hid_dim)
        self.decoder_proj = nn.Linear(decoder_dim, hid_dim)
        self.activation_fn = utils.get_activation_fn(activation)
        self.hid_dim=hid_dim
        if downsample < 1:
            raise ValueError("downsample should be more than 1 for concat_joint")
        
    def forward(self, encoder_out:Dict[str, List[Tensor]], decoder_state, Tensor):
        """
            use dimension same as transformer
            Args:
            encoder_out: "encoder_out": TxBxC
            decoder_state: BxUxC
        """
        encoder_state= encoder_out["encoder_out"][0]
        encoder_state= encoder_state[::self.downsample].contiguous()
        encoder_state= encoder_state.transpose(0,1)

        h_enc= self.encoder_proj(encoder_state)
        h_dec= self.decoder_proj(decoder_state)
        h_joint = h_enc.unsqueeze(2) + h_dec.unsqueeze(1)
        h_joint= self.activation_fn(h_joint)
        return h_joint


class AttentionJointNet(nn.Module):
    def __init__(
        self, encoder_dim, decoder_dim, 
        hid_dim, 
        downsample=1,
        activation="tanh"
    ):
        super().__init__()
        self.downsample=downsample
        self.k_proj = nn.Linear(encoder_dim, hid_dim)
        self.v_proj = nn.Linear(encoder_dim, hid_dim)
        self.q_proj = nn.Linear(decoder_dim, hid_dim)
        self.decoder_proj = nn.Linear(decoder_dim, hid_dim)
        self.activation_fn = utils.get_activation_fn(activation)
        self.hid_dim=hid_dim
    
    def calc_uniattn(
        self, encoder_state, encoder_padding_mask,
        decoder_state
    ):
        q= self.q_proj(decoder_state)
        k = self.k_proj(encoder_state)
        # BxUxD, BxTxD-> BxUxT
        attn_weights= torch.bmm(q, k.transpose(1,2))
        
        attn_weights= attn_weights.masked_fill(
            encoder_padding_mask.unsqueeze(1),
            float("-inf")
        )
        attn_prob_float = F.softmax(attn_weights.float(), dim=-1)
        attn_prob = attn_prob_float.type_as(encoder_state)
        # BxTxD
        v= self.v_proj(encoder_state)
        # BxUxT, BxTxD-> BxGxUxD
        attn_out= torch.einsum("but,btd->bud",attn_prob, v)
        attn_out = attn_out.unsqueeze(1)
        group_lengths= decoder_state.new(decoder_state.shape[0]).long().fill_(1)
        return attn_out, group_lengths, attn_prob
    
    def calc_attn(
        self, encoder_state, encoder_padding_mask,
        decoder_state
    ):
        B,T= encoder_padding_mask.shape
        
        with torch.no_grad():
            group_num = math.ceil(T /self.downsample)
            group_pos= torch.arange(1,group_num+1)*self.downsample
            tidx= torch.arange(T)
            group_mask = group_pos.unsqueeze(1)<=tidx.unsqueeze(0)
            #group_pos= torch.arange(group_num)
            #group_mask = group_pos.unsqueeze(1)<=group_pos.unsqueeze(0)
            # GxG
            group_mask = group_mask.to(encoder_padding_mask.device)
            group_mask_float= encoder_state.new(group_mask.shape).fill_(0)
            group_mask_float = group_mask_float.masked_fill(group_mask, -1e8)
            encout_lengths= (~encoder_padding_mask).sum(1).float()
            group_lengths= (encout_lengths/self.downsample).ceil().long()
        
        q= self.q_proj(decoder_state)
        k = self.k_proj(encoder_state)
        # BxUxD, BxTxD-> BxUxT
        attn_weights= torch.bmm(q, k.transpose(1,2))
        # BxUxT->BxGxUxT
        attn_weights= attn_weights.unsqueeze(1)+group_mask_float.unsqueeze(0).unsqueeze(2)
        attn_weights= attn_weights.masked_fill(
            encoder_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf")
        )
        attn_prob_float = F.softmax(attn_weights.float(), dim=-1)
        attn_prob = attn_prob_float.type_as(encoder_state)
        # BxTxD
        v= self.v_proj(encoder_state)
        # BxGxUxT, BxTxD-> BxGxUxD
        attn_out= torch.einsum("bgut,btd->bgud",attn_prob, v)

        return attn_out,group_lengths, attn_prob

    def forward(
        self, encoder_out:Dict[str, List[Tensor]], decoder_state:Tensor,
        incremental_state= None
    ):
        """
            incremental_state is just flag for train or inference
        """
        encoder_state= encoder_out["encoder_out"][0]
        encoder_padding_mask= encoder_out["encoder_padding_mask"][0]
        encoder_state= encoder_state.transpose(0,1)
        
        if self.downsample <0 or (incremental_state is not None):
            # only shallow joint at full sequence, may be efficient pretrain for SST
            attn_out, group_lengths, attn_prob = self.calc_uniattn(encoder_state, encoder_padding_mask, decoder_state)
        else:
            attn_out, group_lengths, _ = self.calc_attn(encoder_state, encoder_padding_mask, decoder_state)
        
        h_dec= self.decoder_proj(decoder_state)
        
        h_joint= attn_out + h_dec.unsqueeze(1)
        h_joint= self.activation_fn(h_joint)
        return h_joint,group_lengths



@with_incremental_state
class ExpandMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,dropout=0.0):
        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        assert embed_dim %num_heads == 0
        self.embed_dim= embed_dim
        self.num_heads= num_heads
        self.head_dim= embed_dim//num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj=nn.Linear(embed_dim, embed_dim)
        self.k_proj=nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
    
    def forward(
        self, query, key,
        key_padding_mask= None,
        group_attn_mask=None,
        incremental_state= None,
    ):
        """
            Args:
                query: TxBxD
                key:   SxBxD
                key_padding_mask: BxS, bool
                group_attn_mask: BxGxS, mask for each group, used to expand attention energy
                incremental_state: cache k,v for next step, need not rollback,reorder only for beam size
        """
        if query.dim()==3:
            tgt_len,bsz, embed_dim= query.size()
            pre_group_num=1
            query = query.unsqueeze(0)
        else:
            pre_group_num, tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        src_len = key.shape[0]
        key_processed=False
        k,v= None,None
        # TODO: by danliu, should be modified faster( reused processed k,v, only calculate new added)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                prev_len = saved_state['prev_key'].shape[2]
                if prev_len == src_len:
                    k = saved_state['prev_key'].view(-1, src_len, self.head_dim)
                    v= saved_state['prev_value'].view(-1, src_len, self.head_dim)
                    key_processed= True
        q= self.q_proj(query).view(pre_group_num*tgt_len, bsz*self.num_heads, self.head_dim).transpose(0,1)
        q= q*self.scaling
        if not key_processed:
            k = self.k_proj(key).view(src_len, bsz*self.num_heads, self.head_dim).transpose(0,1)
            v= self.v_proj(key).view(src_len, bsz*self.num_heads, self.head_dim).transpose(0,1)
            if incremental_state is not None:
                saved_state = {}
                saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
                saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
                incremental_state = self._set_input_buffer(incremental_state, saved_state)

        #(b*numheads)*(groupnum*tgtlen)*srclen
        attn_weight= torch.bmm(q, k.transpose(1,2))
        if key_padding_mask is not None:
            attn_weight = attn_weight.view(bsz, self.num_heads, pre_group_num, tgt_len, src_len)
            attn_weight = attn_weight.masked_fill(
                key_padding_mask.view(bsz, 1,1,1, src_len).to(torch.bool),
                float("-inf"),
            )
        
        group_num=1
        if group_attn_mask is not None:
            #  BxGxS, (B*numheads)x(G*T)xS->(B*numheads)xGxTxS
            group_num = group_attn_mask.shape[1]
            assert group_num == pre_group_num or pre_group_num ==1
            attn_weight= attn_weight.view(bsz, self.num_heads,pre_group_num, tgt_len, src_len)
            
            group_attn_mask = group_attn_mask.view(bsz, 1, group_num, 1, src_len)
            attn_weight = (attn_weight +group_attn_mask).view(bsz*self.num_heads, group_num, tgt_len, src_len).contiguous()
        else:
            assert pre_group_num == 1
            attn_weight = attn_weight.view(bsz*self.num_heads, pre_group_num, tgt_len, src_len)
        
        attn_prob = F.softmax(attn_weight.float(), dim= -1).to(attn_weight)
        attn_prob_drop = self.dropout_module(attn_prob)
       
        attn_out = torch.einsum("bgts,bsd->bgtd", attn_prob_drop, v)
        attn_out= attn_out.view(bsz,self.num_heads,group_num, tgt_len, self.head_dim).permute(2,3,0,1,4)\
            .contiguous().view(group_num, tgt_len, bsz,embed_dim)
        out= self.out_proj(attn_out)
        return out,attn_prob
        
            
        



class TransformerJointerLayer(nn.Module):
    def __init__(self, args:Namespace):
        super().__init__()
        self.embed_dim= getattr(args,'jointer_embed_dim', 256)
        num_heads= getattr(args, "jointer_attention_heads", 4)
        downsample= getattr(args, "transducer_downsample", 4)
        self.enc_attn = ExpandMultiheadAttention(
            self.embed_dim, num_heads, dropout= args.attention_dropout
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        hid_dim= getattr(args, "jointer_ffn_embed_dim", self.embed_dim*4)
        self.fc1= nn.Linear(self.embed_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, self.embed_dim)
        self.attn_layer_norm= LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
    
    def forward(
        self,
        x,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        group_attn_mask:Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        if x.dim()==3:
            x= x.unsqueeze(0)
        residual=x
        if self.normalize_before:
            x= self.attn_layer_norm(x)
        x, _ = self.enc_attn(
            x, encoder_out, encoder_padding_mask, 
            group_attn_mask = group_attn_mask,
            incremental_state =incremental_state
        )
        x= self.dropout_module(x)
        x= x+residual
        if not self.normalize_before:
            x= self.attn_layer_norm(x)
        residual= x
        if self.normalize_before:
            x= self.final_layer_norm(x)
        x= self.activation_fn(self.fc1(x))
        x= self.activation_dropout_module(x)
        x= self.fc2(x)
        x= self.dropout_module(x)
        x= x+residual
        if not self.normalize_before:
            x= self.final_layer_norm(x)
        return x
        



class MHAJointNet(nn.Module):
    def __init__(
        self, args
    ):
        super().__init__()
        self.downsample= getattr(args,'transducer_downsample',-1)
        nlayers= getattr(args,"jointer_layers",1)
        self.layers= nn.ModuleList([])
        self.layers.extend(
            [TransformerJointerLayer(args) for _ in range(nlayers)]
        )
    
    def _gen_group_mask(self, encoder_out, encoder_padding_mask):
        B,T= encoder_padding_mask.shape
        with torch.no_grad():
            group_num = math.ceil(T /self.downsample)
            group_pos= torch.arange(1,group_num+1)*self.downsample
            tidx= torch.arange(T)
            group_mask = group_pos.unsqueeze(1)<=tidx.unsqueeze(0)
            group_mask = group_mask.to(encoder_padding_mask.device)
            group_mask_float= encoder_out.new(group_mask.shape).fill_(0)
            group_mask_float = group_mask_float.masked_fill(group_mask, float("-inf"))
            group_mask_float = group_mask_float.unsqueeze(0).repeat(B,1,1)
            encout_lengths= (~encoder_padding_mask).sum(1).float()
            group_lengths= (encout_lengths/self.downsample).ceil().long()
        return group_mask_float, group_lengths
    
    def forward(
        self, encoder_out:Dict[str, List[Tensor]], decoder_state:Tensor,
        incremental_state= None
    ):
        encoder_state= encoder_out['encoder_out'][0]
        encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
        if self.downsample >0:
            group_mask, group_lengths= self._gen_group_mask(encoder_state, encoder_padding_mask)
        else:
            group_mask= None
            group_lengths= decoder_state.new(decoder_state.shape[0]).long().fill_(1)
        x= decoder_state.transpose(0,1)
        for layer in self.layers:
            x= layer(
                x,
                encoder_state,
                encoder_padding_mask,
                group_attn_mask= group_mask,
                incremental_state= incremental_state
            )
        #gxtxbxd->bxgxtxd
        x= x.permute(2,0,1,3)
        return x, group_lengths


class TransducerMHADecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary:Dictionary, embed_tokens:nn.Module):
        super().__init__(dictionary)
        self.lm = IsolatedDecoder(args, dictionary, embed_tokens)
        self.output_embed_dim = args.decoder_output_dim
        out_proj = nn.Linear(args.decoder_output_dim, len(dictionary), bias=False)
        if args.share_decoder_input_output_embed:
            out_proj.weight= embed_tokens.weight
        else:
            nn.init.normal_(
                out_proj.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        self.blank= dictionary.bos()
        self.transducer_out= TransducerOut(
            out_proj,
            delay_scale= getattr(args,"delay_scale",1.0),
            tokens_per_step = getattr(args,"tokens_per_step", 20000),
            blank=self.blank,
            smoothing= getattr(args, "transducer_smoothing", 0.),
            label_smoothing= getattr(args, "transducer_label_smoothing", 0.1),
            delay_func=getattr(args,"delay_func", "zero"),
            pad= dictionary.pad(),
            ce_scale= getattr(args, "transducer_ce_scale", 1.0),
            temperature= getattr(args, "transducer_temperature",1.0)
        )

        self.downsample=  getattr(args, "transducer_downsample", 1)
        self.jointer = MHAJointNet(args)
        self.train_as_ed = getattr(args, "train_as_ed", False)
        if self.train_as_ed:
            assert self.downsample == -1, "train_as_ed need downsample to be -1"
    
    def forward(
        self,
        prev_output_tokens:Tensor,
        encoder_out:Dict[str, List[Tensor]],
        incremental_state:Optional[Dict[str, Dict[str, Optional[Tensor]]]]=None
    ):
        h_lm = self.lm(prev_output_tokens, incremental_state)
        if incremental_state is not None:
            self.jointer.downsample =-1
       
        joint_result, group_lengths= self.jointer(encoder_out, h_lm, incremental_state= incremental_state)
        
        if incremental_state is not None or self.train_as_ed:
            logits=self.transducer_out(joint_result)
            logits= logits.squeeze(1)
            return logits, {"attn":None, "h_lm":h_lm}
        else:
            #return hidden, and do multi-step forward at transducer_out
            return joint_result,group_lengths
    
    def recalc_logits(
        self,
        h_lm,
        encoder_out, incremental_state
    ):
        joint_result,group_lengths = self.jointer(encoder_out, h_lm, incremental_state= incremental_state)
        logits= self.transducer_out(joint_result).squeeze(1)
        return logits
    
    def rollback_steps(self, incremental_state, step_to_keep:int):
        for layer in self.lm.layers:
            input_buffer = layer.self_attn._get_input_buffer(incremental_state)
            input_buffer["prev_key"]= input_buffer["prev_key"][:,:,:step_to_keep]
            input_buffer["prev_value"]= input_buffer["prev_value"][:,:,:step_to_keep]
            input_buffer["prev_key_padding_mask"] = input_buffer["prev_key_padding_mask"][:,:step_to_keep]
            layer.self_attn._set_input_buffer(incremental_state, input_buffer)
   


