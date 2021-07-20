from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import os
from torch import Tensor
import torch.nn as nn
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

from rain.layers.multihead_attention_relative import MultiheadRelativeAttention, replace_relative_attention

class IncrementalDictState(FairseqIncrementalState):
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
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "buffer")
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
        return self.set_incremental_state(incremental_state, "buffer", buffer)

def with_incremental_dict(cls):
    cls.__bases__ = (IncrementalDictState,) + tuple(
        b for b in cls.__bases__ if b != IncrementalDictState
    )
    return cls

@with_incremental_dict
class UnidirectConv2D(nn.Module):
    """
        similar to Shallow2D, for online, we remove padding for length, 
        and add prev_extra frame for current block processing
    """
    downsample_ratio= 4
    def __init__(
        self, 
        num_mel_bins,
        output_dim,
        conv_channels= (128,128)
    ):
        super().__init__()
        self.num_mel_bins= num_mel_bins
        self.output_dim= output_dim
        self.in_channels= input_channels= 1
        self.conv_layers=nn.ModuleList()
        self.pooling_kernel_sizes= []
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(
                    input_channels, out_channels,
                    (3,3),
                    stride=(2,1),
                    padding=(0,1)
                )
            )
            input_channels = out_channels
            self.conv_layers.append(nn.ReLU())
            self.pooling_kernel_sizes.append(2)
        
        conv_agg_dim = num_mel_bins*conv_channels[-1]
        self.out_proj= nn.Linear(conv_agg_dim, output_dim)
        self._extra_frames= 0
        kernel_size= 3
        for pool_size in self.pooling_kernel_sizes[::-1]:
            self._extra_frames = self._extra_frames* pool_size +kernel_size -1
    
    @property
    def extra_frames(self):
        return self._extra_frames

    def forward(
        self, fbank, fbk_lengths,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz, seq_len, _ = fbank.shape
        x= fbank.view(bsz, seq_len, self.in_channels, self.num_mel_bins)
        x= x.transpose(1,2).contiguous()

        if incremental_state is not None:
            input_state= self._get_input_buffer(incremental_state)
            curr_x = x
            if "raw_fea" in input_state:
                pre = input_state["raw_fea"]
                pre_length = min(self.extra_frames, pre.shape[2])
                x= torch.cat((pre[:,:,-pre_length:], x),dim =2)
                fbk_lengths += pre_length
                pre = torch.cat((pre, curr_x), dim =2)
                input_state["raw_fea"] = pre
            else:
                input_state["raw_fea"] = curr_x
            incremental_state = self._set_input_buffer(incremental_state, input_state)
        
        for layer in self.conv_layers:
            x= layer(x)

        input_lengths= fbk_lengths - self.extra_frames
        for s in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float()/s).ceil().long()
        #(B,C,T,fea)->(T,B,C*feature)
        bsz, _, out_seq_len, _ = x.shape
        x= x.permute(2,0,1,3).contiguous().view(out_seq_len, bsz, -1)
        x= self.out_proj(x)
        padding_mask = lengths_to_padding_mask(input_lengths, x)
        return x, padding_mask
 

@with_incremental_dict
class UnidirectTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, main_context= 1, right_context= 0):
        """
            block length may be infected by dowansmaple, so we get from additional params, not args
        """
        super().__init__(args)
        self.block_size= main_context
        self.right_context= right_context
    
    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True, 
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
    
    def forward(
        self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        rel_pos: Optional[Tensor] = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (FloatTensor): -1e8 for masked, 0 for normal, remove magic tricks from fairseq
            incremental_state: for online decoding, calculate current 
                `main_context+right_context` frames, and cache `main_context` frames
                for next inference

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e4
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        key =x
        key_padding_mask= encoder_padding_mask
        if incremental_state is not None:
            # catch h instead of key, value, maybe waste computation, just for simplify code
            input_state= self.self_attn._get_input_buffer(incremental_state)
            if "prev_key" in input_state and attn_mask is not None:
                prev_len = input_state["prev_key"].shape[2]
                pre_attn_mask = attn_mask.new(attn_mask.shape[0], prev_len).fill_(0)
                attn_mask= torch.cat((pre_attn_mask, attn_mask),dim=1)
                
        if isinstance(self.self_attn, MultiheadRelativeAttention):
            x, _ = self.self_attn(
                query=x,
                key=key,
                value=key,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                incremental_state= incremental_state,
                rel_pos = rel_pos
            )    
        else:        
            x, _ = self.self_attn(
                query=x,
                key=key,
                value=key,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                incremental_state= incremental_state
            )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

def gen_block_atten_mask(
    x:Tensor,
    padding_mask:Tensor, 
    main_context:int =1,
    right_context:Tensor = 0
):
    """
    Args:
        x: inpout embedding, TxBxC
    """
    bsz,seq_len = padding_mask.shape
    block_num = seq_len//main_context
    block_idx = torch.arange(seq_len).to(padding_mask.device)//main_context
    pos = torch.arange(seq_len).to(padding_mask.device)
    if right_context == 0:
        attn_mask = block_idx.unsqueeze(1)<block_idx.unsqueeze(0)
        rel_pos = None
    else:
        
        with torch.no_grad():
            rc_block_idx = torch.arange(block_num)
            rc_block_pos = rc_block_idx.unsqueeze(1).repeat(1, right_context).view(-1).to(padding_mask.device)
            rc_blcok_step= (rc_block_idx.unsqueeze(1)+1)*main_context
            rc_inc_idx= torch.arange(right_context).unsqueeze(0)
            rc_idx= (rc_blcok_step +rc_inc_idx).view(-1).to(padding_mask.device)
            rc_idx_mask = (rc_idx >(seq_len -1)).to(padding_mask)
            rc_idx= rc_idx.clamp(0, seq_len -1)
            
            rc_padding_mask = padding_mask.index_select(1, rc_idx)
            # mask extra length
            rc_padding_mask= rc_padding_mask| rc_idx_mask.unsqueeze(0)
            
            padding_mask = torch.cat((padding_mask, rc_padding_mask), dim=1)
            full_idx = torch.cat((block_idx, rc_block_pos), dim= 0)
            attn_mask1 = full_idx.unsqueeze(1)< block_idx.unsqueeze(0)
            attn_mask2= full_idx.unsqueeze(1).ne(rc_block_pos.unsqueeze(0))
            attn_mask = torch.cat([attn_mask1,attn_mask2], dim=1)
            
            rel_pos = torch.cat((pos,rc_idx))
        rc_x = x.index_select(0, rc_idx)
        x = torch.cat((x, rc_x), dim= 0)
    attn_mask_float= x.new(attn_mask.shape).fill_(0)
    attn_mask_float = attn_mask_float.masked_fill(attn_mask.to(torch.bool), -1e4)
    return x, padding_mask, attn_mask_float,rel_pos



@with_incremental_dict
class UnidirectTransoformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        self.embed_dim = embed_tokens.embedding_dim
        self.main_context= getattr(args, "main_context", 1)
        self.right_context = getattr(args, "right_context", 0)
        super().__init__(args, dictionary, embed_tokens)
        
    def build_encoder_layer(self, args):
        layer = UnidirectTransformerEncoderLayer(args, self.main_context,self.right_context)
        
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        if getattr(args,"encoder_max_relative_position",-1) >0:
            layer.self_attn=replace_relative_attention(layer.self_attn, args.encoder_max_relative_position)
        return layer
    
    @property
    def init_frames(self):
        return self.main_context +self.right_context
    
    @property
    def step_frames(self):
        return self.main_context
    
    def forward_infer(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        incremental_state= None,
        finished=False,
        **kwargs
    ):
        """
            a faster version for inference, we use mask trick similar to training, need not to run step by step
        """
        token_embedding = self.embed_tokens(src_tokens)
        x = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            # cache src_tokens if incremental_state
            if incremental_state is not None:
                input_state= self._get_input_buffer(incremental_state)
                full_tokens=src_tokens
                if "prev_tokens" in input_state:
                    full_tokens= torch.cat((input_state["prev_tokens"], full_tokens),dim=1)
                pos_emb = self.embed_positions(full_tokens)
                x= x+ pos_emb[:, -src_tokens.shape[1]:]
                input_state["prev_tokens"] = full_tokens
                incremental_state = self._set_input_buffer(
                    incremental_state, input_state
                )
            else:
                x = x + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        attn_mask =None
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        
        if self.right_context >0 and incremental_state is not None:
            # cache current input for next block
            input_state= self._get_input_buffer(incremental_state)
            if "rc_input" in input_state:
                pre = input_state["rc_input"].transpose(0,1)
                x = torch.cat([pre, x], dim= 0)
                if "rc_mask" in input_state:
                    pre_mask = input_state["pre_mask"]
                else:
                    pre_mask = encoder_padding_mask.new(pre.shape[1], pre.shape[0]).fill_(0)
                encoder_padding_mask = torch.cat((pre_mask, encoder_padding_mask), dim =1)
            rc_input = x[-self.right_context:].transpose(0,1)
            rc_mask = encoder_padding_mask[:, -self.right_context:]
            input_state["rc_input"] = rc_input
            input_state["rc_mask"] = rc_mask
            incremental_state = self._set_input_buffer(
                incremental_state, input_state
            )
        curr_frames= x.shape[0]
        x, encoder_padding_mask,attn_mask,rel_pos = gen_block_atten_mask(
            x, encoder_padding_mask, self.main_context, self.right_context
        )
       
        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state = incremental_state,
                rel_pos = rel_pos
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        removed_length= x.shape[0]- curr_frames
        x=x[:curr_frames]
        encoder_padding_mask=encoder_padding_mask[:,:curr_frames]
        if not finished and self.right_context >0:
            removed_length+= self.right_context
            x= x[:-self.right_context]
            encoder_padding_mask= encoder_padding_mask[:,:-self.right_context]
        
        
        self.rollback_steps(incremental_state, removed_length)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
    
    def rollback_steps(self, incremental_state, removed_length:int):
        if incremental_state is None:
            return
        if removed_length == 0:
            return
        for layer in self.layers:
            input_buffer = layer.self_attn._get_input_buffer(incremental_state)
            input_buffer["prev_key"]= input_buffer["prev_key"][:,:,:-removed_length]
            input_buffer["prev_value"]= input_buffer["prev_value"][:,:,:-removed_length]
            input_buffer["prev_key_padding_mask"] = None
            layer.self_attn._set_input_buffer(incremental_state, input_buffer)


    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        incremental_state= None,
        finished=False,
        **kwargs
    ):
        if incremental_state is not None:
            return self.forward_infer(src_tokens, src_lengths, incremental_state, finished, **kwargs)
        token_embedding = self.embed_tokens(src_tokens)
        x = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            # cache src_tokens if incremental_state
            if incremental_state is not None:
                input_state= self._get_input_buffer(incremental_state)
                full_tokens=src_tokens
                if "prev_tokens" in input_state:
                    full_tokens= torch.cat((input_state["prev_tokens"], full_tokens),dim=1)
                pos_emb = self.embed_positions(full_tokens)
                x= x+ pos_emb[:, -src_tokens.shape[1]:]
                input_state["prev_tokens"] = full_tokens
                incremental_state = self._set_input_buffer(
                    incremental_state, input_state
                )
            else:
                x = x + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        curr_frames= x.shape[0]
        attn_mask =None
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if incremental_state is None:
            # build attn_mask for left and right context
            x, encoder_padding_mask,attn_mask = gen_block_atten_mask(
                x, encoder_padding_mask, self.main_context, self.right_context
            )
            pass
        elif self.right_context >0:
            # cache current input for next block
            input_state= self._get_input_buffer(incremental_state)
            if "rc_input" in input_state:
                pre = input_state["rc_input"].transpose(0,1)
                x = torch.cat([pre, x], dim= 0)
                if "rc_mask" in input_state:
                    pre_mask = input_state["pre_mask"]
                else:
                    pre_mask = encoder_padding_mask.new(pre.shape[1], pre.shape[0]).fill_(0)
                encoder_padding_mask = torch.cat((pre_mask, encoder_padding_mask), dim =1)
            rc_input = x[-self.right_context:].transpose(0,1)
            rc_mask = encoder_padding_mask[:, -self.right_context:]
            input_state["rc_input"] = rc_input
            input_state["rc_mask"] = rc_mask
            incremental_state = self._set_input_buffer(
                incremental_state, input_state
            )
        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state = incremental_state
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if incremental_state is None:
            x=x[:curr_frames]
            encoder_padding_mask=encoder_padding_mask[:,:curr_frames]
        elif not finished and self.right_context >0:
            x= x[:-self.right_context]
            encoder_padding_mask= encoder_padding_mask[:,:-self.right_context]
            if incremental_state is not None:
                self.rollback_steps(incremental_state, self.right_context)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
    
    def reorder_incremental_state_scripting(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        for module in self.modules():
            # self is in self.modules()
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result

@with_incremental_dict
class UnidirectAudioTransformerEncoder(AudioTransformerEncoder):
    def __init__(self, args,):
        ds_ratio = UnidirectConv2D.downsample_ratio
        self.ds_ratio = ds_ratio
        mc= getattr(args, "main_context", 16)
        rc = getattr(args, "right_context", 16)
        self.main_context = mc//ds_ratio
        self.right_context= rc//ds_ratio
        assert (self.main_context*ds_ratio == mc) and (self.right_context*ds_ratio == rc)
        super().__init__(args)
        self.extra_frames = self.conv_layers.extra_frames
    
    @property
    def init_frames(self):
        #return (self.main_context +self.right_context)*self.ds_ratio + self.extra_frames
        return (self.main_context +self.right_context)*self.ds_ratio
    
    @property
    def step_frames(self):
        return self.main_context*self.ds_ratio

    def build_conv_layers(self, args):
        """
            should support incremental_state, extra_framse
        """
        convs= UnidirectConv2D(args.num_mel_bins, args.encoder_embed_dim,(64,64))
        self.extra_frames= convs.extra_frames
        return convs
    
    def build_encoder_layer(self, args):
        layer = UnidirectTransformerEncoderLayer(args, self.main_context, self.right_context)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
       
        if getattr(args,"encoder_max_relative_position",-1) >0:
            layer.self_attn=replace_relative_attention(layer.self_attn, args.encoder_max_relative_position)
        return layer
    
    def forward(
        self,
        fbank:torch.Tensor,
        fbk_lengths:torch.Tensor,
        incremental_state= None,
        finished=False,
        **kwargs
    ):
        if incremental_state is not None:
            return self.forward_infer(fbank, fbk_lengths, incremental_state, finished, **kwargs)
        #padding at the first block or offline
        if incremental_state is None or len(incremental_state)== 0:
            B,T,C= fbank.shape
            fbk_lengths+= self.extra_frames
            head= fbank.new(B,self.extra_frames,C).fill_(0)
            fbank= torch.cat((head,fbank),dim=1)
        # x is already TBC
        x, encoder_padding_mask = self.conv_layers(fbank, fbk_lengths, incremental_state)
        curr_frames= x.shape[0]
        
        fake_tokens= encoder_padding_mask.long()
        # layernorm after garbage convs
        x= self.layernorm_embedding(x)
        if self.embed_positions is not None:
            # cache src_tokens if incremental_state
            if incremental_state is not None:
                input_state= self._get_input_buffer(incremental_state)
                full_tokens=fake_tokens
                if "prev_tokens" in input_state:
                    full_tokens= torch.cat((input_state["prev_tokens"], full_tokens),dim=1)
                pos_emb = self.embed_positions(full_tokens)
                x= x+ pos_emb[:, -fake_tokens.shape[1]:].contiguous().transpose(0,1)
                input_state["prev_tokens"] = full_tokens
                incremental_state = self._set_input_buffer(
                    incremental_state, input_state
                )
            else:
                x = x + self.embed_positions(fake_tokens).transpose(0,1)
            
        attn_mask = None
        if incremental_state is None:
            # build attn_mask for left and right context
            x, encoder_padding_mask,attn_mask,rel_pos = gen_block_atten_mask(
                x, encoder_padding_mask, self.main_context, self.right_context
            )
            pass
        elif self.right_context >0:
            # cache current input for next block
            input_state= self._get_input_buffer(incremental_state)
            if "rc_input" in input_state:
                pre = input_state["rc_input"].transpose(0,1)
                x = torch.cat([pre, x], dim= 0)
                if "rc_mask" in input_state:
                    pre_mask = input_state["rc_mask"]
                else:
                    pre_mask = encoder_padding_mask.new(pre.shape[1], pre.shape[0]).fill_(0)
                encoder_padding_mask = torch.cat((pre_mask, encoder_padding_mask), dim =1)
            rc_input = x[-self.right_context:].transpose(0,1)
            rc_mask = encoder_padding_mask[:, -self.right_context:]
            input_state["rc_input"] = rc_input
            input_state["rc_mask"] = rc_mask
            incremental_state = self._set_input_buffer(
                incremental_state, input_state
            )

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state = incremental_state, rel_pos = rel_pos
            )


        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if incremental_state is None:
            x=x[:curr_frames]
            encoder_padding_mask=encoder_padding_mask[:,:curr_frames]
        elif not finished and self.right_context >0:
            x= x[:-self.right_context]
            encoder_padding_mask= encoder_padding_mask[:, :-self.right_context]
            self.rollback_steps(incremental_state, self.right_context)
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "dec1_state":[], # reserved for joint decoding
            "dec1_padding_mask":[],
        }

    def forward_infer(
        self,
        fbank:torch.Tensor,
        fbk_lengths:torch.Tensor,
        incremental_state= None,
        finished=False,
        **kwargs
    ):
        
        #padding at the first block or offline
        if incremental_state is None or len(incremental_state)== 0:
            B,T,C= fbank.shape
            fbk_lengths+= self.extra_frames
            head= fbank.new(B,self.extra_frames,C).fill_(0)
            fbank= torch.cat((head,fbank),dim=1)
        # x is already TBC
        x, encoder_padding_mask = self.conv_layers(fbank, fbk_lengths, incremental_state)
        
        
        fake_tokens= encoder_padding_mask.long()
        # layernorm after garbage convs
        x= self.layernorm_embedding(x)
        if self.embed_positions is not None:
            # cache src_tokens if incremental_state
            if incremental_state is not None:
                input_state= self._get_input_buffer(incremental_state)
                full_tokens=fake_tokens
                if "prev_tokens" in input_state:
                    full_tokens= torch.cat((input_state["prev_tokens"], full_tokens),dim=1)
                pos_emb = self.embed_positions(full_tokens)
                x= x+ pos_emb[:, -fake_tokens.shape[1]:].contiguous().transpose(0,1)
                input_state["prev_tokens"] = full_tokens
                incremental_state = self._set_input_buffer(
                    incremental_state, input_state
                )
            else:
                x = x + self.embed_positions(fake_tokens).transpose(0,1)
            
        attn_mask = None
        if self.right_context >0 and incremental_state is not None:
            # cache current input for next block
            input_state= self._get_input_buffer(incremental_state)
            if "rc_input" in input_state:
                pre = input_state["rc_input"].transpose(0,1)
                x = torch.cat([pre, x], dim= 0)
                if "rc_mask" in input_state:
                    pre_mask = input_state["rc_mask"]
                else:
                    pre_mask = encoder_padding_mask.new(pre.shape[1], pre.shape[0]).fill_(0)
                encoder_padding_mask = torch.cat((pre_mask, encoder_padding_mask), dim =1)
            rc_input = x[-self.right_context:].transpose(0,1)
            rc_mask = encoder_padding_mask[:, -self.right_context:]
            input_state["rc_input"] = rc_input
            input_state["rc_mask"] = rc_mask
            incremental_state = self._set_input_buffer(
                incremental_state, input_state
            )

        curr_frames= x.shape[0]

        x, encoder_padding_mask,attn_mask,rel_pos = gen_block_atten_mask(
            x, encoder_padding_mask, self.main_context, self.right_context
        )
       

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                attn_mask=attn_mask,
                incremental_state = incremental_state,rel_pos =rel_pos
            )


        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        removed_length= x.shape[0]- curr_frames
        
        x=x[:curr_frames]
        encoder_padding_mask=encoder_padding_mask[:,:curr_frames]
        if not finished and self.right_context >0:
            removed_length+= self.right_context
            x= x[:-self.right_context]
            encoder_padding_mask= encoder_padding_mask[:,:-self.right_context]
        if incremental_state is not None:
            self.rollback_steps(incremental_state, removed_length)
        
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "dec1_state":[], # reserved for joint decoding
            "dec1_padding_mask":[],
        }

    def rollback_steps(self, incremental_state, removed_length:int):
        if incremental_state is None:
            return
        if removed_length == 0:
            return
        for layer in self.layers:
            input_buffer = layer.self_attn._get_input_buffer(incremental_state)
            input_buffer["prev_key"]= input_buffer["prev_key"][:,:,:-removed_length]
            input_buffer["prev_value"]= input_buffer["prev_value"][:,:,:-removed_length]
            input_buffer["prev_key_padding_mask"] = None
            layer.self_attn._set_input_buffer(incremental_state, input_buffer)
