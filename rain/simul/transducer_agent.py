from simuleval.agents import Agent,TextAgent, SpeechAgent
from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
from typing import List,Dict, Optional
import numpy as np
import math
import torch
from collections import deque
from torch import Tensor
import torch.nn as nn
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.data import encoders, Dictionary
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from argparse import Namespace
import rain
from rain.data.transforms import audio_encoder, text_encoder
from rain.simul.waitk_agent import WordEndChecker, OnlineSearcher, WaitkAgent
import logging
logger = logging.getLogger('transducer.agent')


class OnlineModels(nn.Module):
    def __init__(self, models:List[FairseqEncoderDecoderModel]):
        super().__init__()
        self.models= nn.ModuleList(models)
        
    @property
    def init_frames(self):
        return self.models[0].encoder.init_frames
    @property
    def step_frames(self):
        return self.models[0].encoder.step_frames
    
    def get_init_frames(self, wait_block=4):
        return self.init_frames + self.step_frames*(wait_block-1)

    def get_step_frames(self, step_block=1):
        return self.step_frames*step_block
    
    def reorder_states(self, encoder_outs,incremental_states, new_order):
        for model in self.models:
            encoder_outs[model] =model.encoder.reorder_encoder_out(encoder_outs[model], new_order)
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[model],
                new_order
            )
    
    # def reorder_lm_h(self, h_lms, new_order):
    #     out=dict()
    #     for model in self.models:
    #         out[model]= h_lms[model].index_select(0, new_order)
    #     return out
    
    # def cat_lm_h(self, prev_h_lms, h_lms):
    #     out=dict()
    #     for model in self.models:
    #         out[model]= torch.cat([prev_h_lms[model], h_lms[model]], dim=1)
    #     return out
    
    def recalc_logits(self, h_lms, encoder_outs, incremental_states, temperature=1.0):
        log_probs = []
        for i, model in enumerate(self.models):
            logits=model.recalc_logits(h_lms[model],encoder_outs['model'], incremental_states[model])
            logits = logits[:,-1:,:]/temperature
            lprobs = utils.log_softmax(logits, dim=-1)
            lprobs = lprobs[:, -1, :]
            if len(self.models) == 1:
                return lprobs
            log_probs.append(lprobs)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        return avg_probs
    
    def convert_cache_pad(self, incremental_states,new_idx):
        for i, model in enumerate(self.models):
            if hasattr(model.decoder, "convert_cache_pad"):
                model.decoder.convert_cache_pad(incremental_states[model], new_idx)

            else:
                model.decoder.lm.convert_cache_pad(incremental_states[model], new_idx)

    
    def recalc_lm(self, prev_output_tokens, incremental_states, encoder_outs=None,processed_length=0):
        h_lms=dict()
        for i, model in enumerate(self.models):
            if hasattr(model.decoder, "recalc_h"):
                h = model.decoder.recalc_h(
                    prev_output_tokens, encoder_outs[model],
                    incremental_states[model],
                    processed_length
                )
            else:
                h = model.decoder.lm.recalc_h(
                    prev_output_tokens, incremental_states[model], processed_length
                )
            h_lms[model] =h
        return h_lms
    
    def fwd_decoder_step(self, tokens, encoder_outs, incremental_states,temperature=1.0):
        log_probs = []
        h_lms=dict()
        for i, model in enumerate(self.models):
            encoder_out = encoder_outs[model]
            # decode each model
            logits, extra = model.decoder.forward(
                tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states[model],
            )
            h_lms[model] = extra['h_lm']
            logits = logits[:,-1:,:]/temperature
            lprobs = utils.log_softmax(logits, dim=-1)
            lprobs = lprobs[:, -1, :]
            if len(self.models) == 1:
                return lprobs, h_lms
            log_probs.append(lprobs)
           
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        return avg_probs, h_lms
    
    def rollback(self,incremental_states, step_to_keep):
        for model in self.models:
            model.decoder.rollback_steps(incremental_states[model], step_to_keep)

    def init_states(self):
        encoder_outs={m:{} for m in self.models}
        incremental_states={m:{} for m in self.models}
        # w= next(self.parameters())
        # lm_dim = self.models[0].decoder.lm.embed_dim
        # h_lms={m:{torch.Tensor(1,0,lm_dim).to(w.device)} for m in self.models}
        # return encoder_outs, incremental_states,h_lms
        return encoder_outs, incremental_states
    
    def fwd_encoder(
        self, src:Tensor, src_lengths:Tensor,
        encoder_outs: Optional[Dict[nn.Module,Dict[str, List[Tensor]]]] ,
        incremental_states:Optional[Dict[nn.Module, Dict[str, Dict[str, Optional[Tensor]]]]],
        finished = False
    ):
        for model in self.models:
            #offline= model.encoder(src,src_lengths)
           
            curr_out= model.encoder.forward_infer(
                src, src_lengths,
                incremental_state=incremental_states[model],
                finished=finished
            )
           
            if "encoder_out" in encoder_outs[model]:
                pre= encoder_outs[model]["encoder_out"][0]
                pre_mask= encoder_outs[model]["encoder_padding_mask"][0]
                pre_b = pre.shape[1]
                if pre_b >1:
                    new_order= torch.LongTensor(pre_b).fill_(0).to(pre.device)
                    curr_out =model.encoder.reorder_encoder_out(curr_out, new_order)
                encoder_outs[model]["encoder_out"][0] = torch.cat([pre,curr_out["encoder_out"][0]], dim=0)
                encoder_outs[model]["encoder_padding_mask"][0] = torch.cat([pre_mask,curr_out["encoder_padding_mask"][0]], dim=1)
            else:
                encoder_outs[model]=curr_out
        return encoder_outs, incremental_states
    
    
class OnlineSpeechModels(OnlineModels):
    def __init__(self, models: List[FairseqEncoderDecoderModel]):
        super().__init__(models)
        self.reserved_fea = None
    
    def init_states(self):
        encoder_outs={m:{} for m in self.models}
        incremental_states={m:{} for m in self.models}
        self.reserved_fea = None
        return encoder_outs, incremental_states
    
    def fwd_encoder(
        self, src:Tensor, src_lengths:Tensor,
        encoder_outs: Optional[Dict[nn.Module,Dict[str, List[Tensor]]]] ,
        incremental_states:Optional[Dict[nn.Module, Dict[str, Dict[str, Optional[Tensor]]]]],
        finished = False
    ):
        if self.reserved_fea is not None:
            src= torch.cat([self.reserved_fea, src], dim=1)
            src_lengths.fill_(src.shape[1])
       
        for model in self.models:
            from rain.layers.unidirect_encoder import UnidirectAudioTransformerEncoder
            encoder:UnidirectAudioTransformerEncoder = model.encoder
            incremental_state = incremental_states[model]
            def cat_encout(pre_out, curr_out):
                if "encoder_out" in pre_out:
                    pre_out["encoder_out"][0] = torch.cat(
                        [pre_out["encoder_out"][0],curr_out["encoder_out"][0]], dim=0
                    )
                    pre_out["encoder_padding_mask"][0] = torch.cat(
                        [pre_out["encoder_padding_mask"][0],curr_out["encoder_padding_mask"][0]], dim=1
                    )
                    return pre_out
                return curr_out

            if len(incremental_state) == 0:
                if finished and src.shape[1]< encoder.init_frames:
                    encoder_outs[model]= encoder(src,src_lengths, incremental_state= incremental_state, finished= True)
                    continue
                assert src.shape[1] >= encoder.init_frames
                curr_src= src[:,: encoder.init_frames]
                src_lengths.fill_(encoder.init_frames)
                src = src[:, encoder.init_frames:]
                curr_out = encoder(curr_src, src_lengths, incremental_state= incremental_state)
                encoder_outs[model] = cat_encout(encoder_outs[model], curr_out)
            
            while src.shape[1] >0:
                if src.shape[1]<= encoder.step_frames:
                    src_lengths.fill_(src.shape[1])
                    if finished:
                        curr_out= encoder(src, src_lengths, incremental_state= incremental_state,finished= True)
                        encoder_outs[model]= cat_encout(encoder_outs[model], curr_out)
                    elif src.shape[1] == encoder.step_frames:
                        curr_out= encoder(src, src_lengths, incremental_state= incremental_state,finished= False)
                        encoder_outs[model]= cat_encout(encoder_outs[model], curr_out)
                    else:
                        self.reserved_fea = src
                    break
                else:
                    curr_src= src[:,: encoder.step_frames]
                    src = src[:,encoder.step_frames:]
                    src_lengths.fill_(curr_src.shape[1])
                    
                    curr_out= encoder(curr_src, src_lengths, incremental_state= incremental_state,finished= False)
                    encoder_outs[model]= cat_encout(encoder_outs[model], curr_out)
        
        return encoder_outs, incremental_states


class TransducerSearcher(nn.Module):
    def __init__(
        self, models:OnlineModels,
        vocab: Dictionary,
        eos:int=1,
        bos:int=0,
        max_step=100,
        bos_bias= 0,
        len_scale=1.0,
        eager=False
    ):
        super().__init__()
        self.models= models
        self.vocab= vocab
        self.eager=eager
        self.eos= eos
        self.bos= bos
        self.vocab_size= len(vocab)
        self.pad= vocab.pad()
        self.word_end= WordEndChecker(vocab)
        self.max_step= max_step
        self.bos_bias= bos_bias
        self.len_scale=len_scale

    def search(
        self,  src,src_lengths,
        prev_tokens, encoder_outs, 
        incremental_states,
        beam=5,
        is_end=False,
        max_steps=40,
    ):
        ninf= float('-inf')
        
        if src is not None:
            self.models.fwd_encoder(src, src_lengths, encoder_outs, incremental_states, is_end)
        # encoder_out = self.models.models[0].encoder.forward(src,src_lengths)
        # import pdb;pdb.set_trace()
        new_order = prev_tokens.new(beam).fill_(0)
        self.models.reorder_states(encoder_outs, incremental_states, new_order)
        prev_tokens = prev_tokens.view(1,-1).repeat(beam,1)
        init_len= prev_tokens.shape[1]
        finished= prev_tokens.new(beam).fill_(0).bool()
        scores= prev_tokens.new(beam,1).float().fill_(0)
        max_steps = min(max_steps, self.max_step)
      
        for nstep in range(max_steps):
            
            lprobs, _= self.models.fwd_decoder_step(prev_tokens, encoder_outs, incremental_states)
           
            lprobs[:, self.pad] = ninf
            # if is_end, ignore score of bos
            if not is_end:
                lprobs[:,self.eos] = torch.logaddexp( lprobs[:,self.eos],lprobs[:,self.bos] + self.bos_bias)
            
            lprobs[:,self.bos] = ninf
            lprobs[finished, :self.eos]= ninf
            lprobs[finished, self.eos]= 0
            lprobs[finished,self.eos+1:] = ninf
            #lprobs: beam*vocab
            expand_score= scores + lprobs
            if nstep ==0:
                expand_score= expand_score[:1]
            tscore, tidx= expand_score.view(-1).topk(beam)
            next_tokens= tidx %self.vocab_size
            new_order= tidx //self.vocab_size
            
            scores[:]= tscore.unsqueeze(1)
            prev_tokens= prev_tokens.index_select(0, new_order)
            prev_tokens= torch.cat([prev_tokens, next_tokens.unsqueeze(1)], dim=1)
            self.models.reorder_states(encoder_outs, incremental_states, new_order)
            finished= finished | next_tokens.eq(self.eos)
            if finished.all():
                break

        seqlen = prev_tokens.ne(self.eos).float().sum(1) - init_len
        seqlen= seqlen.float()**(self.len_scale)
        scores= scores.squeeze(1)/seqlen
        #import pdb;pdb.set_trace()
        score, idx= scores.max(0)
        new_order= idx.view(1)
        self.models.reorder_states(encoder_outs, incremental_states, new_order)
        prev_tokens= prev_tokens[idx]
        out_tokens= prev_tokens[ init_len:]
        removed = out_tokens.eq(self.eos).sum().item() 
        if self.eager:
            out_words, reserved = self.word_end.string(out_tokens, is_finished=True, removed= removed)
        else:
            out_words, reserved= self.word_end.string(out_tokens, is_finished= is_end, removed= removed)
        
  
       
        rollback_to = len(prev_tokens) - reserved 
        prev_tokens= prev_tokens[:rollback_to]
        self.models.rollback(incremental_states, rollback_to)
        return deque( out_words.split()), prev_tokens
    
    def init_states(self):
        encoder_outs, incremental_states = self.models.init_states()
        prev_tokens = torch.LongTensor([self.bos]).to(next(self.parameters()).device)
        return prev_tokens, encoder_outs, incremental_states
        

class TransducerAgent(Agent):
    def __init__(self, args):
        super().__init__(args)
        self.cpu = args.cpu
        self.step_read_blocks= args.step_read_blocks
        self.beam = args.beam
        self.given_init_frames = args.expected_init_frames
        self.max_len_a= args.max_len_a
        self.max_len_b= args.max_len_b
        
        task_cfg=Namespace(
            task="s2s",
            data= args.train_dir, task_type=args.task_type,
            source_lang=args.slang,
            target_lang= args.tlang, 
            text_config=args.text_encoder, audio_cfg=args.audio_encoder,
            bpe_dropout=0,
        )
        
        self.task= tasks.setup_task(task_cfg)
        self.audio_transformer = self.task.audio_transform_test
        self.text_transformer= self.task.src_encoder

        if args.model_path is None:
            raise ValueError("--model-path needed")
        models, saved_cfg= checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.model_path),
            arg_overrides=None,
            task=self.task,
        )
        self.tgt_dict:Dictionary= self.task.target_dictionary
        self.eos= self.tgt_dict.eos()
        self.bos= self.tgt_dict.eos() if args.infer_bos is None else args.infer_bos
        ensemble_models=None
        # if self.data_type== "speech":
        #     ensemble_models = OnlineSpeechModels(models)
        # else:
        #     ensemble_models= OnlineModels(models)
        #ensemble_models = OnlineSpeechModels(models)
        ensemble_models= OnlineModels(models)
        self.searcher = TransducerSearcher(
            ensemble_models, self.tgt_dict, eos = self.eos, bos= self.bos, bos_bias= args.bos_bias,
            len_scale= args.len_scale, eager=args.eager
        )
        
        if not self.cpu:
            self.searcher.cuda()
        self.searcher.eval()
        self.frames= None
        self.processed_frames=0
        self.processed_units=0
        self.hypos=deque()
        self.prev_tokens, self.encoder_outs, self.incremental_states= None,None,None

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--eager", default=False, action="store_true", help="output words without word end check"
        )
        parser.add_argument(
            "--len-scale", default=1, type=float, help="length scale"
        )
        parser.add_argument(
            "--bos-bias", default=0, type=float, help="bos bias"
        )
        parser.add_argument(
            "--cpu", action= "store_true", help= "use cpu instead of cuda"
        )
        parser.add_argument(
            "--task-type", default="mt", metavar='ttype',
            help='task type :st,mt'
        )
        parser.add_argument(
            "--slang", default="en", metavar='SLANG',
            help='task type :st,mt'
        )
        parser.add_argument(
            "--tlang", default="de", metavar='TLANG',
            help='task type :st,mt'
        )
        parser.add_argument(
            "--infer-bos", default=0,type=int,
            help= "bos for decoding"
        )
        parser.add_argument(
            "--model-path", default=None,type=str,
            help= "path for models used (may be splited by `:`)"
        )
        parser.add_argument(
            "--beam", default=5, type=int,
            help="beam size"
        )
        parser.add_argument(
            "--step-read-blocks", default=4, type=int,
            help="do translation while read each blocks input"
        )
        parser.add_argument(
            "--train-dir", default="exp_data/must_filtered2", type=str,
            help="train dir, for other resource such as dict"
        )
        parser.add_argument(
            "--text-encoder", default="text_cfg", type=str,
            help= "text encoder"
        )
        parser.add_argument(
            "--audio-encoder", default= "audio_cfg", type=str,
            help= "audio-encoder"
        )
        parser.add_argument(
            "--expected-init-frames", default=-1, type=int,
            help= "expect init frames, if negative, ignore and use model encoder"
        )
        parser.add_argument(
            "--max-len-a", default=4, type= float,
            help = "max |T|/|S| ratio, may be count by training data"
        )
        parser.add_argument(
            "--max-len-b", default=0, type=float,
            help="points lower than diagonal"
        )
        return parser
    
    def initialize_states(self, states):
        # we recompute feature at each step, the waste seems to be acceptable
        logger.info(f"new sample,id={states.instance_id}")
        self.input_fea= torch.Tensor(0,80)
        self.processed_frames=0
        self.processed_units=0
        self.prev_tokens, self.encoder_outs, self.incremental_states = self.searcher.init_states()
        self.hypos=deque()
    
    def expected_init_frames(self):
        return max(self.searcher.models.get_init_frames(self.step_read_blocks),self.given_init_frames)
    
    def expected_step_frames(self):
        return self.searcher.models.get_step_frames(self.step_read_blocks)
    
    def expected_init_units(self):
        frames= self.searcher.models.get_init_frames(self.step_read_blocks)
        if self.data_type == "text":
            #return frames
            # we must do sentencepiece and then get its length
            return 1
        elif self.data_type == "speech":
            # units per ms
            #return frames*10 +15
            return frames

    def expected_step_units(self):
        frames= self.searcher.models.get_step_frames(self.step_read_blocks)
        if self.data_type == "text":
            return 1
        elif self.data_type == "speech":
            # units per ms
            #return frames*10
            return frames
    
    def _gen_frames(self, states):
        source= states.source
        if self.data_type == "text":
            src = ' '.join(source)
            subwords =self.text_transformer.encode(src)
            tokens = self.tgt_dict.encode_line(subwords,add_if_not_exist=False, append_eos= False).long()
            #fairseq src seq [w1,w2...wn,eos], no bos
            #tokens = torch.cat((torch.LongTensor([self.tgt_dict.eos()]),tokens), dim=0)
            if states.finish_read():
                tokens= torch.cat((tokens,torch.LongTensor([self.tgt_dict.eos()])), dim=0)
            self.input_fea= tokens
            self.processed_units = len(source)
        elif self.data_type == "speech":
            rate_ms= 16
            if len(source[-1]) <160:
                source=source[:-1]
            new_frames= len(source) - self.input_fea.shape[0]
            if new_frames <= 0:
                return
            if self.input_fea.shape[0] ==0 :
                pre= torch.FloatTensor(1, 15*rate_ms).fill_(0)
                new_src= sum(source[-new_frames:],[])
                new_src= torch.FloatTensor(new_src).unsqueeze(0)
                new_src= torch.cat([pre, new_src], dim=1)
            else:
                new_src= sum(source[-(new_frames+2):],[])
                new_src= new_src[5*rate_ms:]
                new_src = torch.FloatTensor(new_src).unsqueeze(0)
            new_src= new_src*( 2**-15)
            fbank= audio_encoder._get_fbank(new_src, sample_rate= 16000, n_bins=80)

            fbank = self.audio_transformer(fbank)
            self.input_fea= torch.cat([self.input_fea, fbank], dim=0)

            # src2= sum(states.source, [])
            # src2= torch.FloatTensor(src2).unsqueeze(0)
            # head= torch.FloatTensor(1, 15*rate_ms).fill_(0)
            # src2= torch.cat([head,src2], dim=1)
            # src2= src2*( 2**-15)
            # fbk2= audio_encoder._get_fbank(src2, sample_rate= 16000, n_bins=80)
            # fbk2=self.audio_transformer(fbk2)
          
            
            self.processed_units= len(source)
        else:
            raise ValueError(f"unknown data type {self.data_type}")
    
    def policy(self, states):
        if len(self.hypos) >0:
            return WRITE_ACTION
        source= states.source
        
        if (len(source) >=self.expected_init_units() and self.processed_units==0) or \
            (len(source) -self.processed_units >= self.expected_step_units() and self.processed_units >0) or \
            states.finish_read():
            self._gen_frames(states)
        
        if states.finish_read():
            self.infer(states)
        if (self.processed_frames ==0 and len(self.input_fea) >= self.expected_init_frames()) or \
            (self.processed_frames >0 and len(self.input_fea)- self.processed_frames >= self.expected_step_frames()):
            self.infer(states)
            # with torch.autograd.profiler.profile(use_cuda = True) as prof:
            #     self.infer(states)
            # print(prof)
           
        if len(self.hypos) >0:
            return WRITE_ACTION
        else:
            return READ_ACTION
    
    def infer(self,states):
        assert len(self.hypos) ==0
        
        new_frames= len(self.input_fea) - self.processed_frames
        if new_frames >0:
            fea = self.input_fea[-new_frames:]
            fea= fea.unsqueeze(0)
            fea_lengths= fea.new(1).fill_(fea.shape[1])
            if not self.cpu:
                fea=fea.cuda()
                fea_lengths=fea_lengths.cuda()
        else:
            if not states.finish_read():
                print(f"infer with no new frames, finished= {states.finish_read()}")
            fea= None
            fea_lengths=None
        # if self.processed_frames ==0:
        #     expected_step=1
        # else:
        #     expected_step = max(new_frames //self.expected_step_frames(),1)
        max_steps= int(self.max_len_a* self.input_fea.shape[0] - self.max_len_b - self.prev_tokens.shape[0])
       
        if states.finish_read():
            max_steps= 100
        if max_steps <=0:
            return
        with torch.no_grad():
            out_words, tokens= self.searcher.search(
                fea, fea_lengths,
                self.prev_tokens,self.encoder_outs,
                self.incremental_states, beam=self.beam,
                is_end = states.finish_read(),
                max_steps=max_steps
            )
        self.prev_tokens= tokens
        
        self.processed_frames = len(self.input_fea)
        if states.finish_read():
            out_words.append(DEFAULT_EOS)
        self.hypos.extend(out_words)

    def predict(self, states):
        assert(len(self.hypos) >0)
        return self.hypos.popleft()