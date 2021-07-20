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
import logging
logger = logging.getLogger('waitk.agent')


class WordEndChecker(object):
    def __init__(self, vocab:Dictionary):
        self.vocab= vocab
        self.wordbegin=[]
        for i in range(len(self.vocab)):
            self.wordbegin.append(self.is_beginning_of_word(self.vocab[i]))
    
    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            return True
        return x.startswith("\u2581")
    
    def string(self,tokens, is_finished=False, removed=0):
        tokens_cpu= tokens.cpu()
        
        tnum= len(tokens_cpu)
        end_pos = tnum -removed
        if is_finished:
            out_str = self.vocab.string(tokens_cpu[:end_pos],bpe_symbol="sentencepiece",)
            return out_str, removed
        next_bow= 0
        for i in range(min(end_pos+1,tnum)):
            if self.wordbegin[tokens_cpu[i]] :
                next_bow=i
        out_str= self.vocab.string(tokens_cpu[:next_bow],bpe_symbol="sentencepiece",)
        return out_str, tnum - next_bow

class OnlineSearcher(nn.Module):
    def __init__(
        self, models:List[FairseqEncoderDecoderModel],
        vocab,
        eos=1,
        bos=1,
        eager=False,
        stop_early=False
    ):
        super().__init__()
        self.models= nn.ModuleList(models)
        self.bos=bos
        self.eos= eos
        self.reserve_step = 0
        self.vocab = vocab
        self.vocab_size= len(vocab)
        self.pad= vocab.pad()
        self.word_end= WordEndChecker(vocab)
        self.eager= eager
        self.stop_early= stop_early

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
    
    def search(
        self,  src,src_lengths,
        prev_tokens, encoder_outs, 
        incremental_states,beam=5, fwd_step= 1, forecast_step = 1,
        is_end=False
    ):
        ninf= float('-inf')
        if src is not None:
            self.fwd_encoder(src, src_lengths, encoder_outs, incremental_states, is_end)
        steps= self.reserve_step + fwd_step + forecast_step
        
        if is_end:
            steps= 100
        new_order = prev_tokens.new(beam).fill_(0)
        self.reorder_states(encoder_outs, incremental_states, new_order)
        
        prev_tokens = prev_tokens.repeat(beam,1)
        init_len= prev_tokens.shape[1]
        finished= prev_tokens.new(beam).fill_(0).bool()
        scores= prev_tokens.new(beam,1).float().fill_(0)
        for nstep in range(steps):
            
            lprobs= self.fwd_decoder_step(prev_tokens, encoder_outs, incremental_states)
            # if not is_end:
            #     lprobs[:, self.eos] = ninf
            lprobs[:, self.pad] = ninf
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
            self.reorder_states(encoder_outs, incremental_states, new_order)
            finished= finished | next_tokens.eq(self.eos)
            if finished.all():
                break
        
        if not is_end and self.stop_early and prev_tokens[0][-1-forecast_step] == self.eos:
           
            prev_tokens= prev_tokens[0]
            out_tokens= prev_tokens[ init_len:]
            removed = out_tokens.eq(self.eos).sum().item()
            out_words, reserved = self.word_end.string(out_tokens, is_finished=True, removed= removed)
            return deque( out_words.split()), prev_tokens,True

        seqlen = prev_tokens.ne(self.eos).float().sum(1) - init_len+1
        
        scores= scores.squeeze(1)/seqlen
        score, idx= scores.max(0)
        new_order= idx.view(1)
        self.reorder_states(encoder_outs, incremental_states, new_order)
        prev_tokens= prev_tokens[idx]
        out_tokens= prev_tokens[ init_len:]
        removed = out_tokens.eq(self.eos).sum().item()
        
        ignore_length= max(removed, forecast_step -(steps- len(out_tokens))) if not is_end else removed
        
        if self.eager:
            out_words, reserved = self.word_end.string(out_tokens, is_finished=True, removed= ignore_length)
        else:
            out_words, reserved= self.word_end.string(out_tokens, is_finished= is_end, removed= ignore_length)
       
        rollback_to = len(prev_tokens) - reserved 
        prev_tokens= prev_tokens[:rollback_to]
        self.rollback(incremental_states, rollback_to)
        self.reserve_step = reserved - forecast_step + (steps-len(out_tokens))
       
        assert is_end or self.reserve_step >=0
        return deque( out_words.split()), prev_tokens,False
    
    def fwd_decoder_step(self, tokens, encoder_outs, incremental_states,temperature=1.0):
        log_probs = []
        for i, model in enumerate(self.models):
            encoder_out = encoder_outs[model]
            # decode each model
            logits,_ = model.decoder.forward(
                tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states[model],
                attn_mask=False
            )
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
    
    def rollback(self,incremental_states, step_to_keep):
        for model in self.models:
            model.decoder.rollback_steps(incremental_states[model], step_to_keep)

    def init_states(self):
        encoder_outs={m:{} for m in self.models}
        incremental_states={m:{} for m in self.models}
        self.reserve_step= 0
        return encoder_outs, incremental_states
    
    def fwd_encoder(
        self, src:Tensor, src_lengths:Tensor,
        encoder_outs: Optional[Dict[nn.Module,Dict[str, List[Tensor]]]] ,
        incremental_states:Optional[Dict[nn.Module, Dict[str, Dict[str, Optional[Tensor]]]]],
        finished = False
    ):
        for model in self.models:
            curr_out= model.encoder(
                src, src_lengths,
                incremental_state=incremental_states[model],
                finished=finished
            )
            if "encoder_out" in encoder_outs[model]:
                pre= encoder_outs[model]["encoder_out"][0]
                pre_mask= encoder_outs[model]["encoder_padding_mask"][0]
                encoder_outs[model]["encoder_out"][0] = torch.cat([pre,curr_out["encoder_out"][0]], dim=0)
                encoder_outs[model]["encoder_padding_mask"][0] = torch.cat([pre_mask,curr_out["encoder_padding_mask"][0]], dim=1)
            else:
                encoder_outs[model]=curr_out
        return encoder_outs, incremental_states

class NaiveWaitk(OnlineSearcher):
    def __init__(
        self, models:List[FairseqEncoderDecoderModel],
        vocab,
        eos=1,
        bos=1,
        eager=False,
        stop_early=False
    ):
        super().__init__(models, vocab, eos, bos)
        self.eager=eager
        self.reserved_subwords=None
        self.stop_early=stop_early
    
    def init_states(self):
        encoder_outs, incremental_states = super().init_states()
        self.reserved_subwords = None
        return encoder_outs, incremental_states
    
    def search(
        self,  src,src_lengths,
        prev_tokens, encoder_outs, 
        incremental_states,beam=5, fwd_step= 1, forecast_step = 1,
        is_end=False
    ):
        forecast_step=0
        ninf= float('-inf')
        if src is not None:
            self.fwd_encoder(src, src_lengths, encoder_outs, incremental_states, is_end)
        
        steps=fwd_step
        if is_end:
            steps= 40
        new_order = prev_tokens.new(beam).fill_(0)
        self.reorder_states(encoder_outs, incremental_states, new_order)
        
        prev_tokens = prev_tokens.repeat(beam,1)
        init_len= prev_tokens.shape[1]
        finished= prev_tokens.new(beam).fill_(0).bool()
        scores= prev_tokens.new(beam,1).float().fill_(0)
        for nstep in range(steps):
            
            lprobs= self.fwd_decoder_step(prev_tokens, encoder_outs, incremental_states)
            # if not is_end:
            #     lprobs[:, self.eos] = ninf
            lprobs[:, self.pad] = ninf
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
            self.reorder_states(encoder_outs, incremental_states, new_order)
            finished= finished | next_tokens.eq(self.eos)
            if finished.all():
                break
        if not is_end and self.stop_early and prev_tokens[0][-1] == self.eos:
            prev_tokens= prev_tokens[0]
            out_tokens= prev_tokens[ init_len:]
            if self.reserved_subwords is not None and len(self.reserved_subwords) >0:
                out_tokens = torch.cat((self.reserved_subwords, out_tokens),dim=0)
            removed = out_tokens.eq(self.eos).sum()
            ignore_length= removed
            out_words, reserved = self.word_end.string(out_tokens, is_finished=True, removed= ignore_length)
            return deque( out_words.split()), prev_tokens,True
        seqlen = prev_tokens.ne(self.eos).float().sum(1) - init_len+1
        
        scores= scores.squeeze(1)/seqlen
        score, idx= scores.max(0)
        new_order= idx.view(1)
        self.reorder_states(encoder_outs, incremental_states, new_order)
        prev_tokens= prev_tokens[idx]
        out_tokens= prev_tokens[ init_len:]
        if self.reserved_subwords is not None and len(self.reserved_subwords) >0:
            out_tokens = torch.cat((self.reserved_subwords, out_tokens),dim=0)
        removed = out_tokens.eq(self.eos).sum()
        ignore_length= removed
        if self.eager:
            out_words, reserved = self.word_end.string(out_tokens, is_finished=True, removed= ignore_length)
        else:
            out_words, reserved= self.word_end.string(out_tokens, is_finished= is_end, removed= ignore_length)
        

        #out_words, reserved= self.word_end.string(out_tokens, is_end)
        self.reserved_subwords=None
        if reserved >0:
            self.reserved_subwords = prev_tokens[-reserved:]
        return deque( out_words.split()), prev_tokens,False




class WaitkAgent(Agent):
    def __init__(self, args):
        self._set_default_args(args)
        super().__init__(args)
        utils.import_user_module("rain")
        self.cpu = args.cpu
        self.wait_blocks= args.wait_blocks
        self.step_read_blocks = args.step_read_blocks
        self.step_generate = args.step_generate
        self.step_forecast = args.step_forecast
        self.beam = args.beam
        
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
        self.src_dict= self.task.src_dict

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
        # if self.data_type == "speech":
        #     self.searcher= SpeechSearcher(models, self.tgt_dict, eos= self.eos, bos= self.bos)
        if args.naive_waitk:
            self.searcher = NaiveWaitk(
                models, self.tgt_dict, eos= self.eos, bos= self.bos, eager= args.eager,
                stop_early=args.stop_early
            )
        else:
            self.searcher = OnlineSearcher(
                models, self.tgt_dict, eos= self.eos, bos= self.bos, eager= args.eager,
                stop_early=args.stop_early
            )
        if not self.cpu:
            self.searcher.cuda()
        self.searcher.eval()
        self.frames= None
        self.finished=True
        self.processed_frames=0
        self.processed_units=0
        self.hypos=deque()
        self.prev_tokens, self.encoder_outs, self.incremental_states= None,None,None


    def _set_default_args(self, args):
        args.wait_blocks= 4 if args.wait_blocks is None else args.wait_blocks
        args.step_read_blocks=1 if args.step_read_blocks is None else args.step_read_blocks
        args.step_generate=1 if args.step_generate is None else args.step_generate
        args.step_forecast= 0 if args.step_forecast is None else args.step_forecast

    
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--stop-early', action='store_true', help='stop early mode for waitk'
        )
        parser.add_argument(
            "--cpu", action= "store_true", help= "use cpu instead of cuda"
        )
        parser.add_argument(
            "--eager", default=False, action="store_true", help="output words without word end check"
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
            "--infer-bos", default=None,type=int,
            help= "bos for decoding"
        )
        parser.add_argument(
            "--model-path", default=None,type=str,
            help= "path for models used (may be splited by `:`)"
        )
        parser.add_argument(
            "--wait-blocks", default=None, type=int,
            help="start translation after wait_blocks samples read, default None and we may set its default in our class"
        )
        parser.add_argument(
            "--beam", default=5, type=int,
            help="beam size"
        )
        parser.add_argument(
            "--step-read-blocks", default=None, type=int,
            help="do translation while read each blocks input"
        )
        parser.add_argument(
            "--step-generate", default=None, type=int,
            help="generate tokens for each step, NOT equal to output words, if current step output un-complete subwords, "
            "reserve to output in next step"
        )
        parser.add_argument(
            "--step-forecast", default= None, type=int,
            help="forecast subword numbers for each step, only for search"
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
            "--naive-waitk", action="store_true",help="use naive waitk"
        )
        return parser
    
    def initialize_states(self, states):
        # we recompute feature at each step, the waste seems to be acceptable
        logger.info(f"new sample,id={states.instance_id}")
        self.input_fea= torch.Tensor(0,80)
        self.finished=False
        self.processed_frames=0
        self.processed_units=0
        self.encoder_outs, self.incremental_states = self.searcher.init_states()
        self.prev_tokens= torch.LongTensor([self.bos])
        if not self.cpu:
            self.prev_tokens= self.prev_tokens.cuda()
        self.hypos=deque()
    
    def expected_init_frames(self):
        return self.searcher.get_init_frames(self.wait_blocks)
    
    def expected_step_frames(self):
        return self.searcher.get_step_frames(self.step_read_blocks)
    
    def expected_init_units(self):
        frames= self.searcher.get_init_frames(self.wait_blocks)
        if self.data_type == "text":
            #return frames
            # we must do sentencepiece and then get its length
            return 1
        elif self.data_type == "speech":
            # units per 10ms
            return frames

    def expected_step_units(self):
        frames= self.searcher.get_step_frames(self.step_read_blocks)
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
            tokens =  self.src_dict.encode_line(subwords,add_if_not_exist=False, append_eos= False).long()
            #tokens = torch.cat((torch.LongTensor([self.tgt_dict.eos()]),tokens), dim=0)
            if states.finish_read():
                tokens= torch.cat((tokens,torch.LongTensor([ self.src_dict.eos()])), dim=0)
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
            
            self.processed_units= len(source)
        else:
            raise ValueError(f"unknown data type {self.data_type}")
    
    def policy(self, states):
        if len(self.hypos) >0:
            return WRITE_ACTION
        source= states.source
        if self.finished:
            if len(self.hypos) >0:
                return WRITE_ACTION
            else:
                return READ_ACTION
        
        if (len(source) >=self.expected_init_units() and self.processed_units==0) or \
            (len(source) -self.processed_units >= self.expected_step_units() and self.processed_units >0) or \
            states.finish_read():
            self._gen_frames(states)
        
        if states.finish_read():
            self.infer(states)
        if (self.processed_frames ==0 and len(self.input_fea) >= self.expected_init_frames()) or \
            (self.processed_frames >0 and len(self.input_fea)- self.processed_frames >= self.expected_step_frames()):
            self.infer(states)
        if len(self.hypos) >0:
            return WRITE_ACTION
        else:
            return READ_ACTION
    
    def infer(self,states):
        assert len(self.hypos) ==0
        
        new_frames= len(self.input_fea) - self.processed_frames
        
        #assert new_frames >0
        if new_frames >0:
            fea = self.input_fea[-new_frames:]
            fea= fea.unsqueeze(0)
            fea_lengths= fea.new(1).fill_(fea.shape[1])
            if not self.cpu:
                fea= fea.cuda()
                fea_lengths=fea_lengths.cuda()
        else:
            print(f"infer with no new frames, finished= {states.finish_read()}")
            fea= None
            fea_lengths=None
        if self.processed_frames ==0:
            expected_step=1
        else:
            expected_step = max(new_frames //self.expected_step_frames(),1)
        with torch.no_grad():
            out_words, tokens, eos_found= self.searcher.search(
                fea, fea_lengths,
                self.prev_tokens,self.encoder_outs,
                self.incremental_states, beam=self.beam,
                fwd_step= self.step_generate*expected_step,
                forecast_step= self.step_forecast,
                is_end = states.finish_read(),
            )
        self.prev_tokens= tokens
            
        # if states.finish_read():
            
        #     print(f"target:{self.searcher.vocab.string(self.prev_tokens)}")
            
        #     # run whole data offline
        #     model= self.searcher.models[0]
        #     def sub_encoder(fea, incremental_state= None):
        #         flen = fea.new(1).fill_(fea.shape[1]).cuda()
        #         enc_out = model.encoder(fea, flen, incremental_state = incremental_state)
        #         return enc_out["encoder_out"][0]
            
        #     fea = self.input_fea
        #     fea= fea.unsqueeze(0).cuda()
        #     import pdb;pdb.set_trace()
            
        #     fea_lengths= fea.new(1).fill_(fea.shape[1]).cuda()
        #     encoder_out= model.encoder(fea, fea_lengths, finished=True)
            
        #     prev_tokens= torch.LongTensor([self.bos]).cuda().unsqueeze(0)
        #     incremental_state={}
            
        #     for step in range(10):
        #         logits, _= model.decoder.forward(
        #             prev_tokens,
        #             encoder_out=encoder_out,
        #             incremental_state=incremental_state,
        #             attn_mask=False
        #         )
        #         lprobs= utils.log_softmax(logits, dim=-1)
        #         v, next_token= lprobs.max(-1)
        #         prev_tokens= torch.cat((prev_tokens, next_token),dim=1)
        #     import pdb;pdb.set_trace()
        #     print(f"new:{self.searcher.vocab.string(prev_tokens)}")
        self.processed_frames = len(self.input_fea)
        if states.finish_read():
            out_words.append(DEFAULT_EOS)
        if eos_found:
            out_words.append(DEFAULT_EOS)
            self.finished=True
        self.hypos.extend(out_words)

    def predict(self, states):
        assert(len(self.hypos) >0)
        return self.hypos.popleft()






