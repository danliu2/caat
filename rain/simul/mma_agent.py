import sys
import os
sys.path.append(os.getcwd())
from simuleval.agents import Agent,TextAgent, SpeechAgent
from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
from typing import List,Dict, Optional
import numpy as np
import math
import torch.nn.functional as F
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
logger = logging.getLogger('mma.agent')

class MMASearcher(nn.Module):
    def __init__(self, model, vocab, eos=1, eager=False, stop_early=False):
        super().__init__()
        self.model= model
        self.vocab= vocab
        self.word_end= WordEndChecker(vocab)
        self.reserved=[]
        self.eos=vocab.eos()
        self.eager= eager
        self.stop_early=stop_early
    
    @property
    def init_frames(self):
        return self.model.encoder.init_frames
    
    @property
    def step_frames(self):
        return self.model.encoder.step_frames
    
    def init_states(self):
        encoder_outs={}
        incremental_states={}
        self.reserved=[]
        return encoder_outs, incremental_states
    
    def search(self, prev_tokens, encoder_out, incremental_state, is_end=False):
        incremental_state["steps"] = {
            "src":encoder_out["encoder_out"][0].size(0),
            "tgt":prev_tokens.shape[1]
        }
        incremental_state["online"] = {"only":torch.tensor(not is_end)}
        max_length =100
       
        for i in range(max_length):
            x, outputs = self.model.decoder.forward(
                prev_output_tokens=prev_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
            if not self.stop_early and not is_end:
                x[...,self.eos] = -1e10
            if outputs.action == 0:
                break
            lprobs = F.log_softmax(x, dim=-1)
            next_token= lprobs.argmax(dim=-1)
            prev_tokens= torch.cat((prev_tokens, next_token), dim=-1)
            ntoken= next_token[0,0].item()
            self.reserved.append(ntoken)
            if ntoken == self.eos:
                break
        if len(self.reserved) ==0:
            return deque(), prev_tokens, False
        if self.reserved[-1] == self.eos:
            self.reserved= self.reserved[:-1]
            out_words, reserved= self.word_end.string(torch.tensor(self.reserved), is_finished= True,)
            #print(f"output: {out_words}")
            return deque( out_words.split()),prev_tokens,True
        if self.eager:
            
            out_words, reserved= self.word_end.string(torch.tensor(self.reserved), is_finished= True,)
        else:
            out_words, reserved= self.word_end.string(torch.tensor(self.reserved), is_finished= is_end,)
        if reserved >0:
            self.reserved= self.reserved[-reserved:]
        else:
            self.reserved = []
        #print(f"output: {out_words}")
        return deque( out_words.split()), prev_tokens,False
    
    def fwd_encoder(
        self, src:Tensor, src_lengths:Tensor,
        encoder_outs: Dict[str, List[Tensor]] ,
        incremental_states:Dict[str, Dict[str, Optional[Tensor]]],
        finished = False
    ):
        model= self.model
        curr_out= model.encoder(
            src, src_lengths,
            incremental_state=incremental_states,
            finished=finished
        )
       
        if "encoder_out" in encoder_outs:
            pre= encoder_outs["encoder_out"][0]
            pre_mask= encoder_outs["encoder_padding_mask"][0]
            encoder_outs["encoder_out"][0] = torch.cat([pre,curr_out["encoder_out"][0]], dim=0)
            encoder_outs["encoder_padding_mask"][0] = torch.cat([pre_mask,curr_out["encoder_padding_mask"][0]], dim=1)
        else:
            encoder_outs.update(curr_out)
        return encoder_outs, incremental_states



class MMAAgent(Agent):
    data_type= "speech"
    speech_segment_size = 10
    def _set_default_args(self, args):
        args.wait_blocks= 32 if args.wait_blocks is None else args.wait_blocks
        args.step_read_blocks=1 if args.step_read_blocks is None else args.step_read_blocks
        args.step_generate=1 if args.step_generate is None else args.step_generate
        args.step_forecast= 0 if args.step_forecast is None else args.step_forecast

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
        self.eager= args.eager
        self.stop_early=args.stop_early
        self.tgt_dict:Dictionary= self.task.target_dictionary
        self.eos= self.tgt_dict.eos()
        self.bos= self.tgt_dict.eos() if args.infer_bos is None else args.infer_bos
        # if self.data_type == "speech":
        #     self.searcher= SpeechSearcher(models, self.tgt_dict, eos= self.eos, bos= self.bos)
        self.model= MMASearcher(models[0], self.tgt_dict, self.eos, eager= self.eager, stop_early= self.stop_early)
        
        if not self.cpu:
            self.model.cuda()
        self.model.eval()
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
        self.encoder_outs, self.incremental_states = self.model.init_states()
        self.prev_tokens= torch.LongTensor([self.bos]).view(1,1)
        if not self.cpu:
            self.prev_tokens= self.prev_tokens.cuda()
        self.hypos=deque()
    
    def expected_init_frames(self):
        init_frames= self.model.init_frames
        step_frames= self.model.step_frames
        return init_frames + (self.step_read_blocks-1)*step_frames
    
    def expected_step_frames(self):
        step_frames= self.model.step_frames
        return step_frames* self.step_read_blocks
    
    def expected_init_units(self):
        frames= self.expected_init_frames()
        if self.data_type == "text":
            #return frames
            # we must do sentencepiece and then get its length
            return 1
        elif self.data_type == "speech":
            # units per 10ms
            return frames

    def expected_step_units(self):
        frames= self.expected_step_frames()
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
            if fea is not None:
                self.encoder_outs, self.incremental_states = self.model.fwd_encoder(
                    fea,fea_lengths, self.encoder_outs,
                    self.incremental_states, finished= states.finish_read()
                )
            
            out_words, tokens, finished_write= self.model.search(self.prev_tokens, self.encoder_outs,self.incremental_states, is_end=states.finish_read())
            
        self.prev_tokens= tokens
            
        self.processed_frames = len(self.input_fea)
        if finished_write:
            out_words.append(DEFAULT_EOS)
            self.finished=True

        if states.finish_read():
            out_words.append(DEFAULT_EOS)
        
        self.hypos.extend(out_words)

    def predict(self, states):
        assert(len(self.hypos) >0)
        return self.hypos.popleft()