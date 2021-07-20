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
import copy
logger = logging.getLogger('transducer.agent')

from .transducer_agent import OnlineModels


class FullTransducerSearcher(nn.Module):
    def __init__(
        self, models:OnlineModels,
        vocab: Dictionary,
        eos:int=1,
        bos:int=0,
        max_step=100,
        bos_bias= 0,
        len_scale=1.0,
        len_penalty=0.,
        merge_add=False,
        eager=False
    ):
        """
            segment_rescore: recalculate score in pre segment, logadd it
        """
        super().__init__()
        self.models= models
        self.vocab= vocab
        self.eos= eos
        self.bos= bos
        self.vocab_size= len(vocab)
        self.pad= vocab.pad()
        self.word_end= WordEndChecker(vocab)
        self.max_step= max_step
        self.bos_bias= bos_bias
        self.len_scale=len_scale
        self.len_penalty=len_penalty
        self.merge_add=merge_add
        self.eager=eager
        # tokens before out_token_pos has processed (set to 1 to ignore blank)
        self.out_token_pos=1

    def init_states(self):
        encoder_outs, incremental_states = self.models.init_states()
        w= next(self.parameters())
        prev_tokens = torch.LongTensor([[self.bos]]).to(w.device)
        prev_scores = torch.Tensor(1).to(w.device).fill_(0)
        self.out_token_pos=1
        return prev_tokens, prev_scores, encoder_outs, incremental_states
    
    def norm_score(self, score, lengths, is_end= False):
        len_penalty = self.len_penalty if not is_end else 0
        #len_penalty = self.len_penalty 
        score= score*lengths**(-self.len_scale) - lengths*len_penalty
        return score

    def unnorm_score(self, score, lengths, is_end=False):
        len_penalty = self.len_penalty if not is_end else 0
        # len_penalty = self.len_penalty 
        score = (score+lengths*len_penalty)*lengths**(self.len_scale)
        return score
    
    def _set_encoder_mask(self, encoder_outs, seen =-1):
        for model, encoder_out in encoder_outs.items():
            if seen <0 :
                encoder_out['encoder_padding_mask'][0][:]=0
            else:
                encoder_out['encoder_padding_mask'][0][:,:seen]=0
                encoder_out['encoder_padding_mask'][0][:,seen:]= 1
    
    # def _backup_incremental_states(self, incremental_states):
    #     backup = []
    #     for _,states in incremental_states.items():
    #         backup.append(copy.deepcopy(states))
    #     return backup

    # def _restore_incremental_states(self, incremental_states, backup):
    #     for i, key in enumerate(incremental_states.keys()):
    #         incremental_states[key] = backup[i]
    #     return incremental_states

    def emit_words(self, prev_tokens, is_end=False):
        prev_tokens = utils.convert_padding_direction(prev_tokens,self.pad, left_to_right=True)
        if is_end:
            tokens= utils.strip_pad(prev_tokens[0], self.pad)
            to_out= tokens[self.out_token_pos:]
            out_words,reserved = self.word_end.string(to_out,is_finished=True, removed=0)
            self.out_token_pos= tokens.shape[0]
            return deque( out_words.split())
        
        lengths= prev_tokens.ne(self.pad).sum(1)
        if prev_tokens.shape[0] == 1:
            ident_pos= lengths[0]
        else:
            neq_pos = prev_tokens.ne(prev_tokens[:1]).any(dim=0).long()
            neq_pos = torch.cumsum(neq_pos,0)
        
            ident_pos = (neq_pos.eq(0)) & (prev_tokens[0].ne(self.pad))
            ident_pos= ident_pos.sum()
        # if ident_pos < self.out_token_pos:
        #     import pdb;pdb.set_trace()
        assert ident_pos >= self.out_token_pos, \
            f"current disambiguation pos {ident_pos} smaller than prev out {self.out_token_pos}"
        if self.eager:
            # out_words= self.vocab.string(prev_tokens[0,self.out_token_pos:ident_pos], bpe_symbol="sentencepiece")
            out_words, _= self.word_end.string(prev_tokens[0,self.out_token_pos:ident_pos],is_finished=True )
            self.out_token_pos = ident_pos
        else:
            end = ident_pos +1 if ident_pos <lengths[0].item() and prev_tokens[0, ident_pos].item() != self.pad else ident_pos
            out_words, reserved = self.word_end.string(prev_tokens[0,self.out_token_pos:end], removed = end-ident_pos)
            self.out_token_pos =end - reserved
        return deque( out_words.split())
    
    def search(
        self,  src,src_lengths,
        prev_tokens, prev_scores,
        encoder_outs, 
        incremental_states,
        intra_beam=5,
        inter_beam=5,
        gen_beam=2,
        read_step =1,
        is_end=False,
        max_steps=40,
    ):
        """
            token_beam: max paths to keep
            gen_beam: beam search until best_finished -best_unfinished >gen_beam
            pre_block_gen: max generated token number in previous block,

        """
        ninf= float('-inf')
        model0= self.models.models[0]
        src_processed= (0 
            if "encoder_out" not in encoder_outs[model0] 
            else encoder_outs[model0]["encoder_out"][0].shape[0]
        )
        if src is not None:
            self.models.fwd_encoder(src, src_lengths, encoder_outs, incremental_states, is_end)
        new_src_len = encoder_outs[model0]["encoder_out"][0].shape[0] - src_processed
      
        if new_src_len == 0:
            assert is_end==True, "input empty while not ended"
            self._set_encoder_mask(encoder_outs, -1)
            prev_tokens,prev_scores = self.search_at(
                encoder_outs, prev_tokens,
                prev_scores, incremental_states,
                beam_size=intra_beam,
                gen_beam=gen_beam,
                max_steps=max_steps,
                is_end=True
            )
        else:
            blocks= max(new_src_len //read_step,1)
            for i in range(blocks):
                seen =(i+1)*read_step if i <blocks -1 else new_src_len
                ended= is_end & (seen == new_src_len)
                src_seen = min(seen, new_src_len) + src_processed
                self._set_encoder_mask(encoder_outs, src_seen)
                # recalculate pre block
                prev_tokens,prev_scores = self.search_at(
                    encoder_outs, prev_tokens,
                    prev_scores, incremental_states,
                    beam_size=intra_beam,
                    gen_beam=gen_beam,
                    max_steps=max_steps,
                    is_end=ended
                )
        # remove path lower than gen_beam, keep at most inter_beam
        prev_scores= self._merge_bpe(prev_tokens, prev_scores, self.merge_add)
        lengths= prev_tokens.ne(self.pad).sum(1).float()
        normed_score= self.norm_score(prev_scores, lengths, is_end= is_end)
        # normed_score= prev_scores/(lengths**(self.len_scale))
        
        gen_keep = normed_score> normed_score[0]- gen_beam
        bidx= torch.arange(prev_scores.shape[0]).to(prev_tokens)
        kept= gen_keep & (bidx <inter_beam)
        sel_idx= bidx[kept]
        prev_tokens= prev_tokens.index_select(0, sel_idx)
        prev_scores= prev_scores.index_select(0, sel_idx)
        self.models.reorder_states(encoder_outs, incremental_states, sel_idx)
        
        out_words= self.emit_words(prev_tokens, is_end= is_end)
        #print(f"output:{' '.join(out_words)}")
        return prev_tokens, prev_scores,out_words
    
    def _merge_bpe(self, finished_tokens, finished_score, add_reduce=False):
        ninf= float('-inf')
       
        sents= [self.vocab.string(sent[sent.ne(self.pad)],bpe_symbol='sentencepiece') for sent in finished_tokens]
        ident = [ [s2==s1 for s2 in sents] for s1 in sents]
        ident_mat= torch.BoolTensor(ident).to(finished_tokens.device)
        ident_triu= torch.triu(ident_mat)
        redundency = torch.triu(ident_mat,1).any(dim=0)
        score_mat = finished_score.unsqueeze(0).repeat(len(finished_score), 1)
        score_mat = score_mat.masked_fill(~ident_triu,ninf)
        if add_reduce:
            score = torch.logsumexp(score_mat, dim=1)
        else:
            score,_= torch.max(score_mat,dim=1)
        score[redundency] = ninf
        return score

    
    def merge_paths(self, finished_tokens, finished_score, new_num, add_reduce=False):
        ninf= float('-inf')
        lpad_tokens,_= self._to_left_pad(finished_tokens, new_num)
        ident_mat = torch.all(lpad_tokens.unsqueeze(1) == lpad_tokens.unsqueeze(0),-1)
        ident_triu= torch.triu(ident_mat)
        redundency = torch.triu(ident_mat,1).any(dim=0)
        score_mat = finished_score.unsqueeze(0).repeat(len(finished_score), 1)
        score_mat = score_mat.masked_fill(~ident_triu,ninf)
        if add_reduce:
            score = torch.logsumexp(score_mat, dim=1)
        else:
            score,_= torch.max(score_mat,dim=1)
        score[redundency] = ninf
        return score

    def search_at(
        self, encoder_outs, 
        prev_tokens,
        prev_scores,
        incremental_states,
        beam_size=5,
        gen_beam=2,
        max_steps=40,
        is_end=False
    ):
        """
            beam search for given encoder_outs (frames of encoder)
            Args:
            prev_tokens: B*T, B is paths reserved inter-frame, may be smaller than token_beam
            prev_scores: B, 
            pre_block_gen: max token number generated in previous routine
        """
        ninf= float('-inf')
        
        prev_len = prev_tokens.shape[1]
        finished_tokens= prev_tokens.new(beam_size*2, prev_len + max_steps).fill_(self.pad)
        finished_score= prev_scores.new(beam_size*2).fill_(ninf)
        lengths = prev_tokens.ne(self.pad).sum(1).float()-1
       
        #backup = self._backup_incremental_states(incremental_states)
        #import pdb;pdb.set_trace()
        for nstep in range(max_steps):
            B,T=prev_tokens.shape
            #print(f'step={nstep}')
            lprobs, h_lms = self.models.fwd_decoder_step(prev_tokens, encoder_outs, incremental_states)
            lprobs[:, self.pad] = ninf
            #print(f"step={nstep}, bos={lprobs[0,0]}")
            if not is_end:
                #lprobs[:,self.eos] = torch.logaddexp( lprobs[:,self.eos],lprobs[:,self.bos] + self.bos_bias)
                lprobs[:,self.eos]= lprobs[:,self.bos] + self.bos_bias
            
            lprobs[:,self.bos] = ninf
            lengths +=1
            blank_score= prev_scores + lprobs[:,self.eos]
            blank_score= self.norm_score(blank_score, lengths, is_end=is_end)
            finished_score[-B:]=blank_score
            finished_tokens[-B:,:T]=prev_tokens
            if T>prev_len:
                
                finished_score= self.merge_paths(
                    finished_tokens[:,:T], finished_score, T-prev_len, 
                    add_reduce=self.merge_add
                )
            fin_score, fin_idx= torch.sort(finished_score, descending=True)
            finished_score= fin_score
            finished_tokens= finished_tokens.index_select(0, fin_idx)
            lprobs[:,self.eos]= ninf
            #B*V
            
            expand_score= prev_scores.unsqueeze(1) + lprobs
            normed_score= self.norm_score(expand_score, lengths.unsqueeze(1), is_end=is_end)
            # normed_score= expand_score/(lengths.unsqueeze(1)**self.len_scale)
            tscore, tidx= normed_score.view(-1).topk(beam_size)
            next_tokens= tidx %self.vocab_size
            new_order= tidx //self.vocab_size
            prev_tokens= prev_tokens.index_select(0, new_order)
            lengths= lengths.index_select(0, new_order)
            prev_tokens= torch.cat([prev_tokens, next_tokens.unsqueeze(1)], dim=1)
            prev_scores= expand_score.view(-1).index_select(0, tidx)
            self.models.reorder_states(encoder_outs, incremental_states, new_order)
            
            max_finished= finished_score[0]
            max_unfinished=tscore[0]
            if max_finished -gen_beam > max_unfinished:
                break
        

        finished_score= finished_score[:beam_size]
        finished_tokens= finished_tokens[:beam_size]
        kept= finished_score >finished_score[0] - gen_beam
        
        if not kept.all():
            kept_idx= torch.arange(kept.shape[0]).to(kept.device)[kept]
            self.models.reorder_states(encoder_outs, incremental_states, kept_idx)
            finished_score= finished_score[kept]
            finished_tokens= finished_tokens[kept]
        
        right_pad_num = finished_tokens[:,prev_len:].eq(self.pad).all(dim=0).sum()
        if right_pad_num >0:
            finished_tokens= finished_tokens[:, :-right_pad_num]
        lengths= finished_tokens.ne(self.pad).sum(1)
        finished_score_unnorm= self.unnorm_score(finished_score, lengths, is_end)
        #finished_score_unnorm= finished_score* lengths.float()**self.len_scale
       
        #incremental_states= self._restore_incremental_states(incremental_states, backup)
        # left-pad and recalculate history
        self.models.rollback(incremental_states, prev_len-1)
        
        new_gen_num = finished_tokens.shape[1] -prev_len
        if new_gen_num >0:
            tmp_prev_tokens, last_token = self._to_prev_tokens(finished_tokens, new_gen_num)
            
            h_lms= self.models.recalc_lm(
                tmp_prev_tokens, incremental_states, 
                encoder_outs=encoder_outs,
                processed_length=prev_len-1
            )
            left_pad_tokens, left_pad_idx= self._to_left_pad(tmp_prev_tokens, new_gen_num)
            finished_tokens= torch.cat([left_pad_tokens,last_token], dim=1)
            # set cached key value to left_pad
            if left_pad_idx is not None:
                self.models.convert_cache_pad(incremental_states, left_pad_idx)
        finished_tokens,incremental_states  =self._remove_all_pad(finished_tokens,incremental_states)
        return finished_tokens, finished_score_unnorm
    
    def _remove_all_pad(self, prev_tokens, incremental_states):
        pad_masks= prev_tokens.eq(self.pad)
        all_pad= pad_masks.all(dim=0)
        if all_pad.sum()==0:
            return prev_tokens, incremental_states
        selected_pos= torch.arange(prev_tokens.shape[1]).to(prev_tokens)
        selected_pos = selected_pos[~all_pad]
        selected_pos = selected_pos.unsqueeze(0).repeat(prev_tokens.shape[0],1)
        prev_tokens= prev_tokens.gather(1,selected_pos)
        self.models.convert_cache_pad(incremental_states, selected_pos[:,:-1])
        return prev_tokens, incremental_states
    
    def _to_left_pad(self, prev_tokens, new_num):
        pad_masks= prev_tokens.eq(self.pad)
        if not pad_masks[:,-1].any():
            return prev_tokens, None
        max_len = prev_tokens.size(1)
        right_pad_num = pad_masks[:,-new_num:].sum(1,keepdim=True)
        range = torch.arange(max_len).to(prev_tokens).expand_as(prev_tokens)
        index = torch.remainder(range - right_pad_num, max_len)
        return prev_tokens.gather(1, index), index
    
    def _to_prev_tokens(self, gen_tokens, new_num):
        """
            gen_tokens: w1 w2 [w3 w4 w5]
            return:     w1 w2 [w3 w4], gen_tokens length -1
        """
        new= gen_tokens[:,-(new_num+1):]
        prev = gen_tokens[:,:-(new_num+1)]
        new_len = new.ne(self.pad).sum(1)
        pos= new_len.unsqueeze(1)-1
        last_token= torch.gather(new, 1,pos)
        new= torch.scatter(new, 1, pos, self.pad)
        new = new[:,:-1]
        out = torch.cat([prev,new],dim =1)
        return out,last_token



class FullyTransducerAgent(Agent):
    def __init__(self, args):
        super().__init__(args)
        self.cpu = args.cpu
        self.step_read_blocks= args.step_read_blocks
        self.decoder_step_read = args.decoder_step_read
        self.gen_beam = args.gen_beam
        self.intra_beam = args.intra_beam
        self.inter_beam = args.inter_beam
        self.given_init_frames = args.expected_init_frames
        self.max_len_a= args.max_len_a
        self.max_len_b= args.max_len_b
        self.merge_add= args.merge_add
        
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

        ensemble_models= OnlineModels(models)
      
        self.searcher = FullTransducerSearcher(
            ensemble_models, self.tgt_dict, eos = self.eos, bos= self.bos,
            max_step= 100, bos_bias= args.bos_bias, len_scale=args.len_scale,
            len_penalty=args.len_penalty,
            merge_add= self.merge_add, eager= args.eager
        )
        
        if not self.cpu:
            self.searcher.cuda()
        self.searcher.eval()
        self.frames= None
        self.processed_frames=0
        self.processed_units=0
        self.hypos=deque()
        self.prev_tokens,self.prev_scores, self.encoder_outs, self.incremental_states= None,None,None, None

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--eager", default=False, action="store_true", help="output words without word end check"
        )
        parser.add_argument(
            "--len-scale", default=1, type=float, help="length scale"
        )
        parser.add_argument(
            "--len-penalty", default= 0, type=float, help= "length penalty (for not end)"
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
            "--intra-beam", default=5, type=int,
            help="beam size"
        )
        parser.add_argument(
            "--inter-beam", default=5, type=int,
            help="beam size inter frames, may affect latency"
        )
        parser.add_argument(
            "--gen-beam", default=2, type=float,
            help="beam size inter frames, may affect latency"
        )
        parser.add_argument(
            "--step-read-blocks", default=4, type=int,
            help="do translation while read each blocks input"
        )
        parser.add_argument(
            "--decoder-step-read", default=4, type=int,
            help="decoder process each step"
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
        parser.add_argument(
            "--merge-add", action= "store_true", default= False,
            help="prob add for merge paths"
        )
        return parser
    
    def initialize_states(self, states):
        # we recompute feature at each step, the waste seems to be acceptable
        logger.info(f"new sample,id={states.instance_id}")
        self.input_fea= torch.Tensor(0,80)
        self.processed_frames=0
        self.processed_units=0
        self.prev_tokens,self.prev_scores, self.encoder_outs, self.incremental_states = self.searcher.init_states()
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
        max_steps= int(self.max_len_a* self.input_fea.shape[0] - self.max_len_b - self.prev_tokens.shape[1])
       
        if states.finish_read():
            max_steps+= 100
        if max_steps <=0:
            return
        with torch.no_grad():
            prev_tokens, prev_scores,out_words = self.searcher.search(
                fea,fea_lengths,
                self.prev_tokens, self.prev_scores,
                self.encoder_outs,
                self.incremental_states,
                intra_beam= self.intra_beam,
                inter_beam= self.inter_beam,
                gen_beam= self.gen_beam,
                read_step = self.decoder_step_read,
                is_end = states.finish_read(),
                max_steps= max_steps
            )
        self.prev_tokens= prev_tokens
        self.prev_scores= prev_scores
        
        self.processed_frames = len(self.input_fea)
        if states.finish_read():
            out_words.append(DEFAULT_EOS)
        self.hypos.extend(out_words)

    def predict(self, states):
        assert(len(self.hypos) >0)
        return self.hypos.popleft()
            




