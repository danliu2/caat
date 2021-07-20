import os
import sys
import argparse
import time
import logging
import json
from fairseq.data import Dictionary
from simuleval.agents import Agent,TextAgent, SpeechAgent
from .text_waitk import TextWaitkAgent
from .text_fullytransducer_agent import TextFullyTransducerAgent
from rain.data.transforms.text_encoder import SpaceSplitter
from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
from simuleval.metrics import latency
from collections import deque, OrderedDict
import sacrebleu
from argparse import Namespace
import numpy as np

Agents={
    "waitk": TextWaitkAgent,
    "transducer": TextFullyTransducerAgent
}

def general_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default= None, type= str,
                        help = "agent type")
    parser.add_argument('--source', default=None,type=str,
                        help='source file')
    parser.add_argument("--target", default= None, type= str,
                        help = "target file")
    parser.add_argument('--tokenizer', type=str, default="13a",
                        help='tokenizer used in sacrebleu')
    return parser

class Bleuer(object):
    def __init__(self, tokenizer='13a'):
        self.tokenizer= tokenizer
    
    def sent_bleu(self, ref, pred):
        args = Namespace(
        smooth_method='floor', smooth_value=0.1, force=False,
        short=False, lc=False, tokenize=self.tokenizer)
        metric = sacrebleu.metrics.bleu.BLEU(args)
        return metric.sentence_score(
            pred, [ref], use_effective_order=True).score
    
    def corpuse_bleu(self, refs, preds):
        return sacrebleu.corpus_bleu(
            preds, [refs], tokenize=self.tokenizer
        ).score

class FakeStates():
    def __init__(self, inst):
        self.instance_id= inst
        self.source=[]
        self.finished=False
    
    def finish_read(self):
        return self.finished

class SmtEvaluator(object):
    def __init__(self, agent:Agent, encoder, srcfile, reffile, tokenizer='13a'):
        self.bleuer=Bleuer(tokenizer)
        self.agent= agent
        self.encoder= encoder
        self.srcfile= srcfile
        self.reffile=reffile
        self.source_texts = []
        self.ref_texts= []
        self.preds=[]
        self.als=[]
        self.aps=[]
    
    def process_one(self, src, ref, instance_id):
        src_bpe= self.encoder.encode(src)
        
        src_tokens= src_bpe.split()
        states= FakeStates(instance_id)
        self.agent.initialize_states(states)
        delays=[]
        preds=[]
        for i,stoken in enumerate(src_tokens):
            states.source.append(stoken)
            if i== len(src_tokens)-1:
                states.finished=True
                
            while self.agent.policy(states) == WRITE_ACTION:
                curr=self.agent.predict(states)
                for token in curr.split():
                    preds.append(token)
                    delays.append(i)
                if preds[-1] == DEFAULT_EOS:
                    break
        self.source_texts.append(src)
        self.ref_texts.append(ref)
        if preds[-1] == DEFAULT_EOS:
            preds.pop()
        preds=' '.join(preds)
        preds=self.encoder.decode(preds)
        self.preds.append(preds)
        src_len= len(src_tokens)
        al = latency.AverageLagging(delays, src_len)
        ap = latency.AverageProportion(delays, src_len)
        self.als.append(al)
        self.aps.append(ap)
        score= self.bleuer.sent_bleu(ref, preds)
        print(f"I-{instance_id}\tAL={al}\tAP={ap}\tBLEU={score}")
        print(f"SRC:\t {src}")
        print(f"REF:\t {ref}")
        print(f"PRED:\t{preds}")
    
    def process(self):
        with open(self.srcfile, 'r',encoding='utf-8') as fsrc, open(self.reffile, 'r',encoding='utf-8') as fref:
            for i,(src,ref) in enumerate(zip(fsrc, fref)):
                self.process_one(src.strip(), ref.strip(), i) 
        all_bleu= self.bleuer.corpuse_bleu(self.ref_texts, self.preds)
        all_al = np.array(self.als).mean()
        all_ap = np.array(self.aps).mean()
        print(f'total BLEU={all_bleu}, AL={all_al}, AP={all_ap}')

            



class SubWordEndChecker(object):
    def __init__(self, vocab:Dictionary):
        self.vocab= vocab
    
    def string(self,tokens, is_finished=False, removed=0):
        tokens_cpu= tokens.cpu()
        
        tnum= len(tokens_cpu)
        end_pos = tnum -removed
        out_str = self.vocab.string(tokens_cpu[:end_pos],bpe_symbol=None)
        return out_str, removed

def main():
    parser= general_parser()
    args,_ = parser.parse_known_args()
    if args.agent not in Agents:
        raise NotImplementedError("Agents now only can be waitk and transducer")
    agent_cls= Agents[args.agent]
    parser2= general_parser()
    agent_cls.add_args(parser2)
    args, _ = parser2.parse_known_args()
    agent= agent_cls(args)
    real_encoder= agent.text_transformer
    agent.text_transformer= SpaceSplitter()
    agent.searcher.word_end = SubWordEndChecker(agent.task.target_dictionary)
    evaluator= SmtEvaluator(agent, real_encoder, args.source, args.target, args.tokenizer)
    evaluator.process()

if __name__=='__main__':
    main()
