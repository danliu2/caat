import sys
import os
sys.path.append(os.getcwd())
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
from simuleval.agents import SpeechAgent

class DummySpeechAgent(SpeechAgent):
    def __init__(self,args):
        super().__init__(args)
        self.frames_per_token= 50
        self.curr_frames=0
    
    def initialize_states(self, states):
        self.curr_frames == 0
    
    def policy(self, states):
        source= states.source
        if len(source) - self.curr_frames >= self.frames_per_token:
            self.curr_frames= len(source)
            return WRITE_ACTION
        if states.finish_read():
            return WRITE_ACTION
        return READ_ACTION
    
    def predict(self, states):
        if states.finish_read():
            return DEFAULT_EOS
        return "a"

    