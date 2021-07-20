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
from rain.simul.waitk_agent import WaitkAgent

class SpeechWaitkAgent(WaitkAgent):
    data_type= "speech"
    speech_segment_size = 10
    def _set_default_args(self, args):
        args.wait_blocks= 32 if args.wait_blocks is None else args.wait_blocks
        args.step_read_blocks=8 if args.step_read_blocks is None else args.step_read_blocks
        args.step_generate=1 if args.step_generate is None else args.step_generate
        args.step_forecast= 0 if args.step_forecast is None else args.step_forecast