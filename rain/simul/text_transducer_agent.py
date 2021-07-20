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
from rain.simul.transducer_agent import TransducerAgent

class TextTransducerAgent(TransducerAgent):
    data_type= "text"
   