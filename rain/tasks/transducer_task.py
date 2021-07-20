import logging
import os.path as op
from argparse import Namespace
from dataclasses import dataclass
import numpy as np
import json
import torch
from fairseq import search, utils,metrics
from fairseq.tasks import LegacyFairseqTask, register_task, translation
from fairseq.data import (
    Dictionary, indexed_dataset,data_utils,LanguagePairDataset,encoders
)
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from rain.data import (
    FbankZipDataset, RawTextDataset, SpeechTranslationDataset,
    BpeDropoutDataset
)
from rain.data.transforms import text_encoder, audio_encoder
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from . import s2s_task
from fairseq.optim import FP16Optimizer
logger = logging.getLogger(__name__)


@register_task("transducer", dataclass= s2s_task.SNMTTaskConfig)
class TransducerTask(s2s_task.SNMTTask):
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        raise NotImplementedError("TODO")

    def build_generator(self, models, args, seq_gen_cls = None, extra_gen_cls_kwargs=None):
        raise NotImplementedError("TODO")
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
            model.train_step returns
            {"loss":losses[0], "loss_prob":losses[1], "loss_delay":losses[2], "sample_size": B}
        """
        model.train()
        model.set_num_updates(update_num)
        scaler=None
        if hasattr(optimizer, "scaler"):
            scaler= optimizer.scaler
        if ignore_grad:
            loss_info = model.eval_step(sample)
        else:
            loss_info= model.train_step(sample,scaler=scaler)
            if isinstance(optimizer, FP16Optimizer):
                optimizer._needs_sync=True
        loss, sample_size, logging_output = criterion(loss_info, sample, model)
        return loss,sample_size, logging_output
    
    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss_info = model.eval_step(sample)
        loss, sample_size, logging_output = criterion(loss_info, sample, model)
        return loss,sample_size, logging_output
    
    def build_criterion(self, args: Namespace):
        """
        should only build fake criterion
        """
        from rain.criterions.fake_creterion import FakeCriterion

        return FakeCriterion(self)