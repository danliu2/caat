import logging
import os.path as op
from argparse import Namespace
from dataclasses import dataclass
import numpy as np
import torch
from fairseq import search
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data import Dictionary, indexed_dataset,data_utils
from fairseq.data.multi_corpus_dataset import MultiCorpusDataset
from rain.data import FbankZipDataset, RawTextDataset, SpeechTranslationDataset
from rain.data.transforms import text_encoder, audio_encoder
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .speech_translation_task import SpeechTranslationTaskConfig, SpeechTranslationTask

@register_task("speech_translation2", dataclass=SpeechTranslationTaskConfig)
class SpeechTranslationTas2(SpeechTranslationTask):
    def __init__(self, args):
        super().__init__(args)
        self.task_type= args.task_type
        voc_path = op.join(args.data, args.text_config)
        vocab_mgr:text_encoder.VocabManager = text_encoder.VocabManager(voc_path)
        self.vocab_mgr= vocab_mgr
        self.src_dict= self.vocab_mgr.get_dictionary(args.source_lang)
        self.src_encoder= self.vocab_mgr.get_encoder(args.source_lang)
        self.tgt_dict= self.vocab_mgr.get_dictionary(args.target_lang)
        self.tgt_encoder= self.vocab_mgr.get_encoder(args.target_lang)

        self.bpe_dropout=args.bpe_dropout
        self.bpe_sampling= self.bpe_dropout > 1e-3

        audio_cfg_path = op.join(args.data, args.audio_config)
        
        self.audio_transform_train = audio_encoder.build_audio_transforms(
            audio_cfg_path, transform_names= ["whiten"]
        )
        self.audio_transform_test = audio_encoder.build_audio_transforms(
            audio_cfg_path, transform_names=["whiten"]
        )
        print("training without tf-mask")
