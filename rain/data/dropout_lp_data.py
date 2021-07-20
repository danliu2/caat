import logging

import numpy as np
import torch
from fairseq.data import (
    FairseqDataset, data_utils, BaseWrapperDataset,
    language_pair_dataset,
    Dictionary
)

from .transforms.text_encoder import TextEncoder

class BpeDropoutDataset(BaseWrapperDataset):
    def __init__(
        self, dataset:language_pair_dataset.LanguagePairDataset,
        src_encoder:TextEncoder,
        tgt_encoder:TextEncoder,
        dropout=0.1
    ):
        super().__init__(dataset)
        self.src_encoder= src_encoder
        self.tgt_encoder= tgt_encoder
        self.dropout=dropout
    
    def _rebpe(self, x:torch.Tensor, dict:Dictionary, encoder:TextEncoder):
        text= dict.string(x, bpe_symbol="sentencepiece",unk_string="<unk>")
        text= encoder.encode(text, sampling=True,alpha= self.dropout)
        wids= dict.encode_line(text,add_if_not_exist=False,append_eos=True)
        return wids.long()
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        source,target= sample["source"],sample["target"]
        source= self._rebpe(source, self.dataset.src_dict, self.src_encoder)
        target = self._rebpe(target, self.dataset.tgt_dict, self.tgt_encoder)
        return {
            "id":sample["id"],
            "source":source,
            "target":target
        }

