import io,os
import numpy as np
import torch
from fairseq.data import FairseqDataset,Dictionary
from .transforms.text_encoder import TextEncoder

"""
    this is only temporary code for bpe dropout , read from raw text, and do tokenize online
    may be less efficiency, and we still don't know whether online bpe is useful.
    the num_tokens may be not exact for no bpe
    Mimic all APIs of IndexedDataset
"""

class RawTextDataset(FairseqDataset):
    def __init__(
        self, path, 
        dictionary:Dictionary,
        text_encoder:TextEncoder,
        dropout = 0.,
    ):
        path = path+".raw"
        self.dictionary= dictionary
        self.text_encoder= text_encoder
        self.dropout= dropout
        self.sampling= self.dropout >1e-4
        self.sentences, self.sizes= [],[]
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                tokens = self.text_encoder.split_to_word(line)
                self.sizes.append(len(tokens))
                self.sentences.append(line)
        self.sizes = np.array(self.sizes)
    
    def __getitem__(self,index):
        bpe_out = self.text_encoder.encode(self.sentences[index])
        tokens = self.dictionary.encode_line(
            bpe_out,
            add_if_not_exist=False,
            append_eos=True,
            reverse_order=False
        ).long()
        return tokens
    
    def num_tokens(self,index):
        return self.sizes[index]
    
    def size(self, index):
        return self.sizes[index]
    
    def __len__(self):
        return len(self.sentences)
    
    @staticmethod
    def exists(path):
        return os.path.exists(path+".raw")

