import numpy as np
import torch
from fairseq.data import FairseqDataset,Dictionary,data_utils, LanguagePairDataset
from .audio_dataset import FbankZipDataset
from typing import List,Dict,Optional


class SpeechTranslationDataset(FairseqDataset):
    def __init__(
        self,
        audio_data:FairseqDataset,
        src_vocab:Dictionary,
        tgt_vocab:Dictionary,
        src_data:FairseqDataset = None,
        tgt_data:FairseqDataset = None,
        shuffle=True
    ):
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_data= src_data
        self.tgt_data=tgt_data
        self.audio_data= audio_data
        self.audio_sizes= audio_data.sizes
        if src_data is not None:
            assert len(audio_data) == len(src_data)
        if tgt_data is not None:
            assert len(audio_data) == len(tgt_data)
        self.src_sizes = src_data.sizes if src_data is not None else None
        self.tgt_sizes= tgt_data.sizes if tgt_data is not None else None
        sizes = [s for s in [self.audio_sizes, self.src_sizes, self.tgt_sizes] if s is not None]
        self.sizes= np.vstack(sizes).T
        self.shuffle= shuffle
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, index):
        
        fbank = self.audio_data[index]["fbank"]
        source = self.src_data[index] if self.src_data else None
        target = self.tgt_data[index] if self.tgt_data else None
        return {"id":index,"fbank":fbank, "source":source, "target":target}
    
    def num_tokens(self, index):
        return self.audio_sizes[index]
    
    def size(self, index):
        return tuple(self.sizes[index])

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices[np.argsort(self.audio_sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        ret = getattr(self.audio_data, "supports_prefetch", False)
        if self.src_data:
            ret = ret and getattr(self.src_data, "supports_prefetch", False) 
        if self.tgt_data:
            ret = ret and  getattr(self.tgt_data, "supports_prefetch", False) 
        return ret
    
    def prefetch(self, indices):
        self.audio_data.prefetch(indices)
        if self.src_data:
            self.src_data.prefetch(indices)
        if self.tgt_data:
            self.tgt_data.prefetch(indices)
    
    def collater(self, samples, **unused):
        if len(samples) == 0:
            return {}
        pad,eos= self.src_vocab.pad(),self.src_vocab.eos()
        
        def merge(key, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad,
                eos,
                left_pad = False,
                move_eos_to_beginning = move_eos_to_beginning,
                pad_to_length=None,
                pad_to_multiple=1,
            )
        
        def merge_feature(features):
            fdim = features[0].shape[1]
            lengths= [f.shape[0] for f in features]
            bsz, tgt_len = len(features), max(lengths)
            ofeas= features[0].new(bsz, tgt_len, fdim).fill_(0)
            for i,l in enumerate(lengths):
                ofeas[i,:l] = features[i]
            lengths = torch.LongTensor(lengths)
            return ofeas, lengths

        id = torch.LongTensor([s["id"] for s in samples])
        fbank, fbk_lengths = merge_feature([s['fbank'] for s in samples]) 
        source, target= None, None
        prev_source, prev_target= None, None
        if samples[0].get("source",None) is not None:
            source = merge("source")
            prev_source = merge("source", move_eos_to_beginning = True)
        if samples[0].get("target", None) is not None:
            target = merge("target")
            prev_target= merge("target", move_eos_to_beginning=True)
        return {
            "id":id, 
            "net_input":{
                "fbank":fbank, "fbk_lengths":fbk_lengths,
                "prev_source":prev_source,
                "prev_target":prev_target
            },
            "source":source,
            "target":target, 
        }
    
    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils.filter_paired_dataset_indices_by_size(
            self.audio_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
        # ignore2 = indices[self.audio_sizes[indices]<= self.src_sizes[indices]]
        # if len(ignore2) >0:
        #     ignored.extend(ignore2)
        #     indices = indices[self.audio_sizes[indices] > self.src_sizes[indices]]
        return indices, ignored

        




