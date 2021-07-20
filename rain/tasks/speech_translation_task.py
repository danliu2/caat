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

logger = logging.getLogger(__name__)
"""
    num_mel_bins:int = II("task.num_mel_bins")
    #should be one of asr, st, joint
    task_type:str = II("task.task_type")
    max_audio_positions: Optional[int] = II("task.max_audio_positions")
    max_text_positions: Optional[int] = II("task.max_text_positions")
    max_target_postions:Optional[int] = II("task.max_text_positions")
    pretrained_encoder_path:Optional[str] = II("task.pretrained_encoder_path")
    pretrained_decoder_path:Optional[str] = II("task.pretrained_decoder_path")
"""
@dataclass
class SpeechTranslationTaskConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    num_mel_bins:int= field(
        default=80, metadata={"help":"mel bins number"}
    )
    audio_config:Optional[str] = field(
        default= "audio_cfg",metadata={"help":"relative path to read audio processing config"}
    )
    text_config:Optional[str] = field(
        default= "text_cfg",metadata={"help":"relative path to read text processing config"}
    )
    source_lang:Optional[str] = field(
        default= "en",metadata={"help":"relative path to read text processing config"}
    )
    target_lang:Optional[str] = field(
        default= "de",metadata={"help":"relative path to read text processing config"}
    )
    bpe_dropout:Optional[float]=field(
        default=0., metadata= {"help":"bep dropout"}
    )
    # move back to task class, for more info to control dataset process
    task_type:ChoiceEnum(["asr","st","joint", "mt"]) =field(
        default="asr", metadata={"help":"task type"}
    )
    # task_type:ChoiceEnum(["asr","st","joint"]) =field(
    #     default="asr", metadata={"help":"task type"}
    # )
    #don't set these params to prune data, these may work on test set too. filter trainset before you run
    max_audio_positions:Optional[int] = field(
        default= 7000, metadata={"help":"max audio frams"}
    )
    max_text_positions:Optional[int]=field(
        default= 512, metadata={"help":"max text positions"}
    )
    pretrained_encoder_path:Optional[str] = field(
        default= None, metadata={"help":"pretrained_encoder_path"}
    )
    pretrained_decoder_path:Optional[str] = field(
        default= None, metadata={"help":"pretrained_decoder_path"}
    )
    infer_bos:Optional[int] = field(
        default= None, metadata = {"help":"bos id for inference, default to eos"}
    )

@register_task("speech_translation", dataclass=SpeechTranslationTaskConfig)
class SpeechTranslationTask(LegacyFairseqTask):
    """ @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--audio-config",
            type=str,
            default="audio_cfg",
            help="relative path to read audio processing config",
        )
        parser.add_argument(
            "--text-config",
            type= str,
            default= "text_cfg",
            help = "relative path to read text processing config"
        )
        parser.add_argument(
            "--pretrained-encoder-path",
            type= str,
            default= None,
            help = "path to pretrained encoder"
        )
        parser.add_argument(
            "--pretrained-decoder-path",
            type= str,
            default= None,
            help = "path to pretrained decoder"
        )
        parser.add_argument(
            "--num-mel-bins", type=int, default=80,
            help="mel bins shape"
        )
        parser.add_argument(
            "--source-lang",type=str, default="en",
            help= "lang id for source"
        )
        parser.add_argument(
            "--target-lang", type=str, default= "de",
            help = "lang id for target"
        )
        parser.add_argument(
            "--bpe-dropout", type=float, default= 0.,
            help ="alpha for bpe dropout, 0 for no dropout, ATTENTION: binary dataset only works with no bpe dropout"
        )
        parser.add_argument(
            "--task-type", type=str, default= 0.,
            help ="should be one of asr, st, joint"
        )

        parser.add_argument(
            "--max-audio-positions",
            default=3000,
            type=int,
            metavar="N",
            help="max number of frames in the audio",
        )
        parser.add_argument(
            "--max-text-positions",
            default=512,
            type=int,
            metavar="N",
            help="max number of tokens in the source and target sequence",
        ) """

    def __init__(self, args):
        super().__init__(args)
        #self.task_type= args.task_type
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
            audio_cfg_path, transform_names= ["whiten", "tfmask"]
        )
        self.audio_transform_test = audio_encoder.build_audio_transforms(
            audio_cfg_path, transform_names=["whiten"]
        )


    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)
    
    def build_criterion(self, args):
        return super().build_criterion(args)
    
    @property
    def source_dictionary(self):
        return self.src_dict
    
    @property
    def target_dictionary(self):
        return self.tgt_dict
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train = split.startswith("train")
        src= self.args.source_lang
        tgt= self.args.target_lang
        def build_path(prefix, src,tgt, lang):
            return f"{prefix}.{src}-{tgt}.{lang}"
        datas = split.split(",")
        src_needed=tgt_needed=True
        # if self.task_type == "asr":
        #     tgt_needed= False
        # if self.task_type =="st":
        #     src_needed=False
        datasets=[]
        for data in datas:
            prefix= op.join(self.args.data, data)
            audio_path = build_path(prefix,src, tgt, "audio")
            src_path = build_path(prefix, src, tgt, src)
            tgt_path = build_path(prefix, src,tgt,tgt)
            if not FbankZipDataset.exists(audio_path):
                raise FileNotFoundError(f"no audio data for {data}")
            audio_data= FbankZipDataset(
                audio_path, self.audio_transform_train if is_train else self.audio_transform_test,
            )
            src_data, tgt_data= None,None
            if not self.bpe_sampling:
                if indexed_dataset.dataset_exists(src_path, "cached"):
                    src_data = data_utils.load_indexed_dataset(
                        src_path, self.src_dict, None
                    )
                if indexed_dataset.dataset_exists(tgt_path, "cached"):
                    tgt_data = data_utils.load_indexed_dataset(
                        tgt_path, self.tgt_dict, None
                    )
            if src_data is None:
                if RawTextDataset.exists(src_path):
                    src_data= RawTextDataset(
                        src_path, self.src_dict,self.src_encoder,
                        dropout= self.bpe_dropout if is_train else 0.
                    )
                if RawTextDataset.exists(tgt_path):
                    tgt_data = RawTextDataset(
                        tgt_path, self.tgt_dict, self.tgt_encoder,
                        dropout= self.bpe_dropout if is_train else 0.
                    )
            assert not src_needed or src_data
            assert not tgt_needed or tgt_data
            dataset= SpeechTranslationDataset(
                audio_data, self.src_dict, self.tgt_dict,
                src_data, tgt_data,
                shuffle= is_train
            )
            datasets.append(dataset)
        if len(datasets) == 1:
            loaded_data= datasets[0]
        else:
            distribution = np.ones(len(datasets))/len(datasets)
            loaded_data= MultiCorpusDataset(
                dict(zip(datas,datasets)),
                distribution,
                seed= 1527+epoch,
                sort_indices=True
            )
        self.datasets[split] =loaded_data
    
    def max_positions(self):
        return self.args.max_audio_positions, self.args.max_text_positions

    def build_dataset_for_inference(self, tsvfile):
        audio_data= FbankZipDataset(tsvfile, self.audio_transform_test)
        return SpeechTranslationDataset(
            audio_data, self.src_dict, self.tgt_dict, shuffle=False
        )
    
    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        from rain.sequence_generator2 import SequenceGenerator2

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        seq_gen_cls = SequenceGenerator2

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        """
            we should build new infer tools for simultaneuos speech translation, 
            but some baseline needs fairseq tools,
            fairseq-generate needs sample["target"] and task.target_dictionary,
            share dict works ok then
        """
        # to make current fairseq-generate happy, for asr, mt
        # if hasattr(models[0], "task_type"):
        #     models[0].task_type= self.task_type
        sample["target"] = models[0].get_targets(sample, None)
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

