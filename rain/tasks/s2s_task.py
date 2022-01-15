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
from fairseq.dataclass.utils import gen_parser_from_dataclass
from rain.data.transforms import text_encoder, audio_encoder
import rain.models.transducer as transducer
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
EVAL_BLEU_ORDER = 4

@dataclass
class SNMTTaskConfig(FairseqDataclass):
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
    eval_bleu:bool=field(
        default= False, metadata= {"help": "evaluate bleu for validation"}
    )
    eval_bleu_remove_bpe:str= field(
        default="sentencepiece",metadata={"help":"remove bpe symbol for evaluation"}
    )
    eval_bleu_detok:str=field(
        default="space", metadata= {"help":"detok function for evaluation"}
    )
    eval_bleu_args:str = field(
        default="{}", metadata= {"help":"params for inference"}
    )
    eval_bleu_print_samples:bool = field(
        default=False,metadata= {"help":"eval bleu print samples"}
    )
    eval_tokenized_bleu:bool = field(
        default=False, metadata= {"help":"eval_tokenized_bleu"}
    )


@register_task("s2s", dataclass= SNMTTaskConfig)
class SNMTTask(LegacyFairseqTask):
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
            audio_cfg_path, transform_names= ["whiten", "tfmask"]
        )
        
        self.audio_transform_test = audio_encoder.build_audio_transforms(
            audio_cfg_path, transform_names=["whiten"]
        )
        
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if hasattr(model, 'get_ntokens'):
            sample["ntokens"]= model.get_ntokens(sample)
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
    
    def valid_step(self, sample, model, criterion):
        if hasattr(model, 'get_ntokens'):
            sample["ntokens"]= model.get_ntokens(sample)
        return super().valid_step(sample, model, criterion)
         

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

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

    def load_audio_data(self, split, epoch=1, combine=False):
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
        return loaded_data
    
    def load_text_dataset(self, split, epoch=1, combine=False):
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        dataset = translation.load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=None,
            upsample_primary=1,
            left_pad_source=False,
            left_pad_target=False,
            max_source_positions=self.args.max_text_positions,
            max_target_positions=self.args.max_text_positions,
            load_alignments=False,
            truncate_source=False,
            num_buckets= 0,
            shuffle=(split != "test"),
            pad_to_multiple=1
        )
        if split == getattr(self.args, "train_subset", "train") and self.bpe_sampling:
            dataset=BpeDropoutDataset(
                dataset,
                self.src_encoder, self.tgt_encoder,
                dropout=self.bpe_dropout
            )
        return dataset
        
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if self.task_type == "mt":
            self.datasets[split] =self.load_text_dataset(split,epoch, combine)
        else:
            self.datasets[split] =self.load_audio_data(split,epoch, combine)

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        assert self.task_type == "mt", "only mt task may inference from raw"
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )
    
    def build_model(self, args):
        model = super().build_model(args)
        
        if getattr(self.args, "eval_bleu", False):
            assert getattr(self.args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(self.args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(self.args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(self.args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            
            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if self.task_type == "mt":
            return (self.args.max_text_positions, self.args.max_text_positions)
        else:
            return (self.args.max_audio_positions, self.args.max_text_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if "source" in sample:
            sample["target"] = models[0].get_targets(sample, None)
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints,
                bos_token=self.args.infer_bos
            )
    
    def build_text_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

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
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs['print_alignment'] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

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
    
    def build_generator(self, models, args, seq_gen_cls = None, extra_gen_cls_kwargs=None):
        if self.task_type == "mt":
            return self.build_text_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs)
        else:
            return self.build_audio_generator(
                models,args,seq_gen_cls, extra_gen_cls_kwargs
            )
    
    def build_audio_generator(
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

