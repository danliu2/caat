import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from rain.data import BpeDropoutDataset
from rain.data.transforms.text_encoder import TextEncoder
import json
import logging
import os
from argparse import Namespace
from fairseq.data import encoders

@register_task("dropout_translation")
class DropoutTranslationTask(TranslationTask):
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
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--bpe-dropout', type=float, default=0.1,
                            help='bpe_dropout')
        parser.add_argument("--src-encoder", type=str,
                            help= "source encoder sentence piece")
        parser.add_argument("--tgt-encoder", type=str,
                            help= "target encoder sentence piece")
        parser.add_argument("--max-text-positions", type=int,
                            default=512, help = "max text position")
        parser.add_argument(
            "--num-mel-bins", type=int, default=80,
            help="mel bins shape"
        )
        # parser.add_argument("--pretrained-encoder-path", type=str, default= None)
        # parser.add_argument("--pretrained-decoder-path", type=str, default= None)
      
        # parser.add_argument(
        #     "--task-type", type=str, default= 0.,
        #     help ="should be one of asr, st, joint"
        # )

        parser.add_argument(
            "--max-audio-positions",
            default=3000,
            type=int,
            metavar="N",
            help="max number of frames in the audio",
        )
    

    def __init__(self, args, src_dict, tgt_dict):
        
        super().__init__(args, src_dict, tgt_dict)
        self.bpe_dropout=args.bpe_dropout
        self.src_encoder=TextEncoder(args.src_encoder)
        self.tgt_encoder= TextEncoder(args.tgt_encoder)
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch, combine, **kwargs)
       
        if "train" in split:
            self.datasets[split] = BpeDropoutDataset(
                self.datasets[split],
                self.src_encoder, self.tgt_encoder,
                dropout=self.bpe_dropout
            )
        
