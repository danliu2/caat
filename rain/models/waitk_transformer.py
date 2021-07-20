from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import os
from torch import Tensor
import torch.nn as nn
from fairseq import options, utils, checkpoint_utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    transformer,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
    FairseqEncoder, FairseqIncrementalDecoder,
    BaseFairseqModel,FairseqEncoderDecoderModel
)
from fairseq.data import Dictionary
from fairseq.models.transformer import (
    Embedding, TransformerDecoder,
    TransformerEncoder,
    TransformerModel
)
from fairseq.models.transformer import base_architecture as transformer_architecture
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,Fp32LayerNorm,
    LayerDropModuleList,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import II
from rain.layers import (
    get_available_convs, get_conv,
    PositionalEmbedding,AudioTransformerEncoder,
    UnidirectAudioTransformerEncoder, UnidirectTransoformerEncoder,
    WaitkDecoder
)
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from .speech_transformer import (
    SpeechTransformerModelConfig, AudioTransformer,audio_transformer_s,randpos_audio_transformer_s
)
from .posemb_transformer import RandposTransformer,randpos_transformer_small

@dataclass
class OnlineTextTransformerConfig(SpeechTransformerModelConfig):
    max_source_positions: Optional[int] = II("task.max_text_positions")
    max_target_positions:Optional[int] = II("task.max_text_positions")
    main_context:int = field(
        default= 1, metadata={"help":"main context frame"}
    )
    right_context :int = field(
        default= 0, metadata={"help":"right context frame"}
    )
    decoder_delay_blocks :int = field(
        default= 4, metadata={"help":"wait-k delay"}
    )
    decoder_blocks_per_token :int = field(
        default= 1, metadata={"help":"wait k token blocks"}
    )
    online_type:ChoiceEnum(["offline", "waitk"])="offline"

@register_model("online_text_transformer", dataclass = OnlineTextTransformerConfig)
class OnlineTextTransformer(TransformerModel):

    @classmethod
    def add_args(cls, parser):
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc(), delete_default=True)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        model = UnidirectTransoformerEncoder(args, src_dict, embed_tokens)
        embed_dim= embed_tokens.embedding_dim
        if model.embed_positions is not None and args.rand_pos_encoder >0:
            model.embed_positions= PositionalEmbedding(
                model.max_source_positions,
                embed_dim,
                model.padding_idx,
                rand_max = args.rand_pos_encoder,
                learned=args.decoder_learned_pos,
            )
        return model

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.online_type == "offline":
            model = transformer.TransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=False
            )
        elif args.online_type == "waitk":
            model= WaitkDecoder(
                args,
                tgt_dict,
                embed_tokens
            )
        else:
            raise NotImplementedError(f"unknown online type {args.online_type}")
        if model.embed_positions is not None and args.rand_pos_decoder >0:
            model.embed_positions= PositionalEmbedding(
                model.max_target_positions,
                model.embed_dim,
                model.padding_idx,
                rand_max = args.rand_pos_decoder,
                learned=args.decoder_learned_pos,
            )
        return model

@dataclass
class OnlineAudioTransformerConfig(OnlineTextTransformerConfig):
    max_source_positions: Optional[int] = II("task.max_audio_positions")
    max_target_positions:Optional[int] = II("task.max_text_positions")
    main_context:int = field(
        default= 16, metadata={"help":"main context frame"}
    )
    right_context :int = field(
        default= 16, metadata={"help":"right context frame"}
    )
    decoder_delay_blocks :int = field(
        default= 16, metadata={"help":"wait-k delay"}
    )
    decoder_blocks_per_token :int = field(
        default= 4, metadata={"help":"wait k token blocks"}
    )
    online_type:ChoiceEnum(["offline", "waitk"])="offline"


@register_model("online_audio_transformer", dataclass = OnlineAudioTransformerConfig)
class OnlineAudioTransformer(AudioTransformer):
    @classmethod
    def build_encoder(cls, args):
        return UnidirectAudioTransformerEncoder(args)
    
    @classmethod
    def add_args(cls, parser):
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc(), delete_default=True)
       
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.online_type == "offline":
            model = transformer.TransformerDecoder(
                args,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=False
            )
        elif args.online_type == "waitk":
            model= WaitkDecoder(
                args,
                tgt_dict,
                embed_tokens
            )
        else:
            raise NotImplementedError(f"unknown online type {args.online_type}")
        if model.embed_positions is not None and args.rand_pos_decoder >0:
            model.embed_positions= PositionalEmbedding(
                model.max_target_positions,
                model.embed_dim,
                model.padding_idx,
                rand_max = args.rand_pos_decoder,
                learned=args.decoder_learned_pos,
            )
        return model
    
    def forward(
        self,
        fbank,
        fbk_lengths,
        prev_source=None,
        prev_target= None
    ):
        
        encoder_out = self.encoder(fbank, fbk_lengths)
        
        prev_tokens= prev_source if self.task_type=="asr" else prev_target
        assert prev_tokens is not None, f"no prev_tokens for {self.task_type}"
        decoder_out = self.decoder(
            prev_tokens,
            encoder_out=encoder_out,
        )
        logits= decoder_out[0]
        
        return decoder_out


@register_model_architecture("online_audio_transformer", "online_audio_transformer_offline")
def offline_audio(args):
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    args.main_context=getattr(args,"main_context",16)
    args.right_context=getattr(args,"right_context",16)
    args.decoder_delay_blocks=getattr(args,"decoder_delay_blocks",32)
    args.decoder_blocks_per_token= getattr(args,"decoder_blocks_per_token",8)
    args.online_type=getattr(args,"online_type","offline")
    randpos_audio_transformer_s(args)

@register_model_architecture("online_audio_transformer", "online_audio_transformer_waitk")
def waitk_audio(args):
    args.online_type=getattr(args,"online_type","waitk")
    offline_audio(args)

@register_model_architecture("online_text_transformer", "online_text_transformer_offline")
def offline_text(args):
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 30)
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    args.main_context=getattr(args,"main_context",1)
    args.right_context=getattr(args,"right_context",0)
    args.decoder_delay_blocks=getattr(args,"decoder_delay_blocks", 4)
    args.decoder_blocks_per_token= getattr(args,"decoder_blocks_per_token",1)
    args.online_type=getattr(args,"online_type","offline")
    randpos_transformer_small(args)

@register_model_architecture("online_text_transformer", "online_text_transformer_waitk")
def waitk_text(args):
    args.online_type=getattr(args,"online_type","waitk")
    offline_text(args)
