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
from .waitk_transformer import OnlineAudioTransformerConfig, OnlineAudioTransformer,offline_audio
from examples.simultaneous_translation.models.transformer_monotonic_attention import (
    TransformerMonotonicDecoder,
)
from  examples.simultaneous_translation.modules import (
    fixed_pre_decision, monotonic_multihead_attention, monotonic_transformer_layer
)
from examples.simultaneous_translation.models import convtransformer_simul_trans
@dataclass
class MMAAudioConfig(OnlineAudioTransformerConfig):
    simul_type_str: Optional[str] = II("simul_type")
   
    mass_preservation:Optional[bool] =II("simul_type.mass_preservation")
    noise_var:Optional[float] = II("simul_type.noise_var")
    noise_mean:Optional[float] = II("simul_type.noise_mean")
    noise_type:Optional[str] = II("simul_type.noise_type")
    energy_bias:Optional[bool] =II("simul_type.energy_bias")
    energy_bias_init:Optional[float] = II("simul_type.energy_bias_init")
    attention_eps:Optional[float] =II("simul_type.attention_eps")
    fixed_pre_decision_ratio: Optional[int] =II("simul_type.fixed_pre_decision_ratio")
    fixed_pre_decision_type: Optional[str]=II("simul_type.fixed_pre_decision_type")
    fixed_pre_decision_pad_threshold: Optional[float]=II("simul_type.fixed_pre_decision_pad_threshold")

    pass

@register_model('mma_audio', dataclass = MMAAudioConfig)
class AudioMMATransformer(OnlineAudioTransformer):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)
        return decoder


@register_model_architecture("mma_audio", "mma_audio_online")
def mma_audio_online(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    args.main_context=getattr(args,"main_context",16)
    args.right_context=getattr(args,"right_context",16)
    # args.decoder_delay_blocks=getattr(args,"decoder_delay_blocks",32)
    # args.decoder_blocks_per_token= getattr(args,"decoder_blocks_per_token",8)
    #args.online_type=getattr(args,"online_type","offline")
    #args.fixed_pre_decision_ratio = getattr(args, "fixed_pre_decision_ratio", 8)
    #args.simul_type=getattr(args,"simul_type","infinite_lookback_fixed_pre_decision")
 
    randpos_audio_transformer_s(args)
