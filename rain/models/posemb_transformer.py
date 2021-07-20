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
from rain.layers.rand_pos import PositionalEmbedding
from .speech_transformer import SpeechTransformerModelConfig

@register_model("randpos_transformer", dataclass=SpeechTransformerModelConfig)
class RandposTransformer(transformer.TransformerModel):
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        if getattr(args,"max_text_positions", None) is None:
            args.max_text_positions= 1024
        args.max_source_positions = args.max_text_positions
        args.max_target_positions = args.max_text_positions

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        model = transformer.TransformerEncoder(args, src_dict, embed_tokens)
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
        model = transformer.TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        if model.embed_positions is not None and args.rand_pos_decoder >0:
            model.embed_positions= PositionalEmbedding(
                model.max_target_positions,
                model.embed_dim,
                model.padding_idx,
                rand_max = args.rand_pos_decoder,
                learned=args.decoder_learned_pos,
            )
        return model

@register_model_architecture("randpos_transformer", "randpos_transformer2")
def randpos_transformer(args):
    args.rand_pos_encoder= getattr(args, "rand_pos_encoder", 30)
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    transformer.base_architecture(args)
    
@register_model_architecture("randpos_transformer", "randpos_transformer_small")
def randpos_transformer_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.dropout = getattr(args, "dropout", 0.1)
    randpos_transformer(args)
