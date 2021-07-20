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
    PositionalEmbedding,AudioTransformerEncoder
)

DEFAULT_MAX_TEXT_POSITIONS = 256
DEFAULT_MAX_AUDIO_POSITIONS=2000
from rain.layers.multihead_attention_relative import MultiheadRelativeAttention, replace_relative_attention




@dataclass
class SpeechTransformerModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    conv_type: ChoiceEnum(get_available_convs()) = field(
        default= "vgg_base", metadata= {"help": "convolution type for speech encoder"}
    )

    encoder_max_relative_position: int= field(
        default=-1, metadata={"help": "max_relative_position for encoder Relative attention, <0 for traditional attention"}
    )
    decoder_max_relative_position: int= field(
        default=-1, metadata={"help": "max_relative_position for decoder Relative attention, <0 for traditional attention"}
    )
   
    encoder_embed_dim: int = field(
        default= 512, metadata={"help":"encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the encoder"},
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for encoder"}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_audio_positional_embeddings:bool = field(
        default = False,
        metadata={"help":"if set, donot use position embedding on audio encoder"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    no_cross_attention:Optional[bool]=False
    encoder_layers_to_keep:Optional[str]=None
    decoder_layers_to_keep:Optional[str]=None
    cross_self_attention:Optional[bool]=False
    decoder_embed_path:Optional[str]=None
    encoder_embed_path:Optional[str]=None
    share_all_embeddings:bool = field(
        default = False, metadata={"help":"share embedding between source and target"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    tie_adaptive_weights:bool=False
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    # TODO common var add to parent
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    rand_pos_encoder:int = field(
        default= 0,
        metadata={
            "help":"max random start for encoder position embedding"
        }
    )
    rand_pos_decoder:int = field(
        default= 0,
        metadata={
            "help":"max random start for encoder position embedding"
        }
    )
    # pretrained_encoder_path:Optional[str] = field(
    #     default= None, metadata={"help":"pretrained_encoder_path"}
    # )
    # pretrained_decoder_path:Optional[str] = field(
    #     default= None, metadata={"help":"pretrained_decoder_path"}
    # )
    num_mel_bins:int = II("task.num_mel_bins")
    #should be one of asr, st, joint
    task_type:str = II("task.task_type")
    
    max_audio_positions: Optional[int] = II("task.max_audio_positions")
    max_text_positions: Optional[int] = II("task.max_text_positions")
    max_source_positions: Optional[int] = II("task.max_audio_positions")
    
    max_target_positions:Optional[int] = II("task.max_text_positions")
    pretrained_encoder_path:Optional[str] = II("task.pretrained_encoder_path")
    pretrained_decoder_path:Optional[str] = II("task.pretrained_decoder_path")
    # field(
    #     default= None,
    #     metadata={"help":"path to pretrained decoder"}
    # )

    # params for online
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
    online_type:ChoiceEnum(["offline", "waitk"])=field(
        default="offline", metadata={"help":"online type"}
    )
    


@register_model("audio_transformer", dataclass=SpeechTransformerModelConfig) 
class AudioTransformer(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.task_type= getattr(args,"task_type", "asr")
        if self.task_type not in ("asr", "st", "joint"):
            raise ValueError(f"unknown task type {self.task_type}")
        self.padding_idx= encoder.dictionary.pad()
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        #base_architecture(args)

        # if args.encoder_layers_to_keep:
        #     args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        # if args.decoder_layers_to_keep:
        #     args.decoder_layers = len(args.decoder_layers_to_keep.split(","))
        
        if getattr(args, "max_audio_positions", None) is None:
            args.max_audio_positions = DEFAULT_MAX_AUDIO_POSITIONS
        if getattr(args, "max_text_positions", None) is None:
            args.max_text_positions = DEFAULT_MAX_TEXT_POSITIONS
        args.max_source_positions = args.max_audio_positions
        args.max_target_positions= args.max_text_positions

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        task_type= getattr(args,"task_type", "asr")
        if task_type not in ("asr", "st", "joint"):
            raise ValueError(f"unknown task type {task_type}")
        if task_type == "joint":
            assert args.share_all_embeddings==True and src_dict==tgt_dict, "joint model needs to share all"
        if task_type =="asr" or task_type=="joint":
            embed_tokens= cls.build_embedding(
                args, src_dict, args.decoder_embed_dim
            )
            decoder= cls.build_decoder(args, src_dict, embed_tokens)
        elif task_type == "st":
            embed_tokens= cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim
            )
            decoder= cls.build_decoder(args, tgt_dict, embed_tokens)
        else:
            raise NotImplementedError(f"task type {task_type} not supported")
        encoder= cls.build_encoder(args)
        model = cls(args, encoder, decoder)
        if args.pretrained_encoder_path is not None:
            model.load_pretrained_encoder(args.pretrained_encoder_path)
        if args.pretrained_decoder_path is not None:
            model.load_pretrained_decoder(args.pretrained_decoder_path)
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb
    
    def get_targets(self, sample, net_output):
        if self.task_type == "asr":
            return sample["source"]
        elif self.task_type=="st":
            return sample["target"]
        else:
            return sample["target"]
    
    def get_ntokens(self, sample):
        if self.task_type == "asr":
            return sample["source"].ne(self.padding_idx).long().sum().item()
        elif self.task_type=="st":
            return sample["target"].ne(self.padding_idx).long().sum().item()
        else:
            return sample["target"].ne(self.padding_idx).long().sum().item()

    @classmethod
    def build_encoder(cls, args):
        model= AudioTransformerEncoder(args)
        if args.encoder_max_relative_position >0:
            for layer in model.layers:
                layer.self_attn = replace_relative_attention(layer.self_attn, args.encoder_max_relative_position)

        return model
       

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        model = TransformerDecoder(
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
        if args.decoder_max_relative_position >0:
            for layer in model.layers:
                layer.self_attn = replace_relative_attention(layer.self_attn, args.decoder_max_relative_position)
        return model
    
    def load_pretrained_encoder(self, path):
        loaded_state_dict= upgrade_state_dict_with_pretrained_weights(
            self.encoder.state_dict(), path,prefix="encoder."
        )
        self.encoder.load_state_dict(loaded_state_dict)
        

    def load_pretrained_decoder(self,path):
        loaded_state_dict= upgrade_state_dict_with_pretrained_weights(
            self.decoder.state_dict(), path,prefix="decoder."
        )
        self.decoder.load_state_dict(loaded_state_dict)

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
        return decoder_out

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def upgrade_state_dict_with_pretrained_weights(
    state_dict: Dict[str, Any], pretrained_checkpoint: str,
    prefix:str = "encoder."
) -> Dict[str, Any]:

    if not os.path.exists(pretrained_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_checkpoint))

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_checkpoint)
    pretrained_state_dict = state["model"]
    ignored= 0
    for key in pretrained_state_dict.keys():
        if prefix in key:
            subkey = key[key.find(prefix)+len(prefix):]
            if subkey not in state_dict:
                print(f"no key {subkey} for {prefix} in current model")
                ignored+=1
            else:
                state_dict[subkey] = pretrained_state_dict[key]
    if ignored ==0:
        print(f"{prefix} initialsed from {pretrained_checkpoint} complete")
    else:
        print(f" initialsing {prefix} from {pretrained_checkpoint}, {ignored} ignored")
    return state_dict




@register_model_architecture("audio_transformer","audio_transformer2")
def base_architecture(args):
    #args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    
    args.conv_type= getattr(args,"conv_type", "shallow2d_base")
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers",12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", 2048
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    # args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    # args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.no_audio_positional_embeddings = getattr(
        args, "no_audio_positional_embeddings", False
    )
    # args.adaptive_input = getattr(args, "adaptive_input", False)
    # args.no_cross_attention = getattr(args, "no_cross_attention", False)
    # args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)

    #args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    #args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.decoder_max_relative_position = getattr(args,"decoder_max_relative_position", -1)
    args.encoder_max_relative_position= getattr(args,"encoder_max_relative_position", -1)

@register_model_architecture("audio_transformer", "audio_transformer_s")
def audio_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("audio_transformer", "randpos_audio_transformer")
def randpos_audio_transformer(args):
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    base_architecture(args)

@register_model_architecture("audio_transformer", "randpos_audio_transformer_s")
def randpos_audio_transformer_s(args):
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    audio_transformer_s(args)

@register_model_architecture("audio_transformer", "nopos_relative_s")
def nopos_relative_s(args):
    args.no_audio_positional_embeddings = True
    args.no_token_positional_embeddings= True
    args.decoder_max_relative_position = getattr(args,"decoder_max_relative_position", 64)
    args.encoder_max_relative_position= getattr(args,"encoder_max_relative_position", 64)
    audio_transformer_s(args)

@register_model_architecture("audio_transformer", "pos_relative_s")
def pos_relative_s(args):
    args.decoder_max_relative_position = getattr(args,"decoder_max_relative_position", 64)
    args.encoder_max_relative_position= getattr(args,"encoder_max_relative_position", 64)
    audio_transformer_s(args)

@register_model_architecture("audio_transformer", "randpos_relative_s")
def randpos_relative_s(args):
    args.rand_pos_encoder = getattr(args, "rand_pos_encoder", 300)
    args.rand_pos_decoder= getattr(args, "rand_pos_decoder", 30)
    args.decoder_max_relative_position = getattr(args,"decoder_max_relative_position", 64)
    args.encoder_max_relative_position= getattr(args,"encoder_max_relative_position", 64)
    audio_transformer_s(args)


@register_model_architecture("audio_transformer", "audio_transformer_smallenc")
def audio_transformer_smallenc(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("transformer","transformer_small")
def transformer_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.dropout = getattr(args, "dropout", 0.1)
    transformer.base_architecture(args)