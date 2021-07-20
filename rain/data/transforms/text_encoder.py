import yaml
import sacremoses
import sentencepiece as spm
import re
import zipfile
import os
from fairseq.data import Dictionary
from typing import List, Dict
import shutil


SPACE = chr(32)
SPACE_ESCAPE = chr(9601)


class SpaceSplitter(object):
    def __init__(self,  *unused):
        self.space_tok = re.compile(r"\s+")

    def encode(self, x: str) -> str:
        return self.space_tok.sub(" ", x)

    def decode(self, x: str) -> str:
        return x
    
    def split(self, x:str)->List[str]:
        return re.split(self.space_tok, x)


class MosesSplitter(object):
    def __init__(self, lang="en"):
        self.tok = sacremoses.MosesTokenizer(lang)
        self.detok= sacremoses.MosesDetokenizer(lang)
    
    def encode(self, x:str) -> str:
        return self.tok.tokenize(
            x,
            aggressive_dash_splits=True,
            return_str=True,
            escape=False,
        )
    
    def split(self, x:str)->str:
        return self.tok.tokenize(
            x,
            aggressive_dash_splits=True,
            return_str=False,
            escape=False
        )

    
    def decode(self, x:str)->str:
        return self.detok.detokenize(x.split())

def get_splitter(sp_type):
    splitters={"space":SpaceSplitter, "moses":MosesSplitter}
    return splitters[sp_type] if sp_type in splitters else splitters["space"]

class TextEncoder(object):
    def __init__(self, spm_model_prefix, splitter = "space", lang="en"):
        self.splitter = get_splitter(splitter)(lang)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model_prefix +".model")
    
    def encode(self, x:str, sampling=False, alpha=0.0) ->str:
        x= self.splitter.encode(x)
        if not sampling:
            x = SPACE.join(self.sp.EncodeAsPieces(x))
        else:
            assert alpha >1e-5 and alpha <0.9, "sampling need alpha in (0,1)"
            x = self.sp.Encode(
                x, out_type=str, 
                enable_sampling=True,
                alpha = alpha, nbest_size=-1
            )
            x= " ".join(x)
        return x
    
    def decode(self, x:str):
        x = x.replace(SPACE, "").replace(SPACE_ESCAPE, SPACE)
        x= self.splitter.decode(x)
        return x
    
    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            return True
        return x.startswith(SPACE_ESCAPE)
    
    def split_to_word(self,x:str)->List[str]:
        return self.splitter.split(x)
    
    def word_to_ids(self, word:str, sampling=False, alpha= 0.0):
        if not sampling:
            return self.sp.EncodeAsIds(word)
        else:
            return self.sp.Encode(
                word, out_type=int,
                enable_sampling=True,
                alpha= alpha, nbest_size= -1
            )
    

class VocabManager(object):
    def __init__(self, vocab_path:str):
        if not os.path.isdir(vocab_path):
            raise FileNotFoundError(f'{vocab_path} should be directory')

        def check_file(path:str):
            if not os.path.isfile(path):
                raise FileNotFoundError(f'{path} not exists')
            return path
        
        cfgfile = check_file(os.path.join(vocab_path, "config.yaml"))
        with open(cfgfile, "r") as f:
            cfg = yaml.load(f, Loader= yaml.FullLoader)
        
        langs= cfg['langs']
        self.vocabs= dict()
        self.encoders=dict()
        for lang in langs:
            assert lang in cfg, f"language {lang} have no config info"
            _sub= cfg[lang]
            vocabfile = check_file(os.path.join(vocab_path,_sub["vocab"]))
            self.vocabs[lang]= Dictionary.load(vocabfile)
            self.encoders[lang]= TextEncoder(
                os.path.join(vocab_path,_sub["spm"]),
                _sub.get("splitter", "space"),
                lang = lang
            )
    
    def get_dictionary(self, lang):
        if lang not in self.vocabs:
            raise ValueError(f"need dictionary for {lang}")
        return self.vocabs[lang]
    
    def get_encoder(self, lang):
        if lang not in self.encoders:
            raise ValueError(f"need dictionary for {lang}")
        return self.encoders[lang]

from multiprocessing import cpu_count
UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1


def train_bpe(srcfile:str, model_prefix:str, vocab_size= 30000):
    arguments = [
        f"--input={srcfile}",
        f"--model_prefix={model_prefix}",
        f"--model_type=bpe",
        f"--vocab_size={vocab_size}",
        "--character_coverage=1.0",
        f"--num_threads={cpu_count()}",
        f"--unk_id={UNK_TOKEN_ID}",
        f"--bos_id={BOS_TOKEN_ID}",
        f"--eos_id={EOS_TOKEN_ID}",
        f"--pad_id={PAD_TOKEN_ID}",
    ]
    spm.SentencePieceTrainer.Train(" ".join(arguments))


def package_vocabs(
    package_path:str,
    langs:List[str], vocabs:List[str], spm_models:List[str],
    splitters:List[str]
):
    """
        Package dictionaries,bpe models to one directory,
        if dict file names for two languages are same, assume they share same dicts(and bpe models)
        Args:
        package_path:
        langs: 
        vocabs:
        spm_models
        splitters
    """
    if not os.path.exists(package_path):
        os.makedirs(package_path)
    vocab_lang, spm_lang= {},{}
    config = dict(langs= langs)
    for lang, vocab, spm, splitter in zip(langs, vocabs,spm_models, splitters):
        voc_bname= os.path.basename(vocab)
        spm_bname = os.path.basename(spm)
        if voc_bname in vocab_lang:
            assert spm_bname in spm_lang, \
                f"lang {lang} shared vocab with {vocab_lang[voc_bname]}, but with different bpe"
        else:
            vocab_lang[voc_bname] = lang
            spm_lang[spm_bname] = lang
            shutil.copy(vocab, os.path.join(package_path, voc_bname))
            shutil.copy(spm +".model", os.path.join(package_path, spm_bname +".model"))
            shutil.copy(spm +".vocab", os.path.join(package_path, spm_bname +".vocab"))
        config[lang]={"spm":spm_bname, "vocab":voc_bname, "splitter":splitter}
    with open(os.path.join(package_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    



    
         
