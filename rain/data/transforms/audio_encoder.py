
import torch

from torch import Tensor
import numpy as np
from typing import List, Dict
import yaml
import os.path as op
import os

def _get_fbank(waveform:Tensor, sample_rate= 16000,n_bins=80):
    waveform = waveform*( 2**15)
    
    import torchaudio.compliance.kaldi as ta_kaldi
    features = ta_kaldi.fbank(waveform, num_mel_bins=n_bins, sample_frequency=sample_rate)
    return features


def _load_wav(wave_path:str, start= None, duration = None):
    """
    Args:
        start|duration: load from `start` secs, get `duration` secs 
    return: waveform , sample_rate
    """
    import torchaudio
    sample_rate= torchaudio.info(wave_path)[0].rate
    if start is None:
        return torchaudio.load(wave_path)
    offset= int(float(start)*sample_rate)
    num_frames= int(float(duration)*sample_rate)
    return torchaudio.load(wave_path,offset=offset, num_frames=num_frames)


def save_features(features:Tensor, output_path):
    features= features.numpy().astype(np.float32)
    np.save(output_path, features)

class Transform(object):
    def __call__(self,fea):
        return fea

class Whiten(Transform):
    def __init__(self, config:Dict):
        assert "whiten" in config, "param `whiten` needed"
        whiten_parms= np.load( config["whiten"])
        self.mean= torch.from_numpy(whiten_parms["mean"]).unsqueeze(0)
        self.std = torch.from_numpy(whiten_parms["std"]).unsqueeze(0)
    
    def __call__(self, fea):
        fea= (fea - self.mean)/self.std
        return fea


class TFMask(Transform):
    def __init__(self, config:Dict):
        self.tmask_max= config.get("tmask_max", 100)
        self.tmask_p = config.get("tmask_p", 1.)
        self.fmax_max= config.get("fmax_max", 27)
        self.tmask_step= config.get("tmask_step",1)
        self.fmask_step = config.get("fmask_step",1)
        # torchaudio implementation is strange, maybe bsz,freq,time
        #import torchaudio
        # self.tmask= torchaudio.transforms.TimeMasking(self.tmask_max)
        # self.fmask = torchaudio.transforms.FrequencyMasking(self.fmax_max)
        from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
        self.spec_aug= SpecAugmentTransform(
            time_warp_w = 0,
            freq_mask_n = self.fmask_step,
            freq_mask_f = self.fmax_max,
            time_mask_n = self.tmask_step,
            time_mask_t =self.tmask_max,
            time_mask_p = self.tmask_p ,
            mask_value=0.0
        )
        
    
    def __call__(self, fea):
        return torch.from_numpy(self.spec_aug(fea.numpy()))


class CompositeTransform(object):
    def __init__(self, transforms:List):
        self.transforms= transforms
    
    def __call__(self, fea):
        for trans in self.transforms:
            fea= trans(fea)
        return fea


def package_transforms(
    out_path:str, fea_mean:np.ndarray, fea_std:np.ndarray,
    tmask_max:int = 100, tmask_step:int =1,
    fmask_max:int = 27, fmask_step:int =1,
):
    if not op.exists(out_path):
        os.makedirs(out_path)
    whitenfile= op.join(out_path, "whiten.npz")
    np.savez(whitenfile, mean=fea_mean, std= fea_std)
    config={
        "whiten":"whiten.npz",
        "tmask_max":tmask_max, "tmask_step":tmask_step,
        "fmask_max": fmask_max, "fmask_step":fmask_step
    }
    with open(op.join(out_path, "config.yaml"),"w") as f:
        yaml.dump(config, f)

TRANSFORM_MAPPING={
    "whiten":Whiten,
    "tfmask":TFMask
}

def build_audio_transforms(at_path:str, transform_names:List[str] = ["whiten", "tfmask"]):
    def check_file(path:str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} not exists')
        return path
    cfgfile= check_file(op.join(at_path, "config.yaml"))
    with open(cfgfile,"r") as f:
        config= yaml.load(f, yaml.FullLoader)
    if "whiten" in config:
        whitenfile= check_file(op.join(at_path, config["whiten"]))
        config["whiten"] = whitenfile
    transforms=[]
    for trans_name in transform_names:
        if trans_name not in TRANSFORM_MAPPING:
            print(f"unknown audio transform {trans_name}, ignore")
            continue
        trans= TRANSFORM_MAPPING[trans_name](config)
        transforms.append(trans)
    if len(transforms) ==0:
        return Transform()
    return CompositeTransform(transforms)
    
    

