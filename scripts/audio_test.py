import argparse
import logging
import os
import os.path as op
import shutil
from typing import Tuple
import yaml
import pandas as pd
import csv
import numpy as np
import sys
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
import zipfile
from glob import glob
from rain.data.transforms import audio_encoder
import torchaudio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default= "tst-COMMON", help= "dataset prefix")
    parser.add_argument("raw_dir", type=str, help= "raw data dir")
    parser.add_argument("output_dir", type=str, help = "dest dir")
    args = parser.parse_args()
    txt_dir= op.join(args.raw_dir, "txt")
    wav_dir=op.join(args.raw_dir, "wav")
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(op.join(txt_dir, f"{args.prefix}.de"), op.join(args.output_dir, f"{args.prefix}.de"))
    shutil.copy(op.join(txt_dir, f"{args.prefix}.en"), op.join(args.output_dir, f"{args.prefix}.en"))
    yamlfile= op.join(txt_dir, f"{args.prefix}.yaml")
    
    wlistfile= op.join(args.output_dir, f"{args.prefix}.list")
    owav_dir= op.join(args.output_dir, "wav")
    os.makedirs(owav_dir, exist_ok=True)
    segments=[]
    with open(yamlfile,encoding="utf-8") as f:
        for i,line in enumerate(f):
            #segments.extend(yaml.load(StringIO(line), Loader =yaml.BaseLoader))
            line = line.strip()[3:-1]
            parts= line.split(",")
            segment = {
                "duration":float(parts[0].split(":")[1].strip()),
                "offset":float(parts[1].split(":")[1].strip()),
                "speaker_id":str(parts[2].split(":")[1].strip()),
                "wav":str(parts[3].split(":")[1].strip()),
            }
            segments.append(segment)
    with open(wlistfile, "w") as fout:
        for i, segment in tqdm(enumerate(segments)):
            start_sec,duration_sec = segment["offset"],segment["duration"]
            wave_path = op.join(wav_dir, segment["wav"])
            wav_name = op.splitext(segment["wav"])[0]
            utt_id="{:05d}_{}".format(i, wav_name)
            outfile= "{}/{}.wav".format(owav_dir, utt_id)
            fout.write(f"{outfile}\n")
           
            wav,samp_rate= audio_encoder._load_wav(wave_path, start_sec, duration_sec)
            fbank = audio_encoder._get_fbank(wav, samp_rate, n_bins=80)
            torchaudio.save(outfile, wav,samp_rate)

   
if __name__ == "__main__":
    main()