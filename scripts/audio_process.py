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


log = logging.getLogger(__name__)


def create_zip(data_root, zip_path):
    cwd = os.path.abspath(os.curdir)
    os.chdir(data_root)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as f:
        for filename in tqdm(glob("*.npy")):
            f.write(filename)
    os.chdir(cwd)

def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78

def get_zip_manifest(zip_root, zip_filename):
    zip_path = op.join(zip_root, zip_filename)
    with zipfile.ZipFile(zip_path, mode="r") as f:
        info = f.infolist()
    manifest = {}
    for i in tqdm(info):
        utt_id = op.splitext(i.filename)[0]
        offset, file_size = i.header_offset + 30 + len(i.filename), i.file_size
        manifest[utt_id] = f"{zip_filename}:{offset}:{file_size}"
        with open(zip_path, "rb") as f:
            f.seek(offset)
            data = f.read(file_size)
            assert len(data) > 1 and is_npy_data(data)
    return manifest

Name_Mapping={"train":"train", "dev":"valid", "tst-COMMON":"test", "tst-HE":"test1"}
from io import StringIO
import psutil
# train, dev, tst-COMMON, tst-HE
def process(input_dir, output_dir, src_lang="en", tgt_lang="de", split= "train", count_mvn=False):
    os.makedirs(output_dir, exist_ok= True)
    fbk_dir= op.join(output_dir, "fbank")
    os.makedirs(fbk_dir, exist_ok=True)
    txt_dir=op.join(input_dir, "txt")
    wav_dir= op.join(input_dir, "wav")
    segments=[]
    
    with open(op.join(txt_dir, f"{split}.yaml"),encoding="utf-8") as f:
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
            

    with open(op.join(txt_dir, f"{split}.{src_lang}"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            segments[i]["src"] = line.strip()
    with open(op.join(txt_dir, f"{split}.{tgt_lang}"),encoding="utf-8") as f:
        for i, line in enumerate(f):
            segments[i]["tgt"] = line.strip()

    print("before load,memory cost {}".format( psutil.Process(os.getpid()).memory_info().rss))
    for i, segment in tqdm(enumerate(segments)):
        start_sec,duration_sec = segment["offset"],segment["duration"]
        wave_path = op.join(wav_dir, segment["wav"])
        wav_name = op.splitext(segment["wav"])[0]
        utt_id="{:05d}_{}".format(i, wav_name)
        segments[i]["utt_id"] = utt_id
        segments[i]["frames"] = int(duration_sec*100)
    '''
    stat1 = np.zeros(80)
    stat2= np.zeros(80)
    frames= 0
    for i, segment in tqdm(enumerate(segments)):
        start_sec,duration_sec = segment["offset"],segment["duration"]
        wave_path = op.join(wav_dir, segment["wav"])
        wav_name = op.splitext(segment["wav"])[0]
        utt_id="{:05d}_{}".format(i, wav_name)
        out_path = op.join(fbk_dir, utt_id +".npy")
        if os.path.exists(out_path):
            continue
        wav,samp_rate= audio_encoder._load_wav(wave_path, start_sec, duration_sec)
        fbank = audio_encoder._get_fbank(wav, samp_rate, n_bins=80)
        fbank_np = fbank.numpy()
        stat1+= fbank_np.sum(0)
        stat2+= (fbank_np**2).sum(0)
       
        frames+= fbank.shape[0]
        
        segments[i]["utt_id"] = utt_id
        segments[i]["frames"] = fbank.shape[0]
        audio_encoder.save_features(fbank, out_path)
        del fbank, fbank_np
        if (i+1) % 200 == 0:
            print("processed{},memory cost {}\n".format(i, psutil.Process(os.getpid()).memory_info().rss))
            
          
    if count_mvn:
        mean= stat1/frames
        var = stat2/frames - mean**2
        std= np.sqrt(var+1e-8)
        mvn_file= op.join(output_dir, "mvn.npz")
        np.savez(mvn_file, mean= mean.astype(np.float32), std=std.astype(np.float32))
    '''
    outsplit=Name_Mapping[split]
    print("ZIPing features...")
    zip_fname=f"fbank_{outsplit}.zip"
    
    zip_path= op.join(op.realpath(output_dir), zip_fname)
    #create_zip(fbk_dir, zip_path)
    
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(output_dir,zip_fname)
    
    audio_info={"fbk_path":[], "speaker":[], "frames":[]}
    src_lines, tgt_lines=[],[]
    ignores= []
    for segment in segments:
        if segment["utt_id"] not in zip_manifest:
            ignores.append(segment["utt_id"])
            continue
        if segment["frames"]<3 or segment["frames"]> 6000:
            ignores.append(segment["utt_id"])
            continue
        src_lines.append(segment["src"])
        tgt_lines.append(segment['tgt'])
        audio_info["fbk_path"].append(zip_manifest[segment["utt_id"]])
        audio_info["speaker"].append(segment["speaker_id"])
        audio_info["frames"].append(segment["frames"])
    print(f"total {len(segments)} samples processed, {len(ignores)} ignored:")
    print("\n".join(ignores))

    
    out_prefix= op.join(output_dir,f"{outsplit}.{src_lang}-{tgt_lang}")
    df = pd.DataFrame.from_dict(audio_info)
    audio_path = out_prefix + ".audio.tsv"
    df.to_csv(
        audio_path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    src_path = out_prefix + f".{src_lang}"
    tgt_path = out_prefix +f".{tgt_lang}"
    with open(src_path, "w", encoding="utf-8") as fout:
        for line in src_lines:
            fout.write("{}\n".format(line))
    with open(tgt_path, "w", encoding="utf-8") as fout:
        for line in tgt_lines:
            fout.write("{}\n".format(line))
    print(f"remove {fbk_dir}")
    shutil.rmtree(fbk_dir)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", type=str, default="en", help= "source lang")
    parser.add_argument("--tgt-lang", type=str, default="de", help= "target lang")
    parser.add_argument("input_dir", type=str,  help= "raw data dir")
    parser.add_argument("output_dir", type=str, help = "dest dir")
    args = parser.parse_args()
    datasets=[ "dev","tst-COMMON", "tst-HE","train",]
    datasets=[ "train",]
    for data in datasets:
        indir= op.join(args.input_dir, data)
        count_mvn= data=="train"
       
        process(indir, args.output_dir,src_lang=args.src_lang, tgt_lang=args.tgt_lang, split= data, count_mvn=count_mvn)

if __name__ == "__main__":
    main()