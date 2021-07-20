import csv
import io,os
import numpy as np
import torch
from fairseq.data import FairseqDataset
from rain.data.transforms import audio_encoder

def read_from_uncompressed_zip(file_path, offset, file_size) -> bytes:
    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read(file_size)
    return data

def get_features_from_uncompressed_zip(
    path, byte_offset, byte_size
):
    assert path.endswith(".zip")
    data = read_from_uncompressed_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    features = np.load(f)
    features= torch.from_numpy(features)
    return features

class FbankZipDataset(FairseqDataset):
    def __init__(
        self,
        tsvfile:str,
        transform:audio_encoder.Transform,
        mel_bins=80,
    ):
        """
            Dataset to read fbank zip file directly, maybe same speed with LMDB and less memory cache
            tsvfile headers: `fbk_path`, `speaker`, `frames`
            wave_path should be `zip_path:offset:size`
        """
        tsvfile= tsvfile +".tsv"
        self.transform = transform
        self.mel_bins=mel_bins
        self.fbk_paths, self.speakers, self.num_frames=[],[],[]
        data_dir= os.path.dirname(tsvfile)
        with open(tsvfile) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            for sample in reader:
                _path,*extra= sample["fbk_path"].split(":")
                _path= os.path.join(data_dir, _path)
                self.fbk_paths.append((_path, int(extra[0]), int(extra[1])))
                self.speakers.append(sample["speaker"])
                self.num_frames.append(int(sample["frames"]))
        self.num_frames = np.array(self.num_frames)

    def _check_index(self, index):
        if index <0 or index > len(self):
            raise IndexError(f"index {index} out of range")
    
    def __len__(self):
        return len(self.fbk_paths)

    def __getitem__(self, index):
        self._check_index(index)
       
        fbk = get_features_from_uncompressed_zip(*self.fbk_paths[index])
        fbk = self.transform(fbk)
        speaker= self.speakers[index]
        return {"fbank":fbk, "speaker":speaker}
    
    @property
    def sizes(self):
        return self.num_frames
    
    def num_tokens(self, index):
        return self.num_frames[index]
    
    def size(self, index):
        return self.num_frames[index]
    
    @staticmethod
    def exists(path):
        return os.path.exists(path+".tsv")



    


        