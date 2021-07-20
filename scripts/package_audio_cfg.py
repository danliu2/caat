import argparse
import numpy as np
import sys
sys.path.append("../")
import rain
from rain.data.transforms import audio_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm",type=str,  help= "mean-stdfile")

    parser.add_argument("outdir", type= str, help = "outputdir")
    
    args = parser.parse_args()
    norm = np.load(args.norm)
    audio_encoder.package_transforms(
        args.outdir, norm["mean"],norm["std"], 
        tmask_max= 100,tmask_step=2,
        fmask_max=27, fmask_step=2
    )

if __name__ == "__main__":
    main()