import argparse
import sys

import rain
from rain.data.transforms import text_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs",type=str, default= "en,de", help= "lang ids to pack")
    parser.add_argument("--vocabs", type=str, help= "vocab files")
    parser.add_argument("--encoders", type= str,help = "encoder files to pack")
    parser.add_argument("outdir", type= str, help = "outputdir")
    
    args = parser.parse_args()
    langs= args.langs.split(",")
    vocabs= args.vocabs.split(",")
    encoders = args.encoders.split(",")
    text_encoder.package_vocabs(args.outdir, langs, vocabs, encoders, ["space", "space"])

if __name__ == "__main__":
    main()