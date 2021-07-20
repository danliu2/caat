import argparse
import sys
sys.path.append("../")
import rain
from rain.data.transforms import text_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size",type=int, default= 30000, help= "vocab size for spm")
    parser.add_argument("--model", type=str, default="bpe", help= "name for bpe model")
    parser.add_argument("data", type=str, help= "data file")
    args = parser.parse_args()
    text_encoder.train_bpe(args.data, args.model, args.vocab_size)

if __name__ == "__main__":
    main()