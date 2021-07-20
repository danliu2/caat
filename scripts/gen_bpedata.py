import argparse
import sys
sys.path.append("../")
import rain
from rain.data.transforms import text_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bpe", help= "name for bpe model")
    parser.add_argument("--dropout", type=float, default= 0.0, help = "alpha for bpe dropout")
    parser.add_argument("input", type=str, help= "data file")
    parser.add_argument("output", type=str, help="outputfile")
    
    args = parser.parse_args()
    alpha= args.dropout
    sampling= alpha >0.001
    encoder= text_encoder.TextEncoder(args.model)
    with open(args.input, "r", encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as fout:
        for line in f:
            line= encoder.encode(line,sampling= sampling, alpha= alpha)
            fout.write("{}\n".format(line))

if __name__ == "__main__":
    main()