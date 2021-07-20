import argparse
import sys
import rain
import os.path as op
import os
import csv
from rain.data.transforms import text_encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-audio", type=int, default=2000, help= "max length for audio")
    parser.add_argument("--max-text", type=int, default= 256, help = "max length for text")
    parser.add_argument("--min-length", type= int, default= 50, help = "min audio length")
    parser.add_argument("--max-text-ratio", default=0.2, type=float, help = "max text length by audio length" )
    parser.add_argument("input", type=str, help= "data file")
    parser.add_argument("output", type=str, help="outputfile")
    args = parser.parse_args()
    en_file,de_file, audio_file= "train.en-de.en.raw", "train.en-de.de.raw", "train.en-de.audio.tsv"
    fen = open(op.join(args.input, en_file), "r", encoding="utf-8")
    fde = open(op.join(args.input, de_file), "r", encoding= "utf-8")
    faudio = open(op.join(args.input, audio_file), "r", encoding="utf-8")

    fen_out = open(op.join(args.output, en_file), "w", encoding="utf-8")
    fde_out = open(op.join(args.output, de_file), "w", encoding= "utf-8")
    faudio_out = open(op.join(args.output, audio_file), "w", encoding="utf-8")
    fout_err= open(op.join(args.output, "illegal.txt"), "w", encoding="utf-8")
    
    faudio_out.write("{}\n".format( faudio.readline().strip()))
    ignores,total= 0,0
    for en,de,auinfo in zip(fen,fde, faudio):
        en,de= en.strip(),de.strip()
        auinfo = auinfo.strip()
        en_len = len(en.strip().split())
        de_len = len(de.strip().split())
        au_frames= int(auinfo.strip().split()[1])
        err_info= ""
        if de_len > en_len*4 or en_len > de_len*4:
            err_info+= "src_text_mismatch"
        if au_frames > args.max_audio:
            err_info+= "|audio_len"
        if en_len > args.max_text:
            err_info += "|text_len"
        if au_frames < args.min_length:
            err_info += "|audio_small"
        if en_len /au_frames > args.max_text_ratio:
            err_info += "|text_audio_ratio"
        if err_info =="":
            fen_out.write(f"{en}\n")
            fde_out.write(f"{de}\n")
            faudio_out.write(f"{auinfo}\n")
        else:
            fout_err.write(f"{err_info}\t{auinfo}\t{en}\t{de}\n")
            ignores +=1
        total +=1
    print(f"total {total} samples, {ignores} ignored")
    fen.close()
    fde.close()
    faudio.close()
    fen_out.close()
    fde_out.close()
    faudio_out.close()
    fout_err.close()

if __name__ == "__main__":
    main()