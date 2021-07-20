SRC=en
TGT=de

mt_dir=./exp_data/data-bin
st_dir=./exp_data/must_distill
work_dir=/work/superbrain/danliu/exp_relattn/asr_pretrain
mkdir -p $work_dir

olog_dir=./log_asr_pretrain
mkdir -p $olog_dir

func_sub(){
	name=$1
	cmd=$2
	echo source ~/.bashrc >_train_$name.sh
    echo module load gcc/7.3.0-os7.2 >> _train_$name.sh
    echo module load cuda/10.1-cudnn-7.6.5 >>_train_$name.sh
    echo export PYTHONIOENCODING=utf-8 >> _train_$name.sh
	echo $cmd >>_train_$name.sh
	#return
	dlp submit -a danliu  \
						 -d $name -n train \
						 -t PtJob --useGpu  \
						 -g 2 -k TeslaV100-PCIE-16GB \
						 -i reg.deeplearning.cn/dlaas/openmpi:cuda9.1-cudnn7 \
						 -e _train_$name.sh -l $olog_dir/$name.txt
	#-k TeslaV100-PCIE-12GB 
	#-r superbrain-reserved \
}

name=asr_m32r16
cmd="fairseq-train $st_dir --source-lang $SRC --target-lang $TGT \
    --encoder-max-relative-position 32 \
    --max-epoch 100 \
    --user-dir rain \
    --ddp-backend=no_c10d \
    --task s2s --task-type asr \
    --bpe-dropout 0.1  \
    --arch online_audio_transformer_offline --dropout 0.3 --activation-dropout 0.1 \
    --main-context 32 --right-context 16 \
    --share-decoder-input-output-embed \
    --share-all-embeddings   \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt  \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --weight-decay 0.0001 \
    --save-dir $work_dir/$name \
    --max-tokens 20000 --update-freq 8 \
    --log-interval 10 --save-interval 4 --log-format simple  --fp16"

func_sub $name "$cmd"