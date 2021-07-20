SRC=en
TGT=de
mt_dir=/yrfs1/hyperbrain/danliu/snmt/snmt/exp_data/data-bin
st_dir=/yrfs1/hyperbrain/danliu/snmt/snmt/exp_data/must_distill_cat
work_dir=/work/superbrain/danliu/exp_relattn/relattnm32r16
mkdir -p $work_dir
olog_dir=./log_train5
mkdir -p $olog_dir

func_sub(){
	name=$1
	cmd=$2
    train_param=$3
	echo source ~/.bashrc >_train_$name.sh
    echo module load gcc/7.3.0-os7.2 >> _train_$name.sh
    echo module load cuda/10.1-cudnn-7.6.5 >>_train_$name.sh
    echo export PYTHONIOENCODING=utf-8 >> _train_$name.sh
	echo $cmd >>_train_$name.sh
	#return
	 dlp submit -a danliu -x "dlp2-7-141" \
						 -d $name -n train-adistill \
						 -t PtJob --useGpu  \
						 $train_param \
						 -i reg.deeplearning.cn/dlaas/openmpi:cuda9.1-cudnn7 \
						 -e _train_$name.sh -l $olog_dir/$name.txt
    
	#-k TeslaV100-PCIE-12GB 
	#-r superbrain-reserved \
}
param_g2m16="-g 2 -k TeslaV100-PCIE-16GB"
param_g1m16="-g 1 -k TeslaV100-PCIE-16GB"
param_g2m32="-g 2 -k TeslaV100-PCIE-32GB -r superbrain-iwslt-reserved"
param_g1m32="-g 1 -k TeslaV100-PCIE-32GB -r superbrain-iwslt-reserved"

pretrain_model=/work/superbrain/danliu/exp_relattn/asr_pretrain/asr_m32r16/checkpoint_best.pt

waits=( 16 32 80)

for k in ${waits[@]};do
	name=wait${k}
	cmd="fairseq-train $st_dir --source-lang $SRC --target-lang $TGT \
		--encoder-max-relative-position 32 \
		--max-epoch 100 \
		--user-dir rain
		--ddp-backend=no_c10d \
		--main-context 32 --right-context 16 \
		--pretrained-encoder-path $pretrain_model \
		--task s2s --task-type st \
		--decoder-delay-blocks ${k} --decoder-blocks-per-token 8 \
		--bpe-dropout 0.1  \
		--arch online_audio_transformer_waitk --dropout 0.3 --activation-dropout 0.1 \
		--share-decoder-input-output-embed \
		--share-all-embeddings   \
		--optimizer adam --adam-betas '(0.9, 0.98)' \
		--lr 0.0005 --lr-scheduler inverse_sqrt  \
		--warmup-updates 4000 --warmup-init-lr '1e-07' \
		--label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
		--weight-decay 0.0001 \
		--save-dir $work_dir/$name \
		--max-tokens 16000 --update-freq 8 \
		--log-interval 10 --save-interval 4 --log-format simple  --fp16"
	func_sub $name "$cmd" "$param_g2m16"
done
