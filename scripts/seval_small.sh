olog_dir=./log_inferother_smallstep
mkdir -p $olog_dir

func_sub(){
	name=$1
	cmd=$2
	echo source ~/.bashrc >_seval_$name.sh
    echo module load gcc/7.3.0-os7.2 >> _seval_$name.sh
    echo export PYTHONIOENCODING=utf-8 >> _seval_$name.sh
 
	echo export LD_LIBRARY_PATH=/home/superbrain/danliu/lib/libsndfile/lib:\$LD_LIBRARY_PATH >>_seval_$name.sh
	echo $cmd >>_seval_$name.sh
	#return
	dlp submit -a danliu  \
						 -d $name -n seval \
						 -t PtJob --useGpu  \
						 -g 1 -k TeslaV100-PCIE-12GB \
						 -i reg.deeplearning.cn/dlaas/openmpi:cuda9.1-cudnn7 \
						 -e _seval_$name.sh -l $olog_dir/$name.txt
	#-k TeslaV100-PCIE-12GB 
	#-r superbrain-reserved \
}

st_dir=/yrfs1/hyperbrain/danliu/snmt/snmt/exp_data/
audio_cfg=$st_dir/eval_/audio_cfg
text_cfg=$st_dir/eval_/text_cfg
data_dir=$st_dir/eval_
srcfile=$data_dir/tst-COMMON.en
wavfile=$data_dir/tst-COMMON.list
tgtfile=$data_dir/tst-COMMON.de
port=52345


work_dir=/work/superbrain/danliu/exp_relattn/snmt
st_models=(cat_step16 cat_step16 cat_lat0.5_step16 cat_lat0.5_step16 )


st_params=("--step-read-block 1" "--step-read-block 2"  "--step-read-block 1" "--step-read-block 2" )
pnames=("step1" "step2" "step1" "step2")

for ((i=0;i <${#st_models[@]}; i++)); do
	model=${st_models[$i]}
	param=${st_params[$i]}
	modelfile=${work_dir}/${model}/checkpoint_best.pt
    pname=${pnames[$i]}
	name=${model}_${pname}
	outdir=$olog_dir/$name
	cmd="simuleval --agent ./rain/simul/speech_fullytransducer_agent.py --timeout 100  \
		--task-type st --train-dir exp_data/must_filtered2 \
		--source-lang en --target-lang de \
		--source $wavfile --target $tgtfile \
		--data-type speech --task s2s --model-path $modelfile \
		--output $outdir --port $port --timeout 100 \
		--max-len-a 0.043 --len-scale 0.7 --len-penalty 0 --max-len-b -5 \
		--bos-bias 0 --intra-beam 5 --gen-beam 0.5 \
		--inter-beam 1 --decoder-step-read 256 --eager \
			$param"
	
	port=$[port+1]
	func_sub $name "$cmd"
done
