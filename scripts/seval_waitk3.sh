olog_dir=./log_inferwaitk
mkdir -p $olog_dir

func_sub(){
	name=$1
	cmd=$2
	echo source ~/.bashrc >_seval_$name.sh
    echo module load gcc/7.3.0-os7.2 >> _seval_$name.sh
    echo export PYTHONIOENCODING=utf-8 >> _seval_$name.sh
    #echo export LD_LIBRARY_PATH=/opt/lib/cuda-9.2.1/lib64:/opt/lib/mvapich2.2-os7.2-cuda/lib:/opt/lib/liblmdb-0.9/lib:/opt/lib/protobuf-2.5/lib:/opt/lib/gflag-1.4.0/lib:/opt/lib/glog-0.3.3/lib:/opt/lib/boost_1_58_0/stage/lib:/opt/tool/icu-56/lib:/opt/lib/opencv-2.4.9/lib:/opt/lib/log4cplus-1.2.0-rc3/lib:/opt/tool/intel/lib/intel64:/opt/tool/intel/mkl/lib/intel64:/opt/lib/cuda-8.0/lib64:/home/superbrain/danliu/lib/lib:/home/superbrain/danliu/lib/usr/local/lib64 >>_train_$name.sh
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

audio_cfg=./exp_data/eval_/audio_cfg
text_cfg=./exp_data/eval_/text_cfg
data_dir=./exp_data/eval_
srcfile=$data_dir/tst-COMMON.en
wavfile=$data_dir/tst-COMMON.list
tgtfile=$data_dir/tst-COMMON.de
work_dir=/work/superbrain/danliu/exp_relattn/relattnm32r16
port=43572


st_models=(wait16 wait16 wait32 wait64 wait64 wait80 wait80 )
wait_blocks=(1 2 4 6 8 10 12)
step_read=(1 1 1 1 1 1 1)

for ((i=0;i <${#st_models[@]}; i++));
do
	model=${st_models[$i]}
	k=${wait_blocks[$i]}
	step=${step_read[$i]}
	
 	
	modelfile=${work_dir}/${model}/checkpoint_best.pt

	name=${model}_${k}_beam5_forcast2
	outdir=$olog_dir/$name
	cmd="simuleval --agent ./rain/simul/speech_waitk.py \
		--task-type st --train-dir exp_data/must_filtered2 \
		--source-lang en --target-lang de \
		--source $wavfile --target $tgtfile \
		--data-type speech --task s2s --model-path $modelfile \
		--wait-blocks $k --step-read-blocks $step --step-generate 1 --step-forecast 0 \
		--output $outdir  --port $port --timeout 100 --beam 5 --step-forecast 2 --stop-early"
	func_sub $name "$cmd"
	port=$[port+1]


	name=${model}_${k}_beam5_naiive
	outdir=$olog_dir/$name
	cmd="simuleval --agent ./rain/simul/speech_waitk.py \
		--task-type st --train-dir exp_data/must_filtered2 \
		--source-lang en --target-lang de \
		--source $wavfile --target $tgtfile \
		--data-type speech --task s2s --model-path $modelfile \
		--wait-blocks $k --step-read-blocks $step --step-generate 1 --step-forecast 0 \
		--output $outdir  --port $port --timeout 100 --beam 5 --naive-waitk   --stop-early"
	func_sub $name "$cmd"
	port=$[port+1]

done
