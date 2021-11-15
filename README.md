# RAIN Simultaneous Speech Translation
This is the implementation of Cross Attention Augmented Transducer (CAAT). If you found bugs or other questions, fill free to discuss with us by issues or mail to danliu@mail.ustc.edu.cn.

<!-- ## Origin of the package name "rain"
 Don't be confused, it just a typo on "TRAIN". -->


## Installation
Our codes relies on PyTorch, Numpy and Fairseq. Besides, we modified warp-transducer for CAAT loss calculation, if you have already installed it, you should uninstall it first, and reinstall the version in sub-dir warp-transducer as follows :

```bash
cd warp_transducer
mkdir build & cd build
cmake ..
make install
cd ../pytorch_binding
pip install -e .
```

## Experiments on MuST-C speech-to-text simultaneous translation

Preprocessing training data: set MUSTC_DIR to the raw MuST-C dataset path,  DATA_DIR to the path for processed data:

```bash
bash scripts/preporcess_mustc.sh $MUSTC_DIR $DATA_DIR
```
Train offline text-to-text translation model:
```bash
SRC=en
TGT=de
mt_dir=model/mt_base
mt_dir=$DATA_DIR/data-bin
fairseq-train $mt_dir --source-lang $SRC --target-lang $TGT \
    --max-epoch 100 \
    --user-dir rain
    --ddp-backend=no_c10d \
    --task dropout_translation \
    --bpe-dropout 0.1 --src-encoder $DATA_DIR/mustc_en-de/bpe --tgt-encoder $DATA_DIR/mustc_en-de/bpe \
    --arch transformer_small --dropout 0.3 --activation-dropout 0.1 \
    --share-decoder-input-output-embed \
    --share-all-embeddings   \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt  \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --weight-decay 0.0001 \
    --save-dir $mt_dir \
    --max-tokens 4096 \
    --update-freq 8   \
    --eval-bleu \
    --eval-bleu-args '{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}' \
    --eval-bleu-detok space \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
    --log-interval 10 --save-interval 4 --log-format simple 
```
generate sequence distillation training data:
```bash
    fairseq-interactive $mt_dir --user-dir rain \
            --bpe sentencepiece --sentencepiece-model $DATA_DIR/mustc_en-de/text_cfg/bpe.model \
            --task translation \
            --path $mt_dir/checkpoint_best.pt \
            --buffer-size 2000 --batch-size 128 \
            --input $srcfile
            --beam 5  > tmp.txt
    grep ^D tmp.txt | cut -f3 >tmp2.txt
    in_dir=$DATA_DIR/mustc_en-de
    out_dir=$DATA_DIR/mustc_distill
    cp -r $in_dir/audio_cfg $out_dir
    cp -r $in_dir/text_cfg $out_dir
    cp -r $in_dir/test* $out_dir
    cp -r $in_dir/valid* $out_dir
    ln -s $in_dir/fbank_test.zip $out_dir/fbank_test.zip
    ln -s $in_dir/fbank_test1.zip $out_dir/fbank_test1.zip
    ln -s $in_dir/fbank_valid.zip $out_dir/fbank_valid.zip
    ln -s $in_dir/fbank_train.zip $out_dir/fbank_train.zip
    cat $in_dir/train.en-de.en.raw $in_dir/train.en-de.en.raw >$out_dir/train.en-de.en.raw
    cat $in_dir/train.en-de.audio.csv $in_dir/train.en-de.audio.csv >$out_dir/train.en-de.audio.csv
    cat $in_dir/train.en-de.de.raw tmp2.txt >$out_dir/train.en-de.de.raw
```
Pretrain encoder with speech recognition task:
```bash
asr_dir=model/asr
fairseq-train $DATA_DIR/mustc_distill --source-lang $SRC --target-lang $TGT \
    --encoder-max-relative-position 32 \
    --max-epoch 100 \
    --user-dir rain \
    --ddp-backend=no_c10d \
    --task s2s --task-type asr \
    --bpe-dropout 0.1  \
    --arch online_audio_transformer_offline --dropout 0.3 --activation-dropout 0.1 \
    --main-context 32 --right-context 16 \
    --share-decoder-input-output-embed-share-all-embeddings   \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt  \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --weight-decay 0.0001 \
    --save-dir $asr_dir \
    --max-tokens 20000 --update-freq 8 \
    --log-interval 10 --save-interval 4 --log-format simple  --fp16"
```
Train final CAAT simultanslation model
```bash
    sst_dir=model/sst
    pretrain_model=$asr_dir/checkpoint_best.pt
    step=32
    latency_scale=1.0
    fairseq-train $DATA_DIR/mustc_distill  --source-lang $SRC --target-lang $TGT \
		--encoder-max-relative-position 32 \
        --max-audio-positions 2000 \
		--transducer-downsample $step \
		--main-context 32 --right-context 16 \
		--user-dir rain \
		--max-epoch 80 \
		--delay-func diag_positive \
		--pretrained-encoder-path $pretrain_model \
		--delay-scale $latency_scale \
		--transducer-label-smoothing 0.1 --transducer-ce-scale 1. \
		--task transducer --task-type st --bpe-dropout 0.1 \
		--arch audio_cat \
		--tokens-per-step 6000 \
		--dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
		--share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' \
		--lr 0.0005 --lr-scheduler inverse_sqrt  \
		--warmup-updates 4000 --warmup-init-lr '1e-07' \
		--criterion fake_loss \
		--clip-norm 2 \
		--save-dir $work_dir/$name \
		--max-tokens 8000 \
		--jointer-layers 6 --decoder-ffn-embed-dim 1024 \
		--update-freq 16 --max-sentences 64 --skip-invalid-size-inputs-valid-test \
		--log-interval 10 --save-interval 4 --log-format simple  --num-workers 2 \
			--fp16 --min-loss-scale 1e-6
```
### Evaluation
To evaluate with SimulEval, first we extract test dataset as SimulEval like:

    ```bash
    python -m scripts.audio_test --prefix test-COMMON  $MUSTC_DIR/en-de/data $DATA_DIR/test
    ```

Evaluate with SimulEval, note parameter --step-read-block should be "step*downsample_size/main_context", and downsample_size is set to 4 (two convolutions with strides 2) in our experimets:

    ```bash
    step_read_block=4
    simuleval --agent ./rain/simul/speech_fullytransducer_agent.py --timeout 100  \
		--task-type st --train-dir $DATA_DIR/mustc_distill \
		--source-lang en --target-lang de \
		--source $DATA_DIR/test/tst-COMMON.list --target $DATA_DIR/test/tst-COMMON.de \
		--data-type speech --task s2s --model-path $modelfile \
		--output $outdir --port $port --timeout 100 \
		--intra-beam 5  --inter-beam 1 --decoder-step-read 256 --eager \
			--step-read-block $step_read_block
    ```

## Citation
If the paper or the code helps you, please cite the paper in the following format :
```
@inproceedings{liu2021cross,
  title={Cross Attention Augmented Transducer Networks for Simultaneous Translation},
  author={Liu, Dan and Du, Mengge and Li, Xiaoxi and Li, Ya and Chen, Enhong},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={39--55},
  year={2021}
}
```