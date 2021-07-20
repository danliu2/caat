
SRC_LANG=en
TGT_LANG=de
INPUT_DIR=$1
$DATA_DIR=$2
raw_data_dir=$INPUT_DIR/en-de/data
stage=1


stage1_dir=$DATA_DIR/stage1
if [ $stage -le 1 ]; then
    echo STAGE$stage processing raw wave
    mkdir -p $stage1_dir
    python -m scripts.audio_process --src-lang $SRC_LANG --tgt-lang $TGT_LANG  $raw_data_dir $stage1_dir
fi

stage2_dir=$DATA_DIR/stage2
if [ $stage -le 2 ]; then
    echo STAGE2 train bpe
    
    mkdir -p $stage2_dir
    cat $stage1_dir/train.${SRC_LANG}-${TGT_LANG}.${SRC_LANG} $stage1_dir/train.${SRC_LANG}-${TGT_LANG}.${TGT_LANG} > $stage2_dir/joint.txt
    python -m scripts.train_spm --vocab-size 30000 --model $stage2_dir/bpe $stage2_dir/joint.txt
    
    for split in train valid test test1; do
        srcfile=$split.${SRC_LANG}-${TGT_LANG}.${SRC_LANG}
        tgtfile=$split.${SRC_LANG}-${TGT_LANG}.${TGT_LANG}
        python -m scripts.gen_bpedata --model $stage2_dir/bpe --dropout 0.0 $stage1_dir/$srcfile $stage2_dir/${srcfile}
        python -m scripts.gen_bpedata --model $stage2_dir/bpe --dropout 0.0 $stage1_dir/$tgtfile $stage2_dir/${tgtfile}
    done
fi
binary_dir=$DATA_DIR/data-bin
if [ $stage -le 3 ]; then
    mkdir $binary_dir
    fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG --destdir $binary_dir --joined-dictionary \
    --workers 4 --trainpref $stage2_dir/train.${SRC_LANG}-${TGT_LANG} --validpref $stage2_dir/valid.${SRC_LANG}-${TGT_LANG} \
    --testpref $stage2_dir/test.${SRC_LANG}-${TGT_LANG},$stage2_dir/test1.${SRC_LANG}-${TGT_LANG}
    fi
dropout_dir=$DATA_DIR/bpe_dropout
if [ $stage -le 4 ]; then
    mkdir -p $dropout_dir
    for split in train valid test test1; do
        srcfile=$split.${SRC_LANG}-${TGT_LANG}.${SRC_LANG}
        tgtfile=$split.${SRC_LANG}-${TGT_LANG}.${TGT_LANG}
        python -m scripts.gen_bpedata --model $stage2_dir/bpe --dropout 0.1 $stage1_dir/$srcfile $dropout_dir/${srcfile}
        python -m scripts.gen_bpedata --model $stage2_dir/bpe --dropout 0.1 $stage1_dir/$tgtfile $dropout_dir/${tgtfile}
    done
    mkdir -p data/dropout-bin
    fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG --destdir data/dropout-bin --joined-dictionary \
    --workers 4 --trainpref $dropout_dir/train.${SRC_LANG}-${TGT_LANG} --validpref $dropout_dir/valid.${SRC_LANG}-${TGT_LANG} \
    --testpref $dropout_dir/test.${SRC_LANG}-${TGT_LANG},$dropout_dir/test1.${SRC_LANG}-${TGT_LANG}
fi

output_dir=$DATA_DIR/mustc_en-de
vocab_dir=$output_dir/text_cfg
if [ $stage -le 5 ]; then
    mkdir -p $output_dir
    python -m scripts.package_dict --langs en,de --vocabs $DATA_DIR/data-bin/dict.en.txt,$DATA_DIR/data-bin/dict.en.txt \
        --encoders $stage2_dir/bpe,$stage2_dir/bpe $vocab_dir
    python -m scripts.package_audio_cfg --norm $stage1_dir/mvn.npz $output_dir/audio_cfg

    for split in train valid test test1; do
        srcfile=$split.${SRC_LANG}-${TGT_LANG}.${SRC_LANG}
        tgtfile=$split.${SRC_LANG}-${TGT_LANG}.${TGT_LANG}
        cp $stage1_dir/fbank_${split}.zip $output_dir/
        cp $stage1_dir/$split.${SRC_LANG}-${TGT_LANG}.audio.tsv $output_dir/
        cp $stage1_dir/$srcfile $output_dir/${srcfile}.raw
        cp $stage1_dir/$tgtfile $output_dir/${tgtfile}.raw
        cp $binary_dir/${srcfile}.* $output_dir/
        cp $binary_dir/${tgtfile}.* $output_dir/
    done
fi
