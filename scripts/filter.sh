in_dir=/yrfs1/hyperbrain/danliu/snmt/data/must_c.en_de
out_dir=./data/must_filtered2
mkdir -p $out_dir
python -m scripts.filter_data $in_dir $out_dir

cp -r $in_dir/audio_cfg $out_dir
cp -r $in_dir/text_cfg $out_dir
cp -r $in_dir/test* $out_dir
cp -r $in_dir/valid* $out_dir
ln -s $in_dir/fbank_test.zip $out_dir/fbank_test.zip
ln -s $in_dir/fbank_test1.zip $out_dir/fbank_test1.zip
ln -s $in_dir/fbank_valid.zip $out_dir/fbank_valid.zip
ln -s $in_dir/fbank_train.zip $out_dir/fbank_train.zip