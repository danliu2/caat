raw_dir=data/must_filtered2
new_dir=data/must_distill
mkdir -p $new_dir
cp -r $raw_dir/* $new_dir/
rm $new_dir/train.en-de.de.raw
inferfile=./distill_train.txt
grep ^D $inferfile | cut -f3 >tmp.txt
echo infer file size: `wc -l tmp.txt`
echo raw file size: `wc -l $raw_dir/train.en-de.de.raw`
mv tmp.txt $new_dir/train.en-de.de.raw