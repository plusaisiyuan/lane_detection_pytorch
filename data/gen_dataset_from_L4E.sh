#!/bin/bash


sample_dir='/home/julian.jiang/data/FV/lane_batch/L4E'
display_var='4lane/display'
seglabel_var='4lane/seglabel'
sample_var='sample'
var=''

rm -rf $sample_dir/list/*

for disp_path in `find $sample_dir/*/4lane/display/ -name *.png`
do
    seglabel_path=${disp_path/$display_var/$seglabel_var}
    cls_txt=${seglabel_path/png/txt}
    sample_path=${disp_path/$display_var/$sample_var}
    seglabel_path=${seglabel_path/$sample_dir/$var}
    sample_path=${sample_path/$sample_dir/$var}
    echo $sample_path $seglabel_path `cat $cls_txt` >> $sample_dir/list/sample_gt.txt
    if (($RANDOM%8 > 0))
    then
        echo $sample_path $seglabel_path `cat $cls_txt` >> $sample_dir/list/train_gt.txt
    else
        echo $sample_path $seglabel_path `cat $cls_txt` >> $sample_dir/list/val_gt.txt
    fi
done