#!/bin/bash

path="/media/jiangzh/zhihao-2TB/j7-00007/"
for bag_path in `ls $path/*_100to300.bag`
do
  bag=${bag_path##*/}
  out_dir="$path/result/$bag"
  if [ ! -d ${out_dir} ];then
    mkdir -p ${out_dir}
  fi
  python src/run_rosbag_mono.py   --bag ${bag_path} --out ${out_dir} \
                             --cam_topics /front_left_camera/image_color/compressed,/front_right_camera/image_color/compressed \
                             --odom_topics /navsat/odom \
                             --rate 0.1 \
                             --tolerance 0.05 \
                             --frame_limit 10000 \
                             --model_file /home/jiangzh/lane/lane_models/lane_detector_erf_ego_on_480x224_20200910_l4e.8bit.engine \
                             --sim_path /home/jiangzh/lane/tmp/output
done

#            --cam_topics /front_left_camera/image_color/compressed,/front_right_camera/image_color/compressed \
#            --cam_topics /usb_cam_left/image_raw/compressed,/usb_cam_right/image_raw/compressed
