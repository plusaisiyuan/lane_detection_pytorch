#set -x

path="/media/jiangzh/zhihao-2TB/bag_road_test_J7-5_empty_load_0701"
for bag_path in `ls $path/bag/*.bag`
do
  bag=${bag_path##*/}
  out_dir="$path/result/$bag"
  if [ ! -d ${out_dir} ];then
    mkdir -p ${out_dir}
    python tools/extract_dataset_from_bag.py --bags ${bag_path} --out ${out_dir} --cam_topics /front_left_camera/image_color/compressed --rate 0.4 --frame_limit 10000
  fi
done

