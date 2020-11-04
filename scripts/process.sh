#set -x

path="/media/jiangzh/zhihao-2TB/j7-e0008"
for bag_path in `ls $path/bag/*.db`
do
  bag=${bag_path##*/}
  out_dir="$path/result/$bag"
  if [ ! -d ${out_dir} ];then
    mkdir -p ${out_dir}
    python tools/extract_dataset_from_bag.py --bags ${bag_path} --out ${out_dir} --cam_topics /front_left_camera/image_color/compressed --rate 0.4 --frame_limit 10000
  fi
done

