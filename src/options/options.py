import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Semantic Segmentation")

# ========================== Data Configs ============================
parser.add_argument('--dataset', type=str, choices=['VOCAug', 'VOC2012', 'COCO', 'Cityscapes', 'ApolloScape', 'CULane', 'L4E'])
parser.add_argument('--dataset_path', type=str, default="/home/julian/data/lane_batch/L4E")
parser.add_argument('--train_list', type=str, default="train_gt")
parser.add_argument('--val_list', type=str, default="val_gt")

# ========================== Model Configs ===========================
parser.add_argument('--method', type=str, choices=['FCN', 'DeepLab', 'DeepLab3', 'PSPNet', 'ERFNet'], default='ERFNet')
parser.add_argument('--arch', type=str, default="resnet101")

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--weight', default='', type=str, metavar='PATH', help='path to initial weight (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set') # true
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=[0])
parser.add_argument('--save_path', type=str, default=None)

# ========================= Rosbag Configs ==========================
parser.add_argument("--bags", type=str, help="path to bags (comma separated)")
parser.add_argument("--out", type=str, help="output path (a directory)")
parser.add_argument("--rate", type=float, help="desired sample rate in seconds", default=0.1)
parser.add_argument("--tolerance", type=float, help="tolerance", default=0.005)
parser.add_argument("--velo_topics", type=str, help="velodyne topic (comma separated, don't add space between topics)")
parser.add_argument("--cam_topics", type=str, help="camera topics (comma separated, don't add space between topics)")
parser.add_argument("--odom_topics", type=str, help="odometry topic (comma separated, don't add space between topics)")
parser.add_argument("--frame_limit", type=int, help="frame limit if > 0", default=0)

# ========================= Onnx Configs ==========================
parser.add_argument('--onnx_file', type=str, default='erfnet.onnx', help='name of the output onnx file')
parser.add_argument('--onnx_optim_passes', type=str, default='fuse_bn_into_conv', help='optimization passes used by onnx;' 'should be separated by comma')
parser.add_argument('--model_file', type=str, default='', help='name of the output onnx or engine file')
parser.add_argument('--tensorrt_file', type=str, default='erfnet.engine', help='name of the output tensorrt file')
parser.add_argument('--calib_dir', type=str, default='/media/jiangzh/zhihao-2TB/calibration_data_lane', help='name of the cache dir')
parser.add_argument('--calib_batch', type=int, default=1, help='batch of calib')
parser.add_argument('--cache_file', type=str, default='erfnet.calibration_cache', help='name of the cache file')
parser.add_argument('--tensorrt_max_batch', type=int, default=1, help='max batch of tensorrt')
parser.add_argument('--not_use_int8', type=bool, default=False, help='not use int8')