from .options import parser
from easydict import EasyDict as edict


args = parser.parse_args()
cfg = edict()
# ========================= Data Configs ==========================
cfg.dataset = args.dataset
cfg.dataset_path = args.dataset_path
cfg.train_list = args.train_list
cfg.val_list = args.val_list

# ========================= Model Configs ==========================
cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT = 960, 510
cfg.VERTICAL_CROP_SIZE = 80
cfg.IN_IMAGE_H_AFTER_CROP = cfg.LOAD_IMAGE_HEIGHT - cfg.VERTICAL_CROP_SIZE
cfg.RESIZE_WIDTH, cfg.RESIZE_HEIGHT = 560, 280
cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT = 512, 256
cfg.EVAL_WIDTH, cfg.EVAL_HEIGHT = 480, 224
cfg.INPUT_MEAN = [103.939, 116.779, 123.68]
cfg.INPUT_STD = [0.229, 0.224, 0.225]
cfg.DROPOUT = 0.1
cfg.NUM_CLASSES = 4
cfg.NUM_EGO = 2
cfg.CLASS_WEIGHT = [1.43065921, 23.38585625, 26.48577624, 29.52427956] #solid line, dashed line, curb
cfg.method = args.method
cfg.arch = args.arch

# ========================= Learning Configs ==========================
cfg.epochs = 100  # number of total epochs to run
cfg.train_batch_size = 32  # mini-batch size
cfg.val_batch_size = 16  # mini-batch size
cfg.lr = 0.01  # initial learning rate
cfg.lr_steps = [20, 40, 60, 80]  # epochs to decay learning rate by 10
cfg.momentum = 0.9  # momentum
cfg.weight_decay = 1e-4  # weight decay
cfg.use_L1 = False
cfg.optimizer = 'sgd' # or adam
cfg.factor = 0.4
cfg.num_scale = 0.5

# ========================= Monitor Configs ==========================
cfg.print_freq = 1  # print frequency
cfg.eval_freq = 1  # evaluation frequency
cfg.POINTS_COUNT = 20
cfg.EG0_COLORS = ["red", "green", "blue", "yellow"]
cfg.EG0_POINT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),  (255, 255, 255),  (255, 255, 0)]

# ========================= Runtime Configs ==========================
cfg.gpus = args.gpus
cfg.workers = args.workers  # number of data loading workers
cfg.resume = args.resume  # path to latest checkpoint
cfg.finetune = True
cfg.evaluate = False  # evaluate model on validation set
cfg.start_epoch = args.start_epoch  # manual epoch number (useful on restarts)
cfg.snapshot_pref = args.snapshot_pref
cfg.save_path = args.save_path
cfg.THRESHOLD_CLS = 0.8
cfg.THRESHOLD_EGO = 0.8

cfg.cls_mapping = [
    0,
    1,
    2,
    3,
    0,
    1,
    2,
    2
]

cfg.ego_mapping = [
    0,
    0,
    1,
    2,
    0,
]

cfg.exist_mapping = [
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1
]
