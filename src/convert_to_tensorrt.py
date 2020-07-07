from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os
import uff
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from PIL import Image
from random import shuffle
from options.options import parser
from options.config import cfg



class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, calibration_files, batch_size, h, w, means,
                 stds, calibration_data_size):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.input_size = [h, w]
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        assert (isinstance(calibration_files, list))
        self.calib_image_paths = calibration_files
        self.cache_file = cache_file
        self.shape = [self.batch_size, 3] + self.input_size
        self.device_input = cuda.mem_alloc(
            trt.volume(self.shape) * trt.float32.itemsize)
        if calibration_data_size is not None:
            self.calib_image_paths = self.calib_image_paths[:min(calibration_data_size, len(self.calib_image_paths))]
        self.max_batches = (len(self.calib_image_paths) // self.batch_size) + \
                           (1 if (len(self.calib_image_paths) % self.batch_size)
                            else 0)
        self.indices = np.arange(len(self.calib_image_paths))
        np.random.shuffle(self.indices)
        print("Calibration image number: {}\nBatch size: {}\nNumber of batches: {}"
              .format(len(self.calib_image_paths), self.batch_size, self.max_batches))

        def load_batches():
            for i in range(self.max_batches):
                print("======== Calibrating on Batch %d ========" %
                      (i))
                indexs = self.indices[i*self.batch_size:(i+1) * self.batch_size]
                paths = [self.calib_image_paths[i] for i in indexs]
                files = self.read_batch_file(paths)
                yield files

        self.batches = load_batches()
        print('init done')

    def read_batch_file(self, filenames):
        tensors = []
        for filename in filenames:
            assert os.path.exists(filename)
            bgr_img = cv2.imread(filename)
            # bgr_img = cv2.resize(bgr_img, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
            # bgr_img = bgr_img[cfg.VERTICAL_CROP_SIZE:, :, :]  # FIX IT
            bgr_img = cv2.resize(bgr_img, (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)

            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_tensor = (rgb_img - self.means) * self.stds
            rgb_tensor = np.transpose(rgb_tensor, (2, 0, 1))
            rgb_tensor = np.ascontiguousarray(rgb_tensor, dtype=np.float32)
            tensors.append(rgb_tensor)
        return np.ascontiguousarray(tensors, dtype=np.float32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, bindings, names):
        try:
            data = next(self.batches)
            cuda.memcpy_htod(self.device_input, data)
            bindings[0] = int(self.device_input)
            return bindings
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def create_tensorrt(args):
    # calib dataset need only about 500 to 1000 images
    calib_dataset = []
    for root, dirs, files in os.walk(args.calib_dir):
        for name in files:
            calib_dataset.append(os.path.join(root, name))


#        print(os.path.join(root,name))
    shuffle(calib_dataset)

    calib = EntropyCalibrator(cache_file=args.cache_file,
                              calibration_files=calib_dataset,
                              batch_size=args.calib_batch,
                              h=cfg.MODEL_INPUT_HEIGHT,
                              w=cfg.MODEL_INPUT_WIDTH,
                              means=cfg.INPUT_MEAN,
                              stds=cfg.INPUT_STD,
                              calibration_data_size=args.calibration_data_size)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network()
    builder.max_batch_size = args.tensorrt_max_batch
    builder.max_workspace_size = 1 << 31
    builder.int8_mode = args.use_int8
    builder.fp16_mode = False
    builder.int8_calibrator = calib
    # pb_filename = args.onnx_file
    # uff_filename = "frozen_onnx/unified_lane_detector_erf_ins_off_480x224_20200624_l4e.uff"
    # outputs = "output/prob"
    # outputs = outputs.split(',')
    # uff_model = uff.from_tensorflow_frozen_model(pb_filename,
    #                                              outputs,
    #                                              preprocessor=None,
    #                                              output_filename=uff_filename,
    #                                              return_graph_info=True,
    #                                              debug_mode=False,
    #                                              quiet=True,
    #                                              list_nodes=False)
    #
    # uff_parser = trt.UffParser()
    # succ1 = uff_parser.register_input(
    #     name="input/image",
    #     shape=[3, cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH])
    # print("register input: ", succ1)
    # for output in outputs:
    #     uff_parser.register_output(output)
    # print("parsing uff model into engine model")
    # uff_parser.parse(uff_filename, network)
    # with builder.build_cuda_engine(network) as engine:
    #     with open(args.engine_file, "wb") as f:
    #         print("Writing engine model to %s" % args.engine_file)
    #         f.write(engine.serialize())

    onnxparser = trt.OnnxParser(network, TRT_LOGGER)

    model = open(args.onnx_file, 'rb')
    print(onnxparser.parse(model.read()))
    print("Successfully ONNX weights from ", args.onnx_file)
    builder.max_batch_size = args.tensorrt_max_batch
    builder.max_workspace_size = 1 << 31
    builder.int8_mode = args.use_int8
    builder.fp16_mode = False
    builder.int8_calibrator = calib
    try:
        engine = builder.build_cuda_engine(network)
    except Exception as e:
        print("Failed creating engine for TensorRT. Error: ", e)
        quit()
    print("Done generating tensorRT engine.")

    print("Serializing tensorRT engine for C++ interface")
    try:
        serialized_engine = engine.serialize()
    except Exception as e:
        print("Couln't serialize engine. Not critical, so I continue. Error: ",
              e)

    with open(args.engine_file, "wb") as f:
        f.write(serialized_engine)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT = 480,224
    cfg.INPUT_MEAN = [0,0,0]
    cfg.INPUT_STD = [1., 1., 1.]
    print("Convert tensorRT... ")
    create_tensorrt(args)
    print("All Done")
