import cv2
import numpy as np
import torch
import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from options.options import parser
from options.config import cfg

def inference_with_onnx(model_file, model_in):
    ort_session = onnxruntime.InferenceSession(model_file)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: model_in}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

def inference_with_engine(model_file, model_in):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    ort_outs = []
    with open(model_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        model_in = model_in.ravel()
        h_input = np.empty(trt.volume(engine.get_binding_shape(0)), np.float32)
        d_input = cuda.mem_alloc(1 * h_input.nbytes)

        if cfg.NUM_CLASSES and cfg.NUM_EGO:
            output_cls_size = engine.get_binding_shape(1)
            output_ego_size = engine.get_binding_shape(2)
            output_exist_size = engine.get_binding_shape(3)
            h_output_cls = np.empty(trt.volume(engine.get_binding_shape(1)), np.float32)
            h_output_ego = np.empty(trt.volume(engine.get_binding_shape(2)), np.float32)
            h_output_exist = np.empty(trt.volume(engine.get_binding_shape(3)), np.float32)
            d_output_cls = cuda.mem_alloc(1 * h_output_cls.nbytes)
            d_output_ego = cuda.mem_alloc(1 * h_output_ego.nbytes)
            d_output_exist = cuda.mem_alloc(1 * h_output_exist.nbytes)
            bindings = [int(d_input), int(d_output_cls), int(d_output_ego), int(d_output_exist)]
        elif cfg.NUM_CLASSES and cfg.NUM_EGO == 0:
            output_cls_size = engine.get_binding_shape(1)
            h_output_cls = np.empty(trt.volume(engine.get_binding_shape(1)), np.float32)
            d_output_cls = cuda.mem_alloc(1 * h_output_cls.nbytes)
            bindings = [int(d_input), int(d_output_cls)]
        elif cfg.NUM_CLASSES == 0 and cfg.NUM_EGO:
            output_ego_size = engine.get_binding_shape(1)
            output_exist_size = engine.get_binding_shape(2)
            h_output_ego = np.empty(trt.volume(engine.get_binding_shape(1)), np.float32)
            h_output_exist = np.empty(trt.volume(engine.get_binding_shape(2)), np.float32)
            d_output_ego = cuda.mem_alloc(1 * h_output_ego.nbytes)
            d_output_exist = cuda.mem_alloc(1 * h_output_exist.nbytes)
            bindings = [int(d_input), int(d_output_ego), int(d_output_exist)]

        stream = cuda.Stream()
        with engine.create_execution_context() as context:
            cuda.memcpy_htod_async(d_input, model_in, stream)
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            if cfg.NUM_CLASSES:
                cuda.memcpy_dtoh_async(h_output_cls, d_output_cls, stream)
            if cfg.NUM_EGO:
                cuda.memcpy_dtoh_async(h_output_ego, d_output_ego, stream)
                cuda.memcpy_dtoh_async(h_output_exist, d_output_exist, stream)
            stream.synchronize()
            if cfg.NUM_CLASSES:
                h_output_cls = h_output_cls.reshape(output_cls_size)
                ort_outs.append(h_output_cls)
            if cfg.NUM_EGO:
                h_output_ego = h_output_ego.reshape(output_ego_size)
                h_output_exist = h_output_exist.reshape(output_exist_size)
                ort_outs.append(h_output_ego)
                ort_outs.append(h_output_exist)
            return ort_outs

def inference(model_file, image_file):
    image = cv2.imread(image_file)
    h, w, c = image.shape
    image_bgr = cv2.resize(image, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_bgr = image_bgr[cfg.VERTICAL_CROP_SIZE:, :, :]
    image_bgr = cv2.resize(image_bgr, (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  #  image_rgb = (image_rgb - cfg.INPUT_MEAN) * cfg.INPUT_STD
    image_rgb = np.transpose(image_rgb, (2, 0, 1))
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.float32)
    image_rgb = np.expand_dims(image_rgb, 0)
    ort_outs = []
    if model_file.find('onnx') != -1:
        ort_outs = inference_with_onnx(model_file, image_rgb)
        ort_out = torch.from_numpy(ort_outs[0])
        ort_out = ort_out.data.cpu().numpy()
        ort_out = np.squeeze(ort_out, 0)
    elif model_file.find('engine') != -1:
        ort_outs = inference_with_engine(model_file, image_rgb)
        ort_out = ort_outs[0]
        ort_out = np.transpose(ort_out, (2, 0, 1))
    if len(ort_outs):
        threshold_cls = int(cfg.THRESHOLD_CLS * 255)
        result_cls = np.zeros((h, w)).astype(np.uint8)
        # ort_out = F.softmax(ort_out, dim=1)

        for num in range(cfg.NUM_CLASSES-1):
            prob_map = (ort_out[num + 1] * 255).astype(np.uint8)
            map_bak = np.zeros((cfg.LOAD_IMAGE_HEIGHT, cfg.LOAD_IMAGE_WIDTH))
            map_bak[cfg.VERTICAL_CROP_SIZE:, :] = cv2.resize(prob_map, (cfg.LOAD_IMAGE_WIDTH, cfg.IN_IMAGE_H_AFTER_CROP),
                                                         interpolation=cv2.INTER_NEAREST)
            map_bak = cv2.resize(map_bak, (w, h), interpolation=cv2.INTER_NEAREST)
            result_cls[map_bak >= threshold_cls] = num + 1

        result_cls_color = np.copy(image)
        # cv2.imshow("alllane", result_cls_color)
        # cv2.waitKey()
        for i in range(h):
            for j in range(w):
                if result_cls[i, j] != 0:
                    result_cls_color[i, j, :] = cfg.EG0_POINT_COLORS[result_cls[i, j] - 1]
        # cv2.imshow("alllane", result_cls_color)
        # cv2.waitKey()
        print("src/samples/alllane.png")
        cv2.imwrite("src/samples/alllane.png", result_cls_color)

if __name__ == '__main__':
    args = parser.parse_args()
    cfg.THRESHOLD_CLS = 0.85
    cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT = 480,224
    image_file = 'src/samples/lane_detection_l4e_input.png'
    inference(args.model_file, image_file)
    print("All Done")
