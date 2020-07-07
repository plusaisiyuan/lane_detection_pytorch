import cv2
import os
import time
import numpy as np
import torch
import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import json
from options.options import parser
from options.config import cfg
from utils import log_helper

from evaluate_lane import process, get_lane_points, evaluate_rmse, calculate, aggregate_results

def eval_with_model(model_path, image_list, in_samples, out_infer, out_eval, max_pixel_dis, logger):
    output_path = os.path.join(out_infer, 'output')
    mask_path = os.path.join(out_infer, 'mask')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)
    if model_path.find('.onnx') != -1:
        return eval_with_onnx(model_path, image_list, in_samples, mask_path, output_path, out_eval, max_pixel_dis, logger)
    elif model_path.find('.engine') != -1:
        return eval_with_engine(model_path, image_list, in_samples, mask_path, output_path, out_eval, max_pixel_dis, logger)
    else:
        import sys
        logger.error('infer image failed!')
        sys.exit(-1)

def preprocess(image):
    image_bgr = cv2.resize(image, (cfg.LOAD_IMAGE_WIDTH, cfg.LOAD_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    image_bgr = image_bgr[cfg.VERTICAL_CROP_SIZE:, :, :]
    image_bgr = cv2.resize(image_bgr, (cfg.MODEL_INPUT_WIDTH, cfg.MODEL_INPUT_HEIGHT),
                           interpolation=cv2.INTER_NEAREST)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = (image_rgb - cfg.INPUT_MEAN) * cfg.INPUT_STD
    image_rgb = np.transpose(image_rgb, (2, 0, 1))
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.float32)
    image_rgb = np.expand_dims(image_rgb, 0)
    return image_rgb

def postprocess(lane_cls, h, w):
    threshold_cls = int(cfg.THRESHOLD_CLS * 255)
    result_cls = np.zeros((h, w)).astype(np.uint8)

    for num in range(cfg.NUM_CLASSES - 1):
        prob_map = (lane_cls[num + 1] * 255).astype(np.uint8)
        map_bak = np.zeros((cfg.LOAD_IMAGE_HEIGHT, cfg.LOAD_IMAGE_WIDTH))
        map_bak[cfg.VERTICAL_CROP_SIZE:, :] = cv2.resize(prob_map, (cfg.LOAD_IMAGE_WIDTH, cfg.IN_IMAGE_H_AFTER_CROP),
                                                         interpolation=cv2.INTER_NEAREST)
        map_bak = cv2.resize(map_bak, (w, h), interpolation=cv2.INTER_NEAREST)
        result_cls[map_bak >= threshold_cls] = num + 1
    return result_cls

def eval_with_onnx(model_path, image_list, in_samples, mask_infer, out_infer, out_eval, max_pixel_dis, logger):
    logger.info("accessing onnx")
    ort_session = onnxruntime.InferenceSession(model_path)

    # compute ONNX Runtime output prediction
    total_time_elapsed = 0.0
    eval_rets = []
    count = 0
    for img_file in image_list:
        logger.info("predicting for {}".format(img_file))
        begin_ts = time.time()
        image = cv2.imread(img_file)
        h, w, c = image.shape
        image_rgb = preprocess(image)
        end_ts = time.time()
        time_elapsed = end_ts - begin_ts
        logger.info("proprocess cost time: {} s".format(time_elapsed))
        begin_ts = time.time()
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        lane_inputs = {ort_session.get_inputs()[0].name: image_rgb}
        lane_outs = ort_session.run(None, lane_inputs)
        end_ts = time.time()
        time_elapsed = end_ts - begin_ts
        logger.info("infer cost time: {} s".format(time_elapsed))
        total_time_elapsed += time_elapsed
        # output_image to verify
        begin_ts = time.time()
        lane_cls = torch.from_numpy(lane_outs[0])
        lane_cls = lane_cls.data.cpu().numpy()
        lane_cls = np.squeeze(lane_cls, 0)
        result_cls = postprocess(lane_cls, h, w)
        end_ts = time.time()
        time_elapsed = end_ts - begin_ts
        logger.info("post-process cost time: {} s".format(time_elapsed))
        begin_ts = time.time()
        eval_ret_image = eval_per_image(image, img_file, result_cls, in_samples, mask_infer, out_infer, out_eval,
                                        max_pixel_dis, logger)
        end_ts = time.time()
        time_elapsed = end_ts - begin_ts
        logger.info("eval cost time: {} s".format(time_elapsed))
        if len(eval_ret_image):
            eval_rets.extend(eval_ret_image)
            count += 1
            logger.info("completed {}%".format(round(count * 100 / len(image_list), 2)))
    return total_time_elapsed, eval_rets

def eval_with_engine(model_path, image_list, in_samples, mask_infer, out_infer, out_eval, max_pixel_dis, logger):
    logger.info("accessing engine")
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    total_time_elapsed = 0.0
    eval_rets = []
    count = 0
    lane_outs = []
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
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
        for img_file in image_list:
            logger.info("predicting for {}".format(img_file))
            begin_ts = time.time()
            image = cv2.imread(img_file)
            h, w, c = image.shape
            image_rgb = preprocess(image)
            image_rgb = image_rgb.ravel()
            end_ts = time.time()
            time_elapsed = end_ts - begin_ts
            logger.info("proprocess cost time: {} s".format(time_elapsed))
            stream = cuda.Stream()
            with engine.create_execution_context() as context:
                cuda.memcpy_htod_async(d_input, image_rgb, stream)
                context.execute_async(bindings=bindings, stream_handle=stream.handle)
                if cfg.NUM_CLASSES:
                    cuda.memcpy_dtoh_async(h_output_cls, d_output_cls, stream)
                if cfg.NUM_EGO:
                    cuda.memcpy_dtoh_async(h_output_ego, d_output_ego, stream)
                    cuda.memcpy_dtoh_async(h_output_exist, d_output_exist, stream)
                stream.synchronize()
                if cfg.NUM_CLASSES:
                    h_output_cls = h_output_cls.reshape(output_cls_size)
                    lane_outs.append(h_output_cls)
                if cfg.NUM_EGO:
                    h_output_ego = h_output_ego.reshape(output_ego_size)
                    h_output_exist = h_output_exist.reshape(output_exist_size)
                    lane_outs.append(h_output_ego)
                    lane_outs.append(h_output_exist)
            logger.info("infer cost time: {} s".format(time_elapsed))
            total_time_elapsed += time_elapsed
            begin_ts = time.time()
            lane_cls = lane_outs[0]
            result_cls = postprocess(lane_cls, h, w)
            end_ts = time.time()
            time_elapsed = end_ts - begin_ts
            logger.info("post-process cost time: {} s".format(time_elapsed))
            begin_ts = time.time()
            eval_ret_image = eval_per_image(image, img_file, result_cls, in_samples, mask_infer, out_infer, out_eval,
                                            max_pixel_dis, logger)
            end_ts = time.time()
            time_elapsed = end_ts - begin_ts
            logger.info("eval cost time: {} s".format(time_elapsed))
            if len(eval_ret_image):
                eval_rets.extend(eval_ret_image)
                count += 1
                logger.info("completed {}%".format(round(count * 100 / len(image_list), 2)))
        return total_time_elapsed, eval_rets


def eval_per_image(sample, img_file, pred_image, in_samples, mask_infer, out_infer, out_eval, max_pixel_dis, logger):
    bag_name = img_file.split('/')[-2]
    image_name = img_file.split('/')[-1]
    infer_bag_dir = os.path.join(out_infer, bag_name)
    result_file = os.path.join(infer_bag_dir, image_name)

    eval_ret_image = []
    gt_image_name = os.path.join(in_samples, 'groundtruth', bag_name, image_name)
    if not os.path.exists(gt_image_name):
        logger.info("failed to evaluate {}".format(gt_image_name))
        return eval_ret_image
    gt_image = cv2.resize(cv2.imread(gt_image_name), (cfg.EVAL_WIDTH, cfg.EVAL_HEIGHT), interpolation=cv2.INTER_NEAREST)
    pred_image = cv2.resize(pred_image, (cfg.EVAL_WIDTH, cfg.EVAL_HEIGHT), interpolation=cv2.INTER_NEAREST)
    if len(gt_image.shape) == 3:
        gt_image = gt_image[:, :, 0]
    assert len(gt_image.shape) == 2
    cls_map = lambda t: cfg.cls_mapping[t]
    remap = np.vectorize(cls_map)
    gt_image = remap(gt_image)
    for i in range(3):
        lab0 = gt_image[gt_image.shape[0] // 3 * i:gt_image.shape[0] // 3 * (i + 1), :]
        lab1 = pred_image[pred_image.shape[0] // 3 * i:pred_image.shape[0] // 3 * (i + 1), :]
        lane_objs = []
        lane_objs.append(process(lab0))
        lane_objs.append(process(lab1))

        eval_bag_dir = os.path.join(out_eval, str(i), bag_name)
        if not os.path.isdir(eval_bag_dir):
            os.makedirs(eval_bag_dir)
        eval_ret = evaluate_rmse(*lane_objs, max_pixel_dis=max_pixel_dis)
        dump_obj = dict(
            data_file='data/' + bag_name + '/' + image_name,
            label_file='label/' + bag_name + '/' + image_name + '.json',
            ground_truth='out/' + bag_name + '/' + image_name,
            result_file=result_file,
            metrics=calculate(eval_ret)
        )
        json.dump(dump_obj, open(os.path.join(eval_bag_dir, image_name.replace('.png', '.png.json')), 'w'),
                  indent=2)
        eval_ret_image.append(eval_ret)

    lane_objs = []
    lane_objs.append(process(gt_image))
    lane_objs.append(process(pred_image))
    eval_bag_dir = os.path.join(out_eval, 'overall', bag_name)
    if not os.path.isdir(eval_bag_dir):
        os.makedirs(eval_bag_dir)
    eval_ret = evaluate_rmse(*lane_objs, max_pixel_dis=max_pixel_dis)
    dump_obj = dict(
        data_file='data/' + bag_name + '/' + image_name,
        label_file='label/' + bag_name + '/' + image_name + '.json',
        ground_truth='out/' + bag_name + '/' + image_name,
        result_file=result_file,
        metrics=calculate(eval_ret)
    )
    json.dump(dump_obj, open(os.path.join(eval_bag_dir, image_name.replace('.png', '.png.json')), 'w'),
              indent=2)
    eval_ret_image.append(eval_ret)

    begin_ts = time.time()
    for obj in dump_obj['metrics'][:]:
        if obj['category'] == 'all' and obj['name'] == 'f1_score' and obj['value'] < 0.8:
            mask_bag_dir = os.path.join(mask_infer, bag_name)
            if not os.path.isdir(mask_bag_dir):
                os.makedirs(mask_bag_dir)
            imgs = []
            imgs.append(gt_image)
            imgs.append(pred_image)
            debug_img = draw_img(imgs)
            cv2.putText(debug_img, 'f1_score: ' + str(obj['value']), (2*cfg.MODEL_INPUT_WIDTH+10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv_img = cv2.resize(sample, (cfg.EVAL_WIDTH, cfg.EVAL_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(
                np.float32)
            merge_img = np.concatenate([cv_img, debug_img], axis=1)
            mask_file = os.path.join(mask_bag_dir, image_name)
            cv2.imwrite(mask_file, merge_img)
            break
    end_ts = time.time()
    time_elapsed = end_ts - begin_ts
    logger.info("dump cost time: {} s".format(time_elapsed))
    return eval_ret_image

def draw_img(raw_imgs):
    h, w = raw_imgs[0].shape
    raw_img = np.zeros((h, 3*w, 3))
    raw_img[:, :w, 0] += np.where(raw_imgs[0] == 1, 255, 0)
    raw_img[:, :w, 1] += np.where(raw_imgs[0] == 3, 255, 0)
    raw_img[:, :w, 2] += np.where(raw_imgs[0] == 2, 255, 0)
    raw_img[:, w:2*w, 0] += np.where(raw_imgs[1] == 1, 255, 0)
    raw_img[:, w:2*w, 1] += np.where(raw_imgs[1] == 3, 255, 0)
    raw_img[:, w:2*w, 2] += np.where(raw_imgs[1] == 2, 255, 0)
    raw_img[:, 2*w:3*w, 0] += np.where(raw_imgs[0] > 0, raw_imgs[0]*80, 0)
    raw_img[:, 2*w:3*w, 2] += np.where(raw_imgs[1] > 0, raw_imgs[1]*80, 0)
    cv2.putText(raw_img, 'gt', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(raw_img, 'pred', (w+10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return raw_img

def main():
    args = parser.parse_args()

    image_list = []
    with open(os.path.join(args.in_samples, args.in_list)) as f:
        for line in f:
            image_list.append(os.path.join(args.in_samples, line.split('.png')[-2]+ '.png'))

    logger = log_helper.get_logger()
    begin_ts = time.time()
    total_time_elapsed, eval_rets = eval_with_model(args.model_file, image_list, args.in_samples, args.out_infer, args.out_eval, args.max_pixel_dis, logger)
    end_ts = time.time()
    logger.info("total pipeline time elapsed: {} s".format(end_ts-begin_ts))
    logger.info("total infer time elapsed: {} s".format(total_time_elapsed))
    ave_time_elapsed = total_time_elapsed / len(image_list)
    logger.info("average infer time elapsed: {} s".format(ave_time_elapsed))

    for i in range(3):
        eval_rets_i = [eval_rets[x*4+i] for x in range(len(eval_rets)//4)]
        ag_ret = dict(
            count=len(eval_rets_i),
            metrics=calculate(aggregate_results(eval_rets_i)),
            commpare_list=args.in_list,
            out_folder=args.out_eval
        )
        json.dump(ag_ret, open(os.path.join(args.out_eval, str(i), 'L4E_result.json'), 'w'), indent=2)

    eval_rets_overall = [eval_rets[x * 4 + 3] for x in range(len(eval_rets) // 4)]
    ag_ret = dict(
        count=len(eval_rets_overall),
        metrics=calculate(aggregate_results(eval_rets_overall)),
        commpare_list=args.in_list,
        out_folder=args.out_eval
    )
    json.dump(ag_ret, open(os.path.join(args.out_eval, 'overall', 'L4E_result.json'), 'w'), indent=2)

if __name__ == '__main__':
    main()
