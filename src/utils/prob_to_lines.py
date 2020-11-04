from PIL import Image, ImageFilter
import numpy as np
import sys
sys.path.append("..")
from options.config  import cfg

# Alternative to matlab script that converts probability maps to lines

def GetLane(score, thr = 0.3):

    coordinate = np.zeros(cfg.POINTS_COUNT)
    for i in range (cfg.POINTS_COUNT):
        lineId = int(cfg.MODEL_INPUT_HEIGHT - i * 10.0 / cfg.IN_IMAGE_H_AFTER_CROP * cfg.MODEL_INPUT_HEIGHT - 1)
        line = score[lineId, :]
        max_id = np.argmax(line)
        max_values = line[max_id]
        # ys, xs = np.where(line == max_values)
        if max_values / 255.0 > thr:
            coordinate[i] = max_id

    coordSum = np.sum(coordinate > 0)
    if coordSum < 2:
        coordinate = np.zeros(cfg.POINTS_COUNT)

    return coordinate, coordSum


def GetLines(existArray, scoreMaps, thr = 0.3):
    coordinates = []
    YS = cfg.LOAD_IMAGE_HEIGHT - np.arange(cfg.POINTS_COUNT) * 10 - 1

    for l in range(len(scoreMaps)):
        if existArray[l]:
            coordinate, coordSum = GetLane(scoreMaps[l], thr)
            if coordSum > 1:
                xs = coordinate * (float(cfg.LOAD_IMAGE_WIDTH) / cfg.MODEL_INPUT_WIDTH)
                xs = np.round(xs).astype(np.int)
                pos = xs > 0
                curY = YS[pos]
                curX = xs[pos]
                curX += 1
                coordinates.append(list(zip(curX, curY)))
            else:
                coordinates.append([])
        else:
            coordinates.append([])

    return coordinates

def AddMask(img, mask, color, threshold = 0.3):
    back = Image.new('RGB', (img.size[0], img.size[1]), color=color)

    alpha = np.array(mask).astype(float) / 255
    alpha[alpha > threshold] = 1.0
    alpha[alpha <= threshold] = 0.0
    alpha *= 255
    alpha = alpha.astype(np.uint8)
    mask = Image.fromarray(np.array(alpha), 'L')
    mask_blur = mask.filter(ImageFilter.GaussianBlur(3))

    res = Image.composite(back, img, mask_blur)
    return res

def GetAllLines(existArray, prob_result):

    def get_centroid(ary):
        ret = []
        seg = []
        for i in ary:
            if len(seg) == 0 or seg[-1] + 1 == i:
                seg.append(i)
            else:
                ret.append(seg[len(seg)/2])
                seg = [i]
        if len(seg) != 0:
            ret.append(seg[len(seg)/2])
        return ret

    h, w = prob_result.shape
    coordinates = []
    YS = np.arange(h)
    XS = np.arange(h)
    for lid in range(len(existArray)):
        if existArray[lid]:
            ys, xs = np.where(prob_result == lid+1)
            ytox = {}
            for x, y in zip(xs, ys):
                ytox.setdefault(y, []).append(x)
            lane = {}
            for y in range(h):
                xs = ytox.get(y, [])
                # only use the center of consecutive pixels
                xs = get_centroid(xs)
                if len(xs) > 0:
                    XS[y] = xs[0]
                else:
                    XS[y] = -1
            coordinates.append(list(zip(XS, YS)))
        else:
            # for y in range(h):
            #     XS[y] = -1
            coordinates.append([])
    # print coordinates
    return coordinates