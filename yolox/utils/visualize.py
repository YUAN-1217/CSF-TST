#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

__all__ = ["vis", "vis_similarity"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def vis_similarity(img, similarity):
    hm = (similarity * 255.).astype(np.uint8)
    if img.shape[:2] != similarity.shape:
        hm = cv2.resize(hm, img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    maxima = np.unravel_index(np.argmax(hm, axis=None), hm.shape)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_RAINBOW)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    res = cv2.addWeighted(img, 0.6, hm, 0.4, 0)
    cv2.drawMarker(res, maxima[::-1], (255, 255, 255), cv2.MARKER_TILTED_CROSS, thickness=2)

    return res



def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


# def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):#原版只是删除了fps显示
#     im = np.ascontiguousarray(np.copy(image))
#     im_h, im_w = im.shape[:2]

#     top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

#     #text_scale = max(1, image.shape[1] / 1600.)
#     #text_thickness = 2
#     #line_thickness = max(1, int(image.shape[1] / 500.))
#     text_scale = 2
#     text_thickness = 2
#     line_thickness = 3

#     radius = max(5, int(im_w/140.))
#     #cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
#     cv2.putText(im, 'frame: %.2f num: %d' % (frame_id, len(tlwhs)),            
#                 (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

#     for i, tlwh in enumerate(tlwhs):
#         x1, y1, w, h = tlwh
#         intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
#         obj_id = int(obj_ids[i])
#         id_text = '{}'.format(int(obj_id))
#         if ids2 is not None:
#             id_text = id_text + ', {}'.format(int(ids2[i]))
#         color = get_color(abs(obj_id))
#         cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
#         cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
#                     thickness=text_thickness)
#     return im


# def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):  #直接显示类别文字
#     # 定义类别名称字典 - 添加VisDrone数据集的类别映射
#     class_names = {
#         0: "pedestrian", 
#         1: "people",
#         2: "bicycle",
#         3: "car",
#         4: "van",
#         5: "truck",
#         6: "tricycle",
#         7: "awning-tricycle",
#         8: "bus",
#         9: "motor"
#     }
    
#     im = np.ascontiguousarray(np.copy(image))
#     im_h, im_w = im.shape[:2]

#     top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

#     #text_scale = max(1, image.shape[1] / 1600.)
#     #text_thickness = 2
#     #line_thickness = max(1, int(image.shape[1] / 500.))
#     text_scale = 2
#     text_thickness = 2
#     line_thickness = 3

#     radius = max(5, int(im_w/140.))
#     #cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
#     cv2.putText(im, 'frame: %.2f num: %d' % (frame_id, len(tlwhs)),            
#                 (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

#     for i, tlwh in enumerate(tlwhs):
#         x1, y1, w, h = tlwh
#         intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
#         obj_id = int(obj_ids[i])
        
#         # 获取类别并转换为名称
#         if ids2 is not None:
#             cls_id = int(ids2[i])
#             cls_name = class_names.get(cls_id, str(cls_id))
#             id_text = '{}_{}'.format(cls_name, obj_id)
#         else:
#             id_text = '{}'.format(int(obj_id))
            
#         color = get_color(abs(obj_id))
#         cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
#         cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
#                     thickness=text_thickness)
#     return im
def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    # 定义类别名称字典 - 添加VisDrone数据集的类别映射
    class_names = {
        0: "pedestrian", 
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor"
    }
    
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # 调整文本参数，增大字体以提高可读性
    text_scale = max(0.7, image.shape[1] / 2500.)  # 增大字体
    text_thickness = 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    # 帧信息显示
    cv2.putText(im, 'frame: %d num: %d' % (frame_id, len(tlwhs)), ##已经去掉了左上角fps显示           
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        
        # 获取类别并转换为名称
        if ids2 is not None:
            cls_id = int(ids2[i])
            cls_name = class_names.get(cls_id, str(cls_id))
            id_text = '{}{}'.format(cls_name, obj_id)
        else:
            id_text = '{}'.format(int(obj_id))
        
        # 使用原有的颜色分配方式
        color = get_color(abs(obj_id))
            
        # 画框
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
        # 计算文本大小以便创建背景
        font = cv2.FONT_HERSHEY_PLAIN
        txt_size = cv2.getTextSize(id_text, font, text_scale, text_thickness)[0]
        
        # 为文本创建半透明背景，提高可读性
        alpha = 0.5
        bg_img = im.copy()
        cv2.rectangle(
            bg_img,
            (intbox[0], intbox[1] - int(1.5 * txt_size[1])),  # 将文本放在框的上方，稍微增加空间
            (intbox[0] + txt_size[0] + 4, intbox[1]),
            color,  # 使用与框相同的颜色
            -1      # 填充矩形
        )
        # 应用半透明效果
        cv2.addWeighted(bg_img, alpha, im, 1 - alpha, 0, im)
        
        # 放置文本
        cv2.putText(
            im, id_text, 
            (intbox[0] + 2, intbox[1] - int(0.4 * txt_size[1])),  # 轻微调整位置使文字居中
            font, text_scale, (255, 255, 255),  # 白色文本
            thickness=text_thickness
        )
    
    return im
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)