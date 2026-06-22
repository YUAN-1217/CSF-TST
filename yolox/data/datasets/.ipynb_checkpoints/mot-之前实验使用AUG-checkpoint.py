#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import annotations
import cv2
import numpy as np
from pycocotools.coco import COCO

import os
import math
import random
import warnings

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class MOTDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self,
                 data_dir=None,
                 json_file="train_half.json",
                 name="train",
                 img_size=(608, 1088),
                 preproc=None,
                 max_epoch=80,
                 is_training=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))   #使用COCO API加载数据集的标注文件（train_half.json）。
        self.ids = self.coco.getImgIds()  # image ids, not track ids       #获取所有图像的ID列表
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()  # in order     #调用_load_coco_annotations方法，基于图像ID加载所有图像的标注数据
        self.uncertainties = [
            np.zeros_like(ann[0][:, -1]) for ann in self.annotations
        ]
        # self.affine_cache = [None] * len(self.annotations)
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

        self.is_training = is_training
        self.seq_names = self.get_seq_names()

        self.max_epoch = max_epoch
        self.time_scale = 100
        self.cur_epoch = 0

        self.sync_dir = './ann_tmp'
        os.mkdir(self.sync_dir) if not os.path.exists(self.sync_dir) else None

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]  # format: tlbr
            res[ix, 4] = cls  # class id, 0 for person
            res[ix, 5] = obj["track_id"]  # track id

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        (min_id, max_id) = (res[:, -1].min(), res[:, -1].max()) if num_objs else (0, 0)
        seq_name = self.img_path2seq(file_name)
        img_info = (height, width, frame_id, video_id, file_name, seq_name, min_id, max_id)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def get_ann_tmp_path(self, index, prefix='_ann.txt'):       #伪标签 _ann.txt 文件中的内容在训练数据加载过程中被引用，提供伪目标的ID和不确定性信息，作为训练中的额外监督信号
        file_name = self.annotations[index][-1]
        tail = file_name.split('.')[-1]
        _name = '_'.join(file_name.split('/'))
        _name = _name.replace('.' + tail, prefix)
        return f'{self.sync_dir}/{index:06d}_{_name}'

    def save_anns_to_disk(self):
        for i, (res, _, file_name) in enumerate(self.annotations):
            ann = np.concatenate((res, self.uncertainties[i].reshape(-1, 1)), axis=1)
            save_path = self.get_ann_tmp_path(i, '_ann.txt')
            np.savetxt(save_path, ann, fmt="%.2f %.2f %.2f %.2f %d %d %.3f")

    def load_anns_to_mem(self): #如果有伪标签存在，load_anns_to_mem方法会将伪标签加载到 self.annotations 中，并用伪标签替换或补充真实标签中的部分信息（如轨迹ID）
        for i, (res, _, file_name) in enumerate(self.annotations):
            save_path = self.get_ann_tmp_path(i, '_ann.txt')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ann = np.loadtxt(save_path, delimiter=' ').reshape(-1, 7)
            res[:, -1] = ann[:, -2]
            assert (self.annotations[i][0][:, -1] == ann[:, -2]).all()
            # if 'eth' in file_name:
            #     assert (self.annotations[i][0][:, -1] == -1).sum() == 0
            self.uncertainties[i] = ann[:, -1]                               #还存储了不确定性值

    @staticmethod
    def img_path2seq(file_name):
        sep_dict = {
            'VisDrone': '_v/',
            'MOT': 'img1',
            'crowdhuman': ',',
            'eth': 'images',
            'cp': '_0'
        }
        for vid, sep in sep_dict.items():
            if vid in file_name:
                return file_name.split(sep)[0].split('.')[-1]
        if len(os.path.basename(file_name)[:-4].split('-')) == 3:  # bdd100k, stupid
            return os.path.dirname(file_name).split('.')[-1]
        raise ValueError('Unknown dataset: ', file_name)

    @staticmethod
    def check_period(file_name, period):
        if any(d in file_name for d in ['cp_train', 'crowdhuman']):
            return 0
        if any(d in file_name for d in ['ethz_train', 'mot_train', 'mot20_train', 'VisDrone', 'MOT17', 'MOT20']):
            return period
        if len(os.path.basename(file_name)[:-4].split('-')) == 3:  # bdd100k, stupid
            return period
        raise ValueError('Undefined dataset:', file_name)

    @staticmethod
    def get_frame_cnt(file_name):
        if 'eth' in file_name:
            return int(os.path.basename(file_name).split('_')[1]) + 1
        if 'MOT' in file_name:
            return int(os.path.basename(file_name).split('.')[0])
        if len(os.path.basename(file_name)[:-4].split('-')) == 3:  # bdd100k, stupid
            return int(os.path.basename(file_name)[:-4].split('-')[-1])
        return 1  # equals to the 1st frame

    def seq_repeated(self, indices):
        valid_seqs = []
        for index in indices:
            file_name, seq_name = self.annotations[index][1][-4:-2]
            if not self.is_img_static(file_name):
                valid_seqs.append(seq_name)
        return len(valid_seqs) != len(set(valid_seqs))

    def is_img_static(self, file_name):
        return self.check_period(file_name, 2) < 2

    def get_time_scale(self):
        return int(math.pow(self.cur_epoch / self.max_epoch, 3) * self.time_scale + 1)

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def pull_item(self, index, id_off=-1):
        id_ = self.ids[index]                               #id_ = self.ids[index] 表示使用当前图像的 ID 从 self.ids 中获取标注信息

        res, img_info, file_name = self.annotations[index]   #获取融合后的标注数据    在上面的load_anns_to_mem进行融合
        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, img_file
        
        #cv2.imwrite(os.path.join(self.data_dir, "original_images", f"{index}_original.jpg"), img)

        if not self.is_training:
            return img, res.copy(), img_info, np.array([id_])
        else:
            # load auxilliary data
            frame_cnt = self.get_frame_cnt(file_name)
            time_scale = 0 if self.is_img_static(file_name) else self.get_time_scale()
            # time_scale = self.time_scale  # TODO: only for contiguous sequences with detector freezed
            min_delta = min(1, time_scale)  # 1 or 0
            delta = random.randint(min(min_delta, frame_cnt - 1),
                                   min(time_scale, frame_cnt - 1))
            index_prev = index - delta

            res_prev, img_info_prev, file_name_prev = self.annotations[index_prev]
            if self.img_path2seq(file_name) != self.img_path2seq(file_name_prev):  # stupid ETHZ-named paths
                index_prev = index
                res_prev, img_info_prev, file_name_prev = self.annotations[index_prev]

            img_file_prev = os.path.join(self.data_dir, self.name, file_name_prev)
            img_prev = cv2.imread(img_file_prev)
            assert img_prev is not None, img_file_prev

            res_prev = res_prev.copy()
            res = res.copy()
            if self.is_img_static(file_name):  # compatible for Mosaic and Mixup
                res[:, -1] = -np.arange(len(res)) + id_off
                res_prev[:, -1] = -np.arange(len(res)) + id_off
            if len(res) * len(res_prev) > 0:
                img_info = (*img_info[:-2],
                            min(res[:, -1].min(), res_prev[:, -1].min()),
                            max(res[:, -1].max(), res_prev[:, -1].max()))
            else:
                img_info = (*img_info[:-2], 0, 0)
            img_aug, res_aug = self.tracklet_guided_augment(img, res.copy(), index, index_prev)
            #cv2.imwrite(os.path.join(self.data_dir, "augmented_images", f"{index}_augmented.jpg"), img_aug)
            # img_aug, res_aug = img.copy(), res.copy()
           #伪标签和真实标签合在训练过程中通过pull_item方法的返回值img、res、img_info等被引入模型训练流程中
            return (img, img_aug, img_prev), (res, res_aug, res_prev), img_info, np.array([id_])  #(img_info, img_info_prev), np.array([id_, id_prev])

#     def tracklet_guided_augment(self, img, curr_label, curr_index, prev_indices):
#             import os.path as osp
#             from ..data_augment import transform_labels
#             import cv2
#             import numpy as np

#             def _get_perspective_transform(src, dst):
#                 M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
#                 return M

#             # 定义时间窗口，使用前 N 帧
#             N = 2  # 可以根据需要调整时间窗口的大小
#             prev_indices = [curr_index - i for i in range(1, N+1)]
#             valid_prev_indices = [idx for idx in prev_indices if idx >= 0 and len(self.annotations[idx][0]) > 0]

#             if len(valid_prev_indices) == 0 or len(curr_label) == 0:
#                 return img.copy(), curr_label

#             curr_id = curr_label[:, -1].astype(np.int32)
#             motion_info = []  # 存储运动信息

#             for prev_index in valid_prev_indices:
#                 prev_label = self.annotations[prev_index][0].copy()
#                 prev_id = prev_label[:, -1].astype(np.int32)
#                 curr_cls = curr_label[:, 5].astype(np.int32)  # 假设类别标签在第6列
#                 prev_cls = prev_label[:, 5].astype(np.int32)

#                 # 匹配目标 ID 和类别
#                 assign = (curr_id.reshape((-1, 1)) == prev_id.reshape((1, -1))) & (curr_cls.reshape((-1, 1)) == prev_cls.reshape((1, -1)))
#                 i, j = np.nonzero(np.atleast_1d(assign))
#                 n_pair = len(i)
#                 if n_pair == 0:
#                     continue

#                 curr_xyxy = curr_label[i, :4]
#                 prev_xyxy = prev_label[j, :4]

#                 # 计算位移向量（中心点差异）
#                 curr_centers = (curr_xyxy[:, :2] + curr_xyxy[:, 2:]) / 2
#                 prev_centers = (prev_xyxy[:, :2] + prev_xyxy[:, 2:]) / 2
#                 displacement = curr_centers - prev_centers

#                 # 计算运动幅度
#                 motion_magnitude = np.linalg.norm(displacement, axis=1)

#                 # 存储运动信息
#                 for idx in range(len(i)):
#                     motion_info.append({
#                         'curr_idx': i[idx],
#                         'prev_idx': j[idx],
#                         'prev_frame': prev_index,
#                         'displacement': displacement[idx],
#                         'motion_magnitude': motion_magnitude[idx],
#                         'category': curr_cls[i[idx]]
#                     })

#             if len(motion_info) == 0:
#                 return img.copy(), curr_label

#             # 根据类别调整运动幅度的权重（类别特定的增强策略）
#             for info in motion_info:
#                 if info['category'] == 0:  # 假设行人类别标记为 1
#                     info['motion_magnitude'] *= 1.0  # 行人可能有较慢的运动
#                 elif info['category'] == 3:  # 假设车辆类别标记为 2
#                     info['motion_magnitude'] *= 1.5  # 车辆可能有较快的运动

#             # 从 motion_info 中提取运动幅度，并归一化为概率分布
#             motion_magnitudes = np.array([info['motion_magnitude'] for info in motion_info])
#             motion_prob = motion_magnitudes / np.sum(motion_magnitudes)

#             # 根据运动幅度随机选择一个锚点
#             anchor_idx = np.random.choice(len(motion_info), p=motion_prob)
#             anchor_info = motion_info[anchor_idx]

#             # 获取源点（当前帧目标框的四个顶点，用于透视变换）
#             curr_idx = anchor_info['curr_idx']
#             src_bbox = curr_label[curr_idx, :4]
#             x1, y1, x2, y2 = src_bbox
#             src = np.array([
#                 [x1, y1],  # 左上角
#                 [x2, y1],  # 右上角
#                 [x2, y2],  # 右下角
#                 [x1, y2]   # 左下角
#             ])

#             # 获取目标点（先前帧目标框的四个顶点）
#             prev_idx = anchor_info['prev_idx']
#             prev_frame = anchor_info['prev_frame']
#             prev_bbox = self.annotations[prev_frame][0][prev_idx, :4]
#             x1_p, y1_p, x2_p, y2_p = prev_bbox
#             dst = np.array([
#                 [x1_p, y1_p],
#                 [x2_p, y1_p],
#                 [x2_p, y2_p],
#                 [x1_p, y2_p]
#             ])

#             # 计算透视变换矩阵
#             M = _get_perspective_transform(src, dst)

#             # 应用透视变换
#             height, width, _ = img.shape
#             dsize = (width, height)
#             aug_img = cv2.warpPerspective(img, M, dsize=dsize, borderValue=(114, 114, 114))
#             aug_ann = transform_labels(curr_label, M, dsize, perspective=True)
            
#             # 颜色抖动
#             if np.random.rand() < 0.5:
#                 aug_img = self.color_jitter(aug_img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

#             # 随机水平翻转
#             if np.random.rand() < 0.5:
#                 aug_img, aug_ann = self.random_horizontal_flip(aug_img, aug_ann)

#             # 随机裁剪
#             if np.random.rand() < 0.3:
#                 aug_img, aug_ann = self.random_crop(aug_img, aug_ann)
            
#             return aug_img, aug_ann


    def tracklet_guided_augment(self, img, curr_label, curr_index, prev_indices):
        import os.path as osp
        from ..data_augment import transform_labels
        import cv2
        import numpy as np
        from numpy.random import Generator
        
        def _get_perspective_transform(src, dst):
            M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
            return M
        # 定义时间窗口，使用前 N 帧
        N = 2  # 可以根据需要调整时间窗口的大小
        local_prev_indices = [curr_index - i for i in range(1, N+1)]
        valid_prev_indices = [idx for idx in local_prev_indices if idx >= 0 and len(self.annotations[idx][0]) > 0]

        if len(valid_prev_indices) == 0 or len(curr_label) == 0:
            return img.copy(), curr_label

        curr_id = curr_label[:, -1].astype(np.int32)
        motion_info = []

        for prev_index in valid_prev_indices:
            prev_label = self.annotations[prev_index][0].copy()
            prev_id = prev_label[:, -1].astype(np.int32)
            curr_cls = curr_label[:, 5].astype(np.int32)
            prev_cls = prev_label[:, 5].astype(np.int32)

            # 匹配目标 ID 和类别
            assign = (curr_id.reshape((-1, 1)) == prev_id.reshape((1, -1))) & (curr_cls.reshape((-1, 1)) == prev_cls.reshape((1, -1)))
            i, j = np.nonzero(np.atleast_1d(assign))
            n_pair = len(i)
            if n_pair == 0:
                continue

            curr_xyxy = curr_label[i, :4]
            prev_xyxy = prev_label[j, :4]

            # 计算位移向量（中心点差异）
            curr_centers = (curr_xyxy[:, :2] + curr_xyxy[:, 2:]) / 2
            prev_centers = (prev_xyxy[:, :2] + prev_xyxy[:, 2:]) / 2
            displacement = curr_centers - prev_centers

            # 计算运动幅度
            motion_magnitude = np.linalg.norm(displacement, axis=1)

            # 放大运动幅度
            motion_magnitude *= 1.0  # 放大运动幅度

            # 存储运动信息
            for idx in range(len(i)):
                motion_info.append({
                    'curr_idx': i[idx],
                    'prev_idx': j[idx],
                    'prev_frame': prev_index,
                    'displacement': displacement[idx],
                    'motion_magnitude': motion_magnitude[idx],
                    'category': curr_cls[i[idx]]
                })

        if len(motion_info) == 0:
            return img.copy(), curr_label

        # 根据类别调整运动幅度的权重（类别特定的增强策略）
        for info in motion_info:
            if info['category'] == 0:  # 假设行人类别标记为 0
                info['motion_magnitude'] *= 1.0  # 行人可能有较慢的运动
            elif info['category'] == 3:  # 假设车辆类别标记为 3
                info['motion_magnitude'] *= 1.5  # 车辆可能有较快的运动

        # 从 motion_info 中提取运动幅度，并归一化为概率分布
        motion_magnitudes = np.array([info['motion_magnitude'] for info in motion_info])
        motion_prob = motion_magnitudes / np.sum(motion_magnitudes)

        # 根据运动幅度随机选择一个锚点
        rng = np.random.default_rng()
        anchor_idx = rng.choice(len(motion_info), p=motion_prob)
        anchor_info = motion_info[anchor_idx]

        # 获取源点（当前帧目标框的四个顶点，用于透视变换）
        curr_idx = anchor_info['curr_idx']
        src_bbox = curr_label[curr_idx, :4]
        x1, y1, x2, y2 = src_bbox
        src = np.array([
            [x1, y1],  # 左上角
            [x2, y1],  # 右上角
            [x2, y2],  # 右下角
            [x1, y2]   # 左下角
        ])

        # 获取目标点（先前帧目标框的四个顶点）
        prev_idx = anchor_info['prev_idx']
        prev_frame = anchor_info['prev_frame']
        prev_bbox = self.annotations[prev_frame][0][prev_idx, :4]
        x1_p, y1_p, x2_p, y2_p = prev_bbox
        dst = np.array([
            [x1_p, y1_p],
            [x2_p, y1_p],
            [x2_p, y2_p],
            [x1_p, y2_p]
        ])

        # 计算透视变换矩阵
        M = _get_perspective_transform(src, dst)

        # 应用透视变换
        height, width, _ = img.shape
        dsize = (width, height)
        aug_img = cv2.warpPerspective(img, M, dsize=dsize, borderValue=(114, 114, 114))
        aug_ann = transform_labels(curr_label, M, dsize, perspective=True)

#         # 调整噪声和模糊的强度，减轻对图像质量的影响
#         # 降低噪声的标准差，减少模糊的核大小
#         if rng.rand() < 0.3:  # 30%的概率添加轻微的噪声和模糊
#             aug_img = self.add_noise_and_blur(aug_img, noise_std=5, blur_ksize=3)

        # 添加其他数据增强方法
        # 颜色抖动
        if rng.random() < 0.5:
            aug_img = self.color_jitter(aug_img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        return aug_img, aug_ann

#          #随机水平翻转
#         if rng.random() < 0.5:
#             aug_img, aug_ann = self.random_horizontal_flip(aug_img, aug_ann)

#         # 随机裁剪
#         if rng.random() < 0.3:
#             aug_img, aug_ann = self.random_crop(aug_img, aug_ann)

         #return aug_img, aug_ann
        

    def color_jitter(self, img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        # 转换为HSV空间，调整亮度、饱和度和色调
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # 调整亮度
        img[:, :, 2] *= np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        # 调整饱和度
        img[:, :, 1] *= np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        # 调整色调
        img[:, :, 0] += np.random.uniform(-hue * 180, hue * 180)
        img[:, :, 0] = np.mod(img[:, :, 0], 180)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        # 调整对比度
        img = img.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        img *= alpha
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

#     def random_horizontal_flip(self, img, ann):
#         if np.random.rand() < 0.5:
#             img = cv2.flip(img, 1)
#             width = img.shape[1]
#             ann[:, [0, 2]] = width - ann[:, [2, 0]]  # 翻转bbox的x坐标
#         return img, ann

#     def random_crop(self, img, ann):
#         height, width, _ = img.shape
#         # 随机裁剪区域
#         crop_x = np.random.randint(0, width // 10)
#         crop_y = np.random.randint(0, height // 10)
#         crop_w = width - np.random.randint(0, width // 10)
#         crop_h = height - np.random.randint(0, height // 10)
    
#         img = img[crop_y:crop_h, crop_x:crop_w]
    
#         # 调整标注
#         ann[:, [0, 2]] = ann[:, [0, 2]] - crop_x
#         ann[:, [1, 3]] = ann[:, [1, 3]] - crop_y
    
#         # 过滤掉超出图像边界的bbox
#         ann[:, [0, 2]] = np.clip(ann[:, [0, 2]], 0, crop_w - crop_x)
#         ann[:, [1, 3]] = np.clip(ann[:, [1, 3]], 0, crop_h - crop_y)
    
#         # 过滤面积过小的bbox
#         keep = np.logical_and((ann[:, 2] - ann[:, 0]) > 5, (ann[:, 3] - ann[:, 1]) > 5)
#         ann = ann[keep]
    
#         return img, ann

    

#     def tracklet_guided_augment(self, img, curr_label, curr_index, prev_indices):
#         import os.path as osp
#         from ..data_augment import transform_labels
#         import cv2
#         import numpy as np

#         def _get_perspective_transform(src, dst):
#             M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
#             return M

#         # 定义时间窗口，使用前 N 帧
#         N = 2  # 可以根据需要调整时间窗口的大小
#         prev_indices = [curr_index - i for i in range(1, N+1)]
#         valid_prev_indices = [idx for idx in prev_indices if idx >= 0 and len(self.annotations[idx][0]) > 0]

#         if len(valid_prev_indices) == 0 or len(curr_label) == 0:
#             return img.copy(), curr_label

#         curr_id = curr_label[:, -1].astype(np.int32)
#         motion_info = []  # 存储运动信息

#         for prev_index in valid_prev_indices:
#             prev_label = self.annotations[prev_index][0].copy()
#             prev_id = prev_label[:, -1].astype(np.int32)
#             curr_cls = curr_label[:, 1].astype(np.int32)  # 获取类别标签（假设类别在第5列）
#             prev_cls = prev_label[:, 1].astype(np.int32)

#             # 匹配目标 ID 和类别
#             assign = (curr_id.reshape((-1, 1)) == prev_id.reshape((1, -1))) & (curr_cls.reshape((-1, 1)) == prev_cls.reshape((1, -1)))
#             i, j = np.nonzero(np.atleast_1d(assign))
#             n_pair = len(i)
#             if n_pair == 0:
#                 continue

#             curr_xyxy = curr_label[i, :4]
#             prev_xyxy = prev_label[j, :4]

#             # 计算位移向量（中心点差异）
#             curr_centers = (curr_xyxy[:, :2] + curr_xyxy[:, 2:]) / 2
#             prev_centers = (prev_xyxy[:, :2] + prev_xyxy[:, 2:]) / 2
#             displacement = curr_centers - prev_centers

#             # 计算运动幅度
#             motion_magnitude = np.linalg.norm(displacement, axis=1)

#             # 存储运动信息
#             for idx in range(len(i)):
#                 motion_info.append({
#                     'curr_idx': i[idx],
#                     'prev_idx': j[idx],
#                     'prev_frame': prev_index,
#                     'displacement': displacement[idx],
#                     'motion_magnitude': motion_magnitude[idx],
#                     'category': curr_cls[i[idx]]
#                 })

#         if len(motion_info) == 0:
#             return img.copy(), curr_label

#     # 根据类别调整运动幅度的权重（类别特定的增强策略）
#         for info in motion_info:
#             if info['category'] == 'pedestrian':  # 假设行人类别标记为 'pedestrian'
#                 info['motion_magnitude'] *= 1.0  # 行人可能有较慢的运动
#             elif info['category'] == 'car':  # 假设车辆类别标记为 'vehicle'
#                 info['motion_magnitude'] *= 1.5  # 车辆可能有较快的运动

#     # 从 motion_info 中提取运动幅度，并归一化为概率分布
#         motion_magnitudes = np.array([info['motion_magnitude'] for info in motion_info])
#         motion_prob = motion_magnitudes / np.sum(motion_magnitudes)

#     # 根据运动幅度随机选择一个锚点
#         anchor_idx = np.random.choice(len(motion_info), p=motion_prob)
#         anchor_info = motion_info[anchor_idx]

#     # 获取源点（当前帧目标框的四个顶点，用于透视变换）
#         curr_idx = anchor_info['curr_idx']
#         src_bbox = curr_label[curr_idx, :4]
#         x1, y1, x2, y2 = src_bbox
#         src = np.array([
#             [x1, y1],  # 左上角
#             [x2, y1],  # 右上角
#             [x2, y2],  # 右下角
#             [x1, y2]   # 左下角
#         ])

#     # 获取目标点（先前帧目标框的四个顶点）
#         prev_idx = anchor_info['prev_idx']
#         prev_frame = anchor_info['prev_frame']
#         prev_bbox = self.annotations[prev_frame][0][prev_idx, :4]
#         x1_p, y1_p, x2_p, y2_p = prev_bbox
#         dst = np.array([
#             [x1_p, y1_p],
#             [x2_p, y1_p],
#             [x2_p, y2_p],
#             [x1_p, y2_p]
#         ])

#         # 计算透视变换矩阵
#         M = _get_perspective_transform(src, dst)

#         # 应用透视变换
#         height, width, _ = img.shape
#         dsize = (width, height)
#         aug_img = cv2.warpPerspective(img, M, dsize=dsize, borderValue=(114, 114, 114))
#         aug_ann = transform_labels(curr_label, M, dsize, perspective=True)

#         # 随机添加遮挡（模拟无人机视角下的遮挡）
#         if np.random.rand() < 0.3:  # 30%的概率添加遮挡
#             aug_img = self.add_random_occlusion(aug_img)

# #         # 随机添加噪声和模糊（模拟无人机摄像头质量下降）
# #         if np.random.rand() < 0.3:
# #             aug_img = self.add_noise_and_blur(aug_img)

# #         return aug_img, aug_ann

#     def add_random_occlusion(self, img):
#             height, width, _ = img.shape
#             # 随机生成遮挡的位置和大小
#             occ_width = np.random.randint(width // 10, width // 5)
#             occ_height = np.random.randint(height // 10, height // 5)
#             x1 = np.random.randint(0, width - occ_width)
#             y1 = np.random.randint(0, height - occ_height)
#             x2 = x1 + occ_width
#             y2 = y1 + occ_height
#             # 随机颜色
#             color = [np.random.randint(0, 255) for _ in range(3)]
#             img[y1:y2, x1:x2] = color
#             return img

#     def add_noise_and_blur(self, img):
#         # 添加高斯噪声
#         noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
#         img = cv2.add(img, noise)
#         # 添加模糊
#         ksize = np.random.choice([3, 3])
#         img = cv2.GaussianBlur(img, (ksize, ksize), 0)
#         return img


    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def get_seq_names(self):
        seq_names = []
        for annotation in self.annotations:
            seq_name = self.img_path2seq(annotation[-1])
            if seq_name not in seq_names:
                seq_names.append(seq_name)

        return seq_names