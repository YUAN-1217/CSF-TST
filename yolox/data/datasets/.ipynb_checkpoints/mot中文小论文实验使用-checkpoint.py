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
            # 定义新的输出路径
            #output_dir = "/root/lanyun-tmp/MYmot/yolox/data/datasets/aug"
            # 确保输出目录存在
            #os.makedirs(output_dir, exist_ok=True)
            img_aug, res_aug = self.tracklet_guided_augment(img, res.copy(), index, index_prev)
            #cv2.imwrite(os.path.join(output_dir, f"{index}_augmented.jpg"), img_aug)

            return (img, img_aug, img_prev), (res, res_aug, res_prev), img_info, np.array([id_])  #(img_info, img_info_prev), np.array([id_, id_prev])

    def tracklet_guided_augment(self, img, curr_label, curr_index, prev_indices):
        import os.path as osp
        from ..data_augment import transform_labels
        import cv2
        import numpy as np
        from numpy.random import Generator
        
        def _get_perspective_transform(src, dst):
            M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
            return M

        # def is_valid_transform(M, img_shape):
        #     """检查透视变换是否有效"""
        #     h, w = img_shape[:2]
        #     corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        #     transformed = cv2.perspectiveTransform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
            
        #     # 修复广播问题
        #     x_bounds = (transformed[:, 0] >= -0.2*w) & (transformed[:, 0] < 1.2*w)
        #     y_bounds = (transformed[:, 1] >= -0.2*h) & (transformed[:, 1] < 1.2*h)
        #     in_bounds = np.all(x_bounds & y_bounds)
            
        #     # 检查变换是否过于剧烈
        #     original_area = w * h
        #     transformed_area = cv2.contourArea(transformed.astype(np.float32))
        #     area_ratio = transformed_area / original_area
            
        #     return in_bounds and (0.3 < area_ratio < 3.0)
            
        # 定义时间窗口，使用前 N 帧
        N = 1  # 修改为更大的时间窗口
        local_prev_indices = [curr_index - i for i in range(1, N+1)]
        valid_prev_indices = [idx for idx in local_prev_indices if idx >= 0 and len(self.annotations[idx][0]) > 0]

        if len(valid_prev_indices) == 0 or len(curr_label) == 0:
            return img.copy(), curr_label

        curr_id = curr_label[:, -1].astype(np.int32)
        motion_info = []
        
        # 初始化累积运动向量
        accumulated_motion = np.zeros((len(curr_label), 2))  # 存储累积运动向量
        motion_weights = np.zeros(len(curr_label))  # 存储运动权重

        # 收集所有历史帧的运动信息
        for frame_idx, prev_index in enumerate(valid_prev_indices):
            prev_label = self.annotations[prev_index][0].copy()
            prev_id = prev_label[:, -1].astype(np.int32)
            curr_cls = curr_label[:, 5].astype(np.int32)
            prev_cls = prev_label[:, 5].astype(np.int32)

            # 匹配目标 ID 和类别
            assign = (curr_id.reshape((-1, 1)) == prev_id.reshape((1, -1))) & (curr_cls.reshape((-1, 1)) == prev_cls.reshape((1, -1)))
            i, j = np.nonzero(np.atleast_1d(assign))
            
            if len(i) == 0:
                continue

            curr_xyxy = curr_label[i, :4]
            prev_xyxy = prev_label[j, :4]

            # 计算位移向量
            curr_centers = (curr_xyxy[:, :2] + curr_xyxy[:, 2:]) / 2
            prev_centers = (prev_xyxy[:, :2] + prev_xyxy[:, 2:]) / 2
            displacement = curr_centers - prev_centers

            # 计算时间权重（越近的帧权重越大）
            #time_weight = np.exp(-frame_idx / N)  # 指数衰减权重

            # 加快时间权重衰减
            time_weight = np.exp(-2 * frame_idx / N)  # 加快衰减速率

            # 累积运动向量（考虑时间权重）
            for idx, curr_idx in enumerate(i):
                motion_magnitude = np.linalg.norm(displacement[idx])
                # 根据类别调整运动幅度
                if curr_cls[curr_idx] == 0:  # 行人
                    motion_magnitude *= 1.2
                elif curr_cls[curr_idx] == 3:  # 车辆
                    motion_magnitude *= 1.5
                    
                # 累积运动和权重
                #accumulated_motion[curr_idx] += displacement[idx] * motion_magnitude * time_weight
                # 添加运动限制
                max_motion = 50  # 限制最大运动幅度
                displacement_norm = np.clip(displacement[idx], -max_motion, max_motion)
                accumulated_motion[curr_idx] += displacement_norm * min(motion_magnitude, max_motion) * time_weight

                motion_weights[curr_idx] += time_weight

        # 标准化累积运动
        valid_mask = motion_weights > 0
        accumulated_motion[valid_mask] /= motion_weights[valid_mask].reshape(-1, 1)

        # 选择运动最显著的目标作为锚点
        motion_magnitudes = np.linalg.norm(accumulated_motion, axis=1)
        if not np.any(motion_magnitudes > 0):
            return img.copy(), curr_label

        # 选择运动最大的目标进行变换
        anchor_idx = np.argmax(motion_magnitudes)
        motion_vector = accumulated_motion[anchor_idx]

        # 生成透视变换的源点和目标点
        src_bbox = curr_label[anchor_idx, :4]
        x1, y1, x2, y2 = src_bbox
        src = np.array([
            [x1, y1],  # 左上角
            [x2, y1],  # 右上角
            [x2, y2],  # 右下角
            [x1, y2]   # 左下角
        ])

        # 基于累积运动生成目标点
        #motion_scale = min(1.0, motion_magnitudes[anchor_idx] / 100)  # 限制变换幅度
        motion_scale = min(0.3, motion_magnitudes[anchor_idx] / 200)  # 降低变换幅度
        dst = src + motion_vector.reshape(1, 2) * motion_scale

        # 计算透视变换矩阵
        M = _get_perspective_transform(src, dst)

        # # 在应用透视变换前使用检查
        # if not is_valid_transform(M, img.shape):
        #     return img.copy(), curr_label

        # 应用透视变换
        height, width, _ = img.shape
        dsize = (width, height)
        aug_img = cv2.warpPerspective(img, M, dsize=dsize, borderValue=(114, 114, 114))
        aug_ann = transform_labels(curr_label, M, dsize, perspective=True)

        # 添加颜色抖动
        rng = np.random.default_rng()
        if rng.random() < 0.5:
            aug_img = self.color_jitter(aug_img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        return aug_img, aug_ann
        

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