#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
import os.path as osp

from .superglue_models.superpoint import SuperPoint
from .superglue_models.superglue import SuperGlue
from .lightglue.lightglue.lightglue import LightGlue
from .lightglue.lightglue.superpoint import SuperPoint as LightGlueSuperPoint

# XFeat import
try:
    from .xfeat.modules.xfeat import XFeat
    XFEAT_AVAILABLE = True
except:
    XFEAT_AVAILABLE = False


class GMC:
    def __init__(self, method='orb', downscale=2, verbose=None):
        super(GMC, self).__init__()

        self.method = method
        self.downscale = max(1, int(downscale))
        self.xfeat_matcher = 'xfeat'  # 可选: 'lighterglue', 'xfeat', 'xfeat_star'   , xfeat_matcher='lighterglue'

        if self.method == 'orb' or self.method == 'superglue' or self.method == 'lightglue' or self.method == 'xfeat':  # 添加xfeat的文件处理
            self.detector = cv2.FastFeatureDetector_create(20) if self.method == 'orb' else None
            self.extractor = cv2.ORB_create() if self.method == 'orb' else None
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING) if self.method == 'orb' else None
            
            seqName = verbose[0]
            fileDir = verbose[1]

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]
                
            self.gmcFile = open(f"yolox/tracker/GMC_files/{fileDir}/GMC-{seqName}.txt", 'w+')

            if self.method == 'superglue':
                # Initialize SuperPoint and SuperGlue
                self.superpoint = SuperPoint({
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 1024
                }).eval().cuda()
                
                self.superglue = SuperGlue({
                    'weights': 'indoor',
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2,
                }).eval().cuda()
                
            elif self.method == 'lightglue':
                # Initialize LightGlue with SuperPoint
                self.lightglue_superpoint = LightGlueSuperPoint(
                    nms_radius=4,
                    detection_threshold=0.005,
                    max_num_keypoints=1024
                ).eval().cuda()
                
                self.lightglue = LightGlue(features='superpoint').eval().cuda()
                
            elif self.method == 'xfeat':
                # Initialize XFeat
                if not XFEAT_AVAILABLE:
                    raise ImportError("XFeat not available. Please check the xfeat module installation.")
                
                # XFeat权重路径
                weights_path = osp.join(osp.dirname(__file__), 'xfeat', 'weights', 'xfeat.pt')
                self.xfeat = XFeat(
                    weights=weights_path,
                    top_k=2048,  # 特征点数量
                    detection_threshold=0.05
                ).eval()
                
                # XFeat可以在CPU或GPU上运行
                if torch.cuda.is_available():
                    self.xfeat = self.xfeat.cuda()

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == 'file' or self.method == 'files':
            seqName = verbose[0]
            # MOT17_ablation, MOTChallenge, VisDrone/test-dev, BDD100K/val, BDD100K/test
            fileDir = verbose[1]
            filePath = f'yolox/tracker/GMC_files/{fileDir}'

            if '-FRCNN' in seqName:
                seqName = seqName[:-6]
            elif '-DPM' in seqName:
                seqName = seqName[:-4]
            elif '-SDP' in seqName:
                seqName = seqName[:-4]

            self.gmcFile = open(filePath + "/GMC-" + seqName + ".txt", 'r')

            if self.gmcFile is None:
                raise ValueError("Error: Unable to open GMC file in directory:" + filePath)
        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'
        else:
            raise ValueError("Error: Unknown CMC method:" + method)

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False
        self.frameCnt = 0

    def apply(self, raw_frame, detections=None):
        if self.method == 'superglue':
            try:
                H = self.applySuperGlue(raw_frame, detections)
                # 保存GMC结果到文件
                self.gmcFile.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t\n' % \
                                 (self.frameCnt, H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2]))
                self.frameCnt += 1
            except:
                H = np.array([[1., 0., 0.], [0., 1., 0.]])
            return H
        elif self.method == 'lightglue':
            try:
                H = self.applyLightGlue(raw_frame, detections)
                # 保存GMC结果到文件
                self.gmcFile.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t\n' % \
                                 (self.frameCnt, H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2]))
                self.frameCnt += 1
            except:
                H = np.array([[1., 0., 0.], [0., 1., 0.]])
            return H
        elif self.method == 'xfeat':
            try:
                H = self.applyXFeat(raw_frame, detections)
                # 保存GMC结果到文件
                self.gmcFile.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t\n' % \
                                 (self.frameCnt, H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2]))
                self.frameCnt += 1
            except Exception as e:
                print(f"XFeat GMC failed: {str(e)}")
                H = np.array([[1., 0., 0.], [0., 1., 0.]])
            return H
        elif self.method == 'orb' or self.method == 'sift':
            try:
                H = self.applyFeaures(raw_frame, detections)
            except:
                H = np.array([[1., 0., 0.], [0., 1., 0.]])
            self.gmcFile.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t\n' % \
                               (self.frameCnt, H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2]))
            self.frameCnt += 1
            return H
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'file':
            return self.applyFile(raw_frame, detections)
        elif self.method == 'none':
            return np.eye(2, 3)
        else:
            return np.eye(2, 3)

    def applyEcc(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Run the ECC algorithm. The results are stored in warp_matrix.
        # (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            print('Warning: find transform failed. Set warp as identity')

        return H

    def applyFeaures(self, raw_frame, detections=None):

        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image (TODO: consider using pyramids)
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # find the keypoints
        mask = np.zeros_like(frame)
        # mask[int(0.05 * height): int(0.95 * height), int(0.05 * width): int(0.95 * width)] = 255
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors.
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []

        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                   prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Draw the keypoint matches on the output image
        if 0:
            matches_img = np.hstack((self.prevFrame, frame))
            matches_img = cv2.cvtColor(matches_img, cv2.COLOR_GRAY2BGR)
            W = np.size(self.prevFrame, 1)
            for m in goodMatches:
                prev_pt = np.array(self.prevKeyPoints[m.queryIdx].pt, dtype=np.int_)
                curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.int_)
                curr_pt[0] += W
                color = np.random.randint(0, 255, (3,))
                color = (int(color[0]), int(color[1]), int(color[2]))

                matches_img = cv2.line(matches_img, prev_pt, curr_pt, tuple(color), 1, cv2.LINE_AA)
                matches_img = cv2.circle(matches_img, prev_pt, 2, tuple(color), -1)
                matches_img = cv2.circle(matches_img, curr_pt, 2, tuple(color), -1)

            plt.figure()
            plt.imshow(matches_img)
            plt.show()

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySuperGlue(self, raw_frame, detections=None):
        H = np.eye(2, 3)
        
        if not self.initializedFirstFrame:
            self.prevFrame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            if self.downscale > 1.0:
                self.prevFrame = cv2.resize(self.prevFrame, 
                    (raw_frame.shape[1] // self.downscale, 
                     raw_frame.shape[0] // self.downscale))
            self.initializedFirstFrame = True
            return H

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, 
                (raw_frame.shape[1] // self.downscale,
                 raw_frame.shape[0] // self.downscale))

        try:
            with torch.no_grad():
                # 转换为tensor并归一化
                prev_tensor = torch.from_numpy(self.prevFrame)[None, None].float().cuda() / 255.0
                curr_tensor = torch.from_numpy(frame)[None, None].float().cuda() / 255.0

                # 提取特征点
                prev_data = self.superpoint({'image': prev_tensor})
                curr_data = self.superpoint({'image': curr_tensor})

                # 确保张量维度正确
                prev_kpts = prev_data['keypoints'][0]  # [N,2]
                curr_kpts = curr_data['keypoints'][0]  # [N,2]
                prev_desc = prev_data['descriptors'][0].transpose(0,1)  # [N,D]
                curr_desc = curr_data['descriptors'][0].transpose(0,1)  # [N,D]
                prev_scores = prev_data['scores'][0]  # [N]
                curr_scores = curr_data['scores'][0]  # [N]

                if len(prev_kpts) > 0 and len(curr_kpts) > 0:
                    # 准备数据
                    data = {
                        'image0': prev_tensor,  # [1,1,H,W]
                        'image1': curr_tensor,  # [1,1,H,W]
                        'keypoints0': prev_kpts[None],  # [1,N,2]
                        'keypoints1': curr_kpts[None],  # [1,N,2]
                        'descriptors0': prev_desc.transpose(0,1)[None],  # [1,D,N]
                        'descriptors1': curr_desc.transpose(0,1)[None],  # [1,D,N]
                        'scores0': prev_scores[None],  # [1,N]
                        'scores1': curr_scores[None],  # [1,N]
                    }

                    # 匹配特征点
                    pred = self.superglue(data)
                    matches = pred['matches0'][0]  # [N]
                    valid = matches > -1

                    # 提取匹配点对
                    kpts0 = prev_kpts.cpu().numpy()
                    kpts1 = curr_kpts.cpu().numpy()
                    matches = matches.cpu().numpy()

                    pts1 = kpts0[valid.cpu().numpy()]
                    pts2 = kpts1[matches[valid.cpu().numpy()]]

                    # 计算仿射变换
                    if len(pts1) >= 4:
                        if self.downscale > 1.0:
                            pts1 *= self.downscale
                            pts2 *= self.downscale
                        H, inliers = cv2.estimateAffinePartial2D(pts1, pts2, cv2.RANSAC,
                                                                ransacReprojThreshold=4.0,
                                                                confidence=0.999,
                                                                maxIters=200)
                        if H is None:
                            H = np.eye(2, 3)

        except Exception as e:
            print(f"SuperGlue matching failed: {str(e)}")
            H = np.eye(2, 3)
            
        # Store current frame
        self.prevFrame = frame.copy()
        return H

    def applyLightGlue(self, raw_frame, detections=None):
        H = np.eye(2, 3)
        
        if not self.initializedFirstFrame:
            self.prevFrame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            if self.downscale > 1.0:
                self.prevFrame = cv2.resize(self.prevFrame, 
                    (raw_frame.shape[1] // self.downscale, 
                     raw_frame.shape[0] // self.downscale))
            self.initializedFirstFrame = True
            return H

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, 
                (raw_frame.shape[1] // self.downscale,
                 raw_frame.shape[0] // self.downscale))

        try:
            with torch.no_grad():
                # 转换为tensor并归一化
                prev_tensor = torch.from_numpy(self.prevFrame)[None, None].float().cuda() / 255.0
                curr_tensor = torch.from_numpy(frame)[None, None].float().cuda() / 255.0

                # 提取特征点和描述子
                prev_data = self.lightglue_superpoint({'image': prev_tensor})
                curr_data = self.lightglue_superpoint({'image': curr_tensor})

                # 检查是否有足够的特征点
                if prev_data['keypoints'].shape[1] == 0 or curr_data['keypoints'].shape[1] == 0:
                    return H

                # 准备LightGlue输入数据
                data = {
                    'image0': {
                        'keypoints': prev_data['keypoints'],
                        'descriptors': prev_data['descriptors'],
                        'image': prev_tensor,
                        'image_size': torch.tensor([[self.prevFrame.shape[1], self.prevFrame.shape[0]]], device='cuda')
                    },
                    'image1': {
                        'keypoints': curr_data['keypoints'], 
                        'descriptors': curr_data['descriptors'],
                        'image': curr_tensor,
                        'image_size': torch.tensor([[frame.shape[1], frame.shape[0]]], device='cuda')
                    }
                }

                # 使用LightGlue进行匹配
                pred = self.lightglue(data)
                matches0 = pred['matches0'][0]  # [N]
                matching_scores0 = pred['matching_scores0'][0]  # [N]
                
                # 获取有效匹配
                valid = matches0 > -1
                
                if valid.sum() >= 4:
                    # 提取匹配点对
                    kpts0 = prev_data['keypoints'][0].cpu().numpy()  # [N, 2]
                    kpts1 = curr_data['keypoints'][0].cpu().numpy()  # [N, 2]
                    matches = matches0.cpu().numpy()
                    
                    pts1 = kpts0[valid.cpu().numpy()]
                    pts2 = kpts1[matches[valid.cpu().numpy()]]
                    
                    # 计算仿射变换
                    if self.downscale > 1.0:
                        pts1 *= self.downscale
                        pts2 *= self.downscale
                    H, inliers = cv2.estimateAffinePartial2D(pts1, pts2, cv2.RANSAC,
                                                            ransacReprojThreshold=4.0,
                                                            confidence=0.999,
                                                            maxIters=200)
                    if H is None:
                        H = np.eye(2, 3)

        except Exception as e:
            print(f"LightGlue matching failed: {str(e)}")
            H = np.eye(2, 3)
            
        # Store current frame
        self.prevFrame = frame.copy()
        return H

    def applyXFeat(self, raw_frame, detections=None):
        """使用XFeat进行特征提取,支持三种匹配方法对比
        
        匹配方法说明:
        - 'lighterglue': 使用LighterGlue GNN匹配器(推荐,最准确但速度较慢)
        - 'xfeat': 使用简单的互最近邻+余弦相似度匹配(快速但准确度较低)
        - 'xfeat_star': 使用粗到精的双阶段匹配(精度和速度折中方案)
        """
        H = np.eye(2, 3)
        
        if not self.initializedFirstFrame:
            self.prevFrame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            if self.downscale > 1.0:
                height, width = self.prevFrame.shape
                self.prevFrame = cv2.resize(self.prevFrame, 
                    (width // self.downscale, height // self.downscale))
            self.initializedFirstFrame = True
            return H

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1.0:
            height, width = frame.shape
            frame = cv2.resize(frame, 
                (width // self.downscale, height // self.downscale))

        try:
            with torch.no_grad():
                # 根据不同的匹配方法使用不同的API
                if self.xfeat_matcher == 'xfeat':
                    # 方法1: match_xfeat 直接传入图像
                    # 需要将2维灰度图转为3维 (H,W) -> (H,W,1)
                    prev_input = self.prevFrame[..., None]  # 添加通道维度
                    curr_input = frame[..., None]           # 添加通道维度
                    
                    mkpts0, mkpts1 = self.xfeat.match_xfeat(
                        prev_input,  # (H,W,1)
                        curr_input,  # (H,W,1)
                        top_k=2048,
                        min_cossim=0.82
                    )
                    
                elif self.xfeat_matcher == 'xfeat_star':
                    # 方法2: match_xfeat_star 也是直接传入图像
                    prev_input = self.prevFrame[..., None]
                    curr_input = frame[..., None]
                    
                    mkpts0, mkpts1 = self.xfeat.match_xfeat_star(
                        prev_input,  # (H,W,1)
                        curr_input,  # (H,W,1)
                        top_k=2048
                    )
                    
                elif self.xfeat_matcher == 'lighterglue':
                    # 方法3: lighterglue 需要先提取特征
                    # detectAndCompute可以接受2维或3维输入，内部会自动处理
                    prev_out = self.xfeat.detectAndCompute(self.prevFrame, top_k=2048)[0]
                    curr_out = self.xfeat.detectAndCompute(frame, top_k=2048)[0]
                    
                    # 检查特征点数量
                    if len(prev_out['keypoints']) == 0 or len(curr_out['keypoints']) == 0:
                        self.prevFrame = frame.copy()
                        return H
                    
                    # 添加图像尺寸信息
                    prev_out['image_size'] = (self.prevFrame.shape[1], self.prevFrame.shape[0])
                    curr_out['image_size'] = (frame.shape[1], frame.shape[0])
                    
                    mkpts0, mkpts1, idxs = self.xfeat.match_lighterglue(
                        prev_out, 
                        curr_out,
                        min_conf=0.1
                    )
                    
                else:
                    raise ValueError(f"Unknown xfeat_matcher: {self.xfeat_matcher}. "
                                f"Choose from ['lighterglue', 'xfeat', 'xfeat_star']")
                
                # Step 3: 计算仿射变换
                if len(mkpts0) >= 4:
                    # 恢复到原始尺度
                    if self.downscale > 1.0:
                        mkpts0 = mkpts0 * self.downscale
                        mkpts1 = mkpts1 * self.downscale
                    
                    # 使用RANSAC计算仿射变换
                    H, inliers = cv2.estimateAffinePartial2D(
                        mkpts0, mkpts1, 
                        cv2.RANSAC,
                        ransacReprojThreshold=4.0,
                        confidence=0.999,
                        maxIters=200
                    )
                    
                    if H is None:
                        H = np.eye(2, 3)

        except Exception as e:
            print(f"XFeat matching failed (method={self.xfeat_matcher}): {str(e)}")
            import traceback
            traceback.print_exc()
            H = np.eye(2, 3)
            
        # 保存当前帧
        self.prevFrame = frame.copy()
        return H

    def applyFile(self, raw_frame=None, detections=None):
        line = self.gmcFile.readline()
        tokens = line.split("\t")
        H = np.eye(2, 3, dtype=np.float_)
        if len(tokens) > 6:
            H[0, 0] = float(tokens[1])
            H[0, 1] = float(tokens[2])
            H[0, 2] = float(tokens[3])
            H[1, 0] = float(tokens[4])
            H[1, 1] = float(tokens[5])
            H[1, 2] = float(tokens[6])

        return H