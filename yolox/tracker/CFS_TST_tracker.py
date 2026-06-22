#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from collections import deque

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .gmc import GMC
from . import matching

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls=0, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        if feat is not None:
            self.update_features(feat)
        self.alpha = 0.9

        self.birth_count = 0 

        self.birth_threshold = None  

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        
        self.tracklet_len = 0
        if frame_id == 1:  # 第一帧特殊处理
            self.state = TrackState.Tracked
            self.is_activated = True
        else:
            if self.birth_threshold <= 0:  
                self.state = TrackState.Tracked
                self.is_activated = True
            else:
                self.state = TrackState.Tentative  
                self.is_activated = False
        
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.birth_count = 1

    def re_activate(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

        if self.state == TrackState.Tracked:
            self.is_activated = True
        else:
            self.state = TrackState.Tentative
            self.birth_count = 1  

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        #self.state = TrackState.Tracked
        #self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

        # 状态更新逻辑
        if self.state == TrackState.Tentative:
            self.birth_count += 1
            if self.birth_count >= self.birth_threshold:
                self.state = TrackState.Tracked
                self.is_activated = True
                self.birth_count = 0
        
        # 已确认的轨迹保持Tracked状态
        elif self.state == TrackState.Tracked:
            self.is_activated = True

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        # ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class DefaultArgs(object):
    def __init__(self, mot20=False):
        self.track_thresh = 0.6
        self.low_thresh = 0.1
        self.track_buffer = 30
        self.match_thresh = 0.8

        self.birth_threshold = 1 

        self.mot20 = mot20
        self.mask_emb_with_iou = True
        self.fuse_emb_and_iou = 'min'

        self.cmc_method = 'none'
        self.cmc_seq_name = ''
        self.cmc_file_dir = ''


class CFS_TST_Tracker(object):
    def __init__(self, args, frame_rate=30):

        # 修改状态管理
        self.tracked_stracks = []  # 保留此属性但仅用于存储所有跟踪状态的轨迹
        self.confirmed_tracks = []  # 确认的轨迹 
        self.tentative_tracks = [] # 待确认的轨迹
        self.lost_stracks = []     # 丢失的轨迹
        self.removed_stracks = []  # 移除的轨迹

        self.birth_threshold = args.birth_threshold


        # self.tracked_stracks = []  # type: list[STrack]
        # self.lost_stracks = []  # type: list[STrack]
        # self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        # self.args = args
        self.mot20 = args.mot20

        # self.det_thresh = args.track_thresh + 0.1

        self.track_high_thresh = args.track_thresh
        self.track_low_thresh = args.low_thresh
        self.new_track_thresh = args.track_thresh + 0.1
        self.match_thresh = args.match_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size    
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.iou_only = False
        self.mask_emb_with_iou = args.mask_emb_with_iou  # default True
        self.fuse_emb_and_iou = args.fuse_emb_and_iou  # default min, choice: [min, mean]
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25

        self.gmc = GMC(method=args.cmc_method, 
                       verbose=[args.cmc_seq_name, args.cmc_file_dir])

        self.super_cls = []
        # self.super_cls = np.array([0, 0, 1, 1, 1, 2, 3, 3])  # for bdd100k

    def _fuse_cost_matrix(self, iou_cost, emb_cost):
        if self.fuse_emb_and_iou == 'min':  # default min
            cost = np.minimum(iou_cost, emb_cost)
        elif self.fuse_emb_and_iou == 'mean':
            cost = (iou_cost + emb_cost) / 2.
        else:
            raise ValueError('Invalid fusion mode:', self.fuse_emb_and_iou)

        return cost

    @staticmethod
    def _risk(x, i, j):
        x = x.clip(0., 1.)
        pos = x[i, j].copy()
        neg = x[i, :].copy()
        neg[np.arange(len(i)), j] = 0.
        
        eps = 1e-6
        j2 = np.argsort(-neg, axis=1)[:, 0]  # 2nd maximal similarity
        i2 = np.arange(len(j2))
        neg2 = neg[i2, j2].copy()

        x = -np.log(pos + eps) - np.log(1. - neg2 + eps)

        return x

    @staticmethod
    def _threshold(x, m1=0.5, m2=0.05):
        eps = 1e-6
        border = -np.log(1. + m2 - x + eps) - np.log(m1)
        border[x > 0.98] = 1e4
        return border

    def associate(self, trks, dets, thresh, fuse_score=False, iou_only=False):
        iou_dists = matching.iou_distance(trks, dets)

        if fuse_score:
            iou_dists = matching.fuse_score(iou_dists, dets)

        dists = iou_dists
        if not iou_only:
            # Popular ReID method (JDE / FairMOT)
            emb_dists = matching.embedding_distance(trks, dets)

            if len(self.super_cls) > 1:
                emb_dists = matching.gate_cost_matrix_by_cls(emb_dists, trks, dets, self.super_cls)

            # dists = matching.fuse_motion(self.kalman_filter, emb_dists, trks, dets)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            if self.mask_emb_with_iou:
                emb_dists[iou_dists > self.proximity_thresh] = 1.0  # TODO: iou mask or not
            dists = self._fuse_cost_matrix(dists, emb_dists)
        #使用了匈牙利算法进行匹配
        return matching.linear_assignment(dists, thresh=thresh)

    def update(self, output_results, img_info, img_size, embeddings=None, img=None):
    
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 处理检测结果
        if len(output_results):
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, 6]
            
            # 尺度变换
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            bboxes /= scale

            # 分离高低置信度检测
            high_mask = scores > self.track_high_thresh
            low_mask = np.logical_and(scores > self.track_low_thresh, 
                                    scores <= self.track_high_thresh)
            
            # 高置信度检测
            high_dets = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                        (tlbr, s, c, f) in zip(bboxes[high_mask], 
                                            scores[high_mask],
                                            classes[high_mask], 
                                            embeddings[high_mask])]
            
            # 低置信度检测
            low_dets = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                    (tlbr, s, c, f) in zip(bboxes[low_mask],
                                            scores[low_mask], 
                                            classes[low_mask],
                                            embeddings[low_mask])]
        else:
            high_dets = []
            low_dets = []

        """ 第一阶段: 高置信度检测关联 """
        # 分离不同状态的轨迹
        confirmed_tracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        tentative_tracks = [t for t in self.tracked_stracks if t.state == TrackState.Tentative]
        
        # 预测轨迹位置
        STrack.multi_predict(confirmed_tracks)
        if self.gmc is not None:
            warp = self.gmc.apply(img, bboxes[high_mask] if len(output_results) else None)
            STrack.multi_gmc(confirmed_tracks, warp)
        
        # 关联高置信度检测
        matches_high, u_tracks_high, u_dets_high = self.associate(
            confirmed_tracks, high_dets,
            thresh=self.match_thresh,
            fuse_score=not self.mot20
        )
        
        # 更新匹配轨迹
        for itrk, idet in matches_high:
            track = confirmed_tracks[itrk]
            det = high_dets[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        """ 第二阶段: 低置信度检测关联 """
        # 获取未匹配轨迹
        unmatched_tracks = [confirmed_tracks[i] for i in u_tracks_high]
        
        # 关联低置信度检测
        matches_low, u_tracks_low, u_dets_low = self.associate(
            unmatched_tracks, low_dets,
            thresh=0.5,
            fuse_score=False,
            iou_only=True
        )
        
        # 更新匹配轨迹
        for itrk, idet in matches_low:
            track = unmatched_tracks[itrk]
            det = low_dets[idet]
            if track.state == TrackState.Lost:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            else:
                track.update(det, self.frame_id)
                activated_starcks.append(track)

        """ 处理未匹配轨迹 """
        for it in u_tracks_low:
            track = unmatched_tracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # """ 处理待确认轨迹 """
        # 关联待确认轨迹与未匹配的高置信度检测
        if len(tentative_tracks) > 0 and len(u_dets_high) > 0:
            # 获取未匹配的高置信度检测
            unmatched_high_dets = [high_dets[i] for i in u_dets_high]
            
            matches_tent, u_tent, u_dets = self.associate(
                tentative_tracks, 
                unmatched_high_dets,  # 使用未匹配的检测  
                thresh=0.7,  # 可以根据BIRTH_THRESHOLD调整这个阈值
                fuse_score=not self.mot20
            )
            
            # 更新匹配上的待确认轨迹
            for itrk, idet in matches_tent:
                track = tentative_tracks[itrk]
                det = unmatched_high_dets[idet]  # 直接使用未匹配检测列表
                track.update(det, self.frame_id)
                activated_starcks.append(track)
                
            # 移除未匹配的待确认轨迹
            for it in u_tent:
                track = tentative_tracks[it]
                track.mark_removed()
                removed_stracks.append(track)
                
            # 更新未匹配检测索引
            u_dets_high = [u_dets_high[i] for i in u_dets]

        """ 初始化新轨迹 """
        # 处理剩余的高置信度检测
        for idet in u_dets_high:
            det = high_dets[idet]
            if det.score < self.new_track_thresh:
                continue
            track = STrack(det.tlwh, det.score, det.cls, det.curr_feat)

            track.birth_threshold = self.birth_threshold  # 传入birth_threshold参数

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ 更新轨迹状态 """
        # 移除长期丢失轨迹
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                
        # 更新轨迹列表
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state != TrackState.Removed]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        # 移除重复轨迹
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        tracked_stracks = []
        for t in self.tracked_stracks:
            if t.state == TrackState.Tracked or t.state == TrackState.Tentative:
                tracked_stracks.append(t)
                
        return tracked_stracks  # 返回所有活跃轨迹
        

        #return [t for t in self.tracked_stracks if t.state == TrackState.Tracked]

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb