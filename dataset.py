# -*- coding: utf-8 -*-
import os
import json

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from utils import ioa_with_anchors, iou_with_anchors


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class Collator(object):
    def __init__(self, cfg, mode):
        self.is_train = mode in ['train', 'training']
        if self.is_train:
            self.batch_names = ['env_feats', 'agent_feats', 'box_lens', 'obj_feats', 'obj_box_lens', 'conf_labels', 'start_labels', 'end_labels']
            self.label_names = ['conf_labels', 'start_labels', 'end_labels']
        else:
            self.batch_names = ['video_ids', 'env_feats', 'agent_feats', 'box_lens', 'obj_feats', 'obj_box_lens']
            self.label_names = []
        self.feat_names = ['env_feats', 'agent_feats', 'box_lens', 'obj_feats', 'obj_box_lens']
        self.tmp_dim = cfg.DATA.TEMPORAL_DIM
        self.feat_dim = cfg.MODEL.AGENT_DIM
        self.obj_feat_dim = cfg.MODEL.OBJ_DIM   ####

    def process_features(self, bsz, env_feats, agent_feats, box_lens, obj_feats, obj_box_lens):
        if env_feats[0] is not None:
            env_feats = torch.stack(env_feats)
        else:
            env_feats = None

        # Make new order to inputs by their lengths (long-to-short)
        if agent_feats[0] is not None:
            box_lens = torch.stack(box_lens, dim=0)

            max_box_dim = torch.max(box_lens).item()
            # Make padding mask for self-attention
            agent_mask = torch.arange(max_box_dim)[None, None, :] < box_lens[:, :, None]

            # Pad agent features at temporal and box dimension
            pad_agent_feats = torch.zeros(bsz, self.tmp_dim, max_box_dim, self.feat_dim)
            for i, temporal_features in enumerate(agent_feats):
                for j, box_features in enumerate(temporal_features):
                    if len(box_features) > 0:
                        pad_agent_feats[i, j, :len(box_features)] = torch.tensor(box_features)
        else:
            pad_agent_feats = None
            agent_mask = None
        
        # Make new order to inputs by their lengths (long-to-short)
        if obj_feats[0] is not None:
            obj_box_lens = torch.stack(obj_box_lens, dim=0)

            max_box_dim = torch.max(obj_box_lens).item()
            # Make padding mask for self-attention
            obj_mask = torch.arange(max_box_dim)[None, None, :] < obj_box_lens[:, :, None]

            # Pad agent features at temporal and box dimension
            pad_obj_feats = torch.zeros(bsz, self.tmp_dim, max_box_dim, self.obj_feat_dim)
            for i, temporal_features in enumerate(obj_feats):
                for j, box_features in enumerate(temporal_features):
                    if len(box_features) > 0:
                        pad_obj_feats[i, j, :len(box_features)] = torch.tensor(box_features)
        else:
            pad_obj_feats = None
            obj_mask = None
        return env_feats, pad_agent_feats, agent_mask, pad_obj_feats, obj_mask

    def __call__(self, batch):
        input_batch = dict(zip(self.batch_names, zip(*batch)))
        bsz = len(input_batch['env_feats'])
        output_batch = [] if self.is_train else [input_batch['video_ids']]

        # Process environment and agent features
        input_feats = [input_batch[feat_name] for feat_name in self.feat_names]
        output_batch.extend(self.process_features(bsz, *input_feats))

        for label_name in self.label_names:
            output_batch.append(torch.stack(input_batch[label_name]))
        return output_batch


class VideoDataSet(Dataset):
    def __init__(self, cfg, split='training'):
        self.split = split
        self.dataset_name = cfg.DATASET
        self.video_anno_path = cfg.DATA.ANNOTATION_FILE
        self.temporal_dim = cfg.DATA.TEMPORAL_DIM
        self.max_duration = cfg.DATA.MAX_DURATION
        self.temporal_gap = 1. / self.temporal_dim
        self.env_feature_dir = cfg.DATA.ENV_FEATURE_DIR
        self.agent_feature_dir = cfg.DATA.AGENT_FEATURE_DIR
        self.obj_feature_dir = cfg.DATA.OBJ_FEATURE_DIR

        self.use_env = cfg.USE_ENV
        self.use_agent = cfg.USE_AGENT
        self.use_obj = cfg.USE_OBJ

        if split in ['train', 'training']:
            self._get_match_map()

        self.video_prefix = 'v_' if cfg.DATASET == 'anet' else ''

        self._get_dataset()

    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_dim):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.max_duration + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map

        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_dim)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(1, self.temporal_dim + 1)]
        # self.anchor_xmin = [self.temporal_gap * i for i in range(self.temporal_dim)]
        # self.anchor_xmax = [self.temporal_gap * i for i in range(1, self.temporal_dim + 1)]

    def get_filter_video_names(self, json_data, upper_thresh=.98, lower_thresh=.3):
        """
        Select video according to length of ground truth
        :param video_info_file: json file path of video information
        :param gt_len_thres: max length of ground truth
        :return: list of video names
        """
        filter_video_names, augment_video_names = [], []
        video_lists = list(json_data)
        for video_name in video_lists:
        # for video_name in video_lists[::-1]:
            video_info = json_data[video_name]
            if not os.path.isfile(os.path.join(self.env_feature_dir, 'v_' + video_name + '.json')):
                filter_video_names.append(video_name)
                continue
            if video_info['subset'] != "training":
                continue
            video_second = video_info["duration"]
            gt_lens = []
            video_labels = video_info["annotations"]
            for j in range(len(video_labels)):
                tmp_info = video_labels[j]
                tmp_start = tmp_info["segment"][0]
                tmp_end = tmp_info["segment"][1]
                tmp_start = max(min(1, tmp_start / video_second), 0)
                tmp_end = max(min(1, tmp_end / video_second), 0)
                gt_lens.append(tmp_end - tmp_start)
            if len(gt_lens):
                mean_len = np.mean(gt_lens)
                if mean_len >= upper_thresh:
                    filter_video_names.append(video_name)
                if mean_len < lower_thresh:
                    augment_video_names.append(video_name)
        return filter_video_names, augment_video_names

    def _get_dataset(self):
        annotations = load_json(self.video_anno_path)['database']
        if self.dataset_name == 'anet':
            filter_video_names, augment_video_names = self.get_filter_video_names(annotations)
        else:
            filter_video_names, augment_video_names = [], []

        # Read event segments
        self.event_dict = {}
        self.video_ids = []

        for video_id, annotation in annotations.items():
            if annotation['subset'] != self.split or video_id in filter_video_names:
                continue
            self.event_dict[video_id] = {
                'duration': annotation['duration'],
                'events': annotation['annotations']
                # 'events': annotation['timestamps']
            }
            self.video_ids.append(video_id)
        if self.split in ['train', 'training']:
            self.video_ids.extend(augment_video_names)

        print("Split: %s. Dataset size: %d" % (self.split, len(self.video_ids)))

    def __getitem__(self, index):
        env_features, agent_features, box_lengths, obj_features, obj_box_lengths = self._load_item(index)
        if self.split == 'training':
            match_score_start, match_score_end, confidence_score = self._get_train_label(index)
            return env_features, agent_features, box_lengths, obj_features, obj_box_lengths, confidence_score, match_score_start, match_score_end
        else:
            return self.video_ids[index], env_features, agent_features, box_lengths, obj_features, obj_box_lengths

    def _load_item(self, index):
        video_name = self.video_prefix + self.video_ids[index]

        '''
        Read environment features at every timestamp
        Feature size: TxF
        T: number of timestamps
        F: feature size
        '''
        if self.use_env is True:
            env_features = load_json(os.path.join(self.env_feature_dir, video_name + '.json'))['video_features']
            # env_segments = [env['segment'] for env in env_features]
            env_features = torch.tensor([feature['features'] for feature in env_features]).float().squeeze(1)
        else:
            env_features = None

        '''
        Read agents features at every timestamp
        Feature size: TxBxF
        T: number of timestamps
        B: max number of bounding boxes
        F: feature size
        '''
        if self.use_agent is True:
            agent_features = load_json(os.path.join(self.agent_feature_dir, video_name + '.json'))['video_features']
            # agent_segments = [feature['segment'] for feature in agent_features]
            agent_features = [feature['features'] for feature in agent_features]
            # Create and pad agent_box_lengths if train
            box_lengths = torch.tensor([len(x) for x in agent_features])
        else:
            agent_features = None
            box_lengths = None

        '''
        Read agents features at every timestamp
        Feature size: TxBxF
        T: number of timestamps
        B: max number of bounding boxes
        F: feature size
        '''
        if self.use_obj is True:
            try:
                obj_features = load_json(os.path.join(self.obj_feature_dir, video_name + '.json'))['video_features']
            except:
                print('error', video_name)
                pass
            # agent_segments = [feature['segment'] for feature in agent_features]
            obj_features = [feature['features'] for feature in obj_features]
            # Create and pad agent_box_lengths if train
            obj_box_lengths = torch.tensor([len(x) for x in obj_features])
        else:
            obj_features = None
            obj_box_lengths = None
        # assert env_segments == agent_segments and len(env_segments) == 100, 'Two streams must have 100 segments.'

        return env_features, agent_features, box_lengths, obj_features, obj_box_lengths

    def _get_train_label(self, index):
        video_id = self.video_ids[index]
        video_info = self.event_dict[video_id]
        video_labels = video_info['events']  # the measurement is second, not frame
        duration = video_info['duration']

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / duration), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / duration), 0)
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.max_duration, self.temporal_dim])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)
        ##############################################################################################

        ##############################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        ##############################################################################################

        ##############################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.tensor(match_score_start)
        match_score_end = torch.tensor(match_score_end)
        ##############################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_ids)
