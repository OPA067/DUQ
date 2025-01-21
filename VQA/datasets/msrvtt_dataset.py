import os
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.basic_utils import load_json, load_jsonl
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class MSRVTTDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        dir = 'MSRVTT/QA'
        self.train_jsonl = load_jsonl(dir + '/train.jsonl')
        self.test_jsonl = load_jsonl(dir + '/test.jsonl')
        self.val_jsonl = load_jsonl(dir + '/val.jsonl')

        self.ans2label = load_json(dir + '/train_ans2label.json')
        self.num_labels = config.num_labels

        if split_type == 'train':
            self._construct_all_train_pairs()
        else:
            self._construct_all_test_pairs()

    def __getitem__(self, index):

        if self.split_type == 'train':
            video_path, question, answer = self._get_vidpath_and_caption_by_index(index)
            label = self.ans2label[answer] if answer in self.ans2label else -1
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'video_id': video_path,
                'video': imgs,
                'text': question,
                'label': label,
            }
        else:
            video_path, question, answer = self._get_vidpath_and_caption_by_index(index)
            label = self.ans2label[answer] if answer in self.ans2label else -1
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'video_id': video_path,
                'video': imgs,
                'text': question,
                'label': label,
            }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            video_id, question, answer = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, video_id + '.mp4')
            return video_path, question, answer
        else:
            video_id, question, answer = self.all_test_pairs[index]
            video_path = os.path.join(self.videos_dir, video_id + '.mp4')
            return video_path, question, answer
    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = {}
        for itm in self.train_jsonl:
            if itm['answer'] in self.ans2label:
                self.all_train_pairs[len(self.all_train_pairs)] = (itm['video_id'], itm['question'], itm['answer'])
        print("train len is", len(self.all_train_pairs))

    def _construct_all_test_pairs(self):
        self.all_test_pairs = {}
        for itm in self.test_jsonl:
            if itm['answer'] in self.ans2label:
                self.all_test_pairs[len(self.all_test_pairs)] = (itm['video_id'], itm['question'], itm['answer'])
        print("test len is", len(self.all_test_pairs))