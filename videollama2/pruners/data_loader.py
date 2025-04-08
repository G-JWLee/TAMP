import re
import os
import copy
import json
import random
import pathlib
import traceback
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import Dataset, DataLoader

from videollama2.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP
from videollama2.mm_utils import tokenizer_multimodal_token, process_video, process_image, process_audio_file
from videollama2.train import *

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from PIL import Image
import math


# Pretrain or Finetune dataclass
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, processor, data_args):
        super(LazySupervisedDataset, self).__init__()
        self.mix_sampler_tag = False
        if data_path is not None and len(data_path.split(",")) == 1:
            data_path = data_path.split(",")[0]
            list_data_dict = json.load(open(data_path, "r"))
        elif data_path is not None and len(data_path.split(",")) > 1:
            self.mix_sampler_tag = True
            data_path = data_path.split(",")
            for path in data_path:
                if "stage3" in path:
                    self.av_data = json.load(open(path, "r"))
                    random.shuffle(self.av_data)
                elif "stage2" in path and "audio" in path:
                    self.a_data = json.load(open(path, "r"))
                    random.shuffle(self.a_data)
                elif "stage2" in path and "video" in path:
                    self.v_data = json.load(open(path, "r"))
                    random.shuffle(self.v_data)
                else:
                    raise NotImplementedError
            list_data_dict = self.av_data + self.a_data + self.v_data

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        # Choose a task of interest and We randomly select data
        if data_args.sample_select == 'random':
            sampled_indices = np.random.choice(np.arange(0, len(self.list_data_dict)), size=data_args.nsamples, replace=False)
        elif data_args.sample_select == 'longest':
            lang_length_list = []
            for i in range(0, len(self.list_data_dict)):
                total_lang_len = np.array([len(self.list_data_dict[i]['conversations'][j]['value']) for j in range(len(self.list_data_dict[i]['conversations']))])
                lang_length_list.append(total_lang_len.sum())
            sampled_indices = np.argsort(lang_length_list)[-data_args.nsamples:]
            sampled_indices += 0

        self.list_data_dict = [self.list_data_dict[idx] for idx in sampled_indices]
        self.video_processor = processor['video']

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            video_folder = self.data_args.data_folder
            if video_folder:
                video_file = os.path.join(video_folder, video_file)

            try:
                video = self.video_processor(video_file, va = self.data_args.va if not self.mix_sampler_tag else (i < len(self.av_data)))
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)

            # place <video> tag to question head.
            modal_token = "<video>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)

        elif 'audio' in sources[0]:
            audio_file = self.list_data_dict[i]['audio']
            try:
                audio = process_audio_file(audio_file)
            except Exception as e:
                print(e)
                backup_idx = random.randint(0, len(self.list_data_dict)-1)
                print(f"Encounted error when reading audio {audio_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)
            modal_token = "<audio>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)

        else:
            modal_token = None
            sources = copy.deepcopy([e["conversations"] for e in sources])

        if self.data_args.is_pretraining:
            data_dict = preprocess_plain(sources, self.tokenizer, modal_token=modal_token)
        else:
            data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if 'video' in self.list_data_dict[i]:
            data_dict['video'] = video
        elif 'audio' in self.list_data_dict[i]:
            data_dict['audio'] = audio
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            data_dict['image'] = torch.zeros(3, self.data_args.image_size, self.data_args.image_size)
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))

        return batch

# DataLoader
def create_data_loader(tokenizer, processor, data_args, num_workers=4):
        
    dataset = LazySupervisedDataset(data_path=data_args.data_path, tokenizer=tokenizer, 
                                    processor=processor, data_args=data_args)
    collate_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)

    return data_loader