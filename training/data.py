import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
import pickle
import csv

# from dataloaders.densely_captioned_images.impl import get_clip_ready_ds
# from dataloaders.densely_captioned_images.config import get_dci_config

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    from pcache_fileio import fileio
except ImportError:
    fileio = None
 
class _TorchSerializedList(object):
    """
    A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works.
    """

    def __init__(self, txt_file):

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        self._lst = [_serialize(x) for x in txt_file]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)

        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(self._lst))
        print(("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2)))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())

        # @lint-ignore PYTHONPICKLEISBAD
        return pickle.loads(bytes)

def read_csv(csv_files):
    img_list = []
    caption_list = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            for i, line in enumerate(csv_reader):
                if i == 0:
                    continue
                if line:
                    try:
                        caption_list.append(line[1])
                        img_list.append(line[0])
                    except:
                        print(line)

    img_list = _TorchSerializedList(img_list)
    caption_list = _TorchSerializedList(caption_list)
    return img_list, caption_list

def read_long_csv(csv_files, use_longcap = False):
    img_list = []
    dataset_name = []
    caption_raw_list = []
    caption_longSV_list = []
    caption_shortSV_list = []
    caption_longIB_list = []
    caption_shortIB_list = []
    caption_longLLA_list = []
    caption_shortLLA_list = []
    
    for csv_file in csv_files:

        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            for i, line in enumerate(csv_reader):
                if i == 0: continue
                img_list.append(line[0])

                caption_raw_list.append(line[1])
                caption_shortIB_list.append(line[2])
                caption_longIB_list.append(line[3])
                caption_shortSV_list.append(line[4])
                caption_longSV_list.append(line[5])
                caption_shortLLA_list.append(line[6])
                caption_longLLA_list.append(line[7])

                dataset_name.append(csv_file.split('/')[4])
            f.close()

    img_list = _TorchSerializedList(img_list)
    dataset_name = _TorchSerializedList(dataset_name)

    caption_raw_list = _TorchSerializedList(caption_raw_list)
    caption_shortIB_list = _TorchSerializedList(caption_shortIB_list)
    caption_longIB_list = _TorchSerializedList(caption_longIB_list)
    caption_shortSV_list = _TorchSerializedList(caption_shortSV_list)
    caption_longSV_list = _TorchSerializedList(caption_longSV_list)
    caption_shortLLA_list = _TorchSerializedList(caption_shortLLA_list)
    caption_longLLA_list = _TorchSerializedList(caption_longLLA_list)
    return img_list, caption_raw_list, caption_shortIB_list, caption_longIB_list, caption_shortSV_list, caption_longSV_list, caption_shortLLA_list, caption_longLLA_list, dataset_name

class CsvDataset(Dataset):
    def __init__(self,
                 input_filename,
                 transforms,
                 img_key,
                 caption_key,
                 sep="\t",
                 tokenizer=None,
                 root_filename='./datasets/',
                 meta_nouns=None,
                 use_longcap=False,
                 use_synimg=False,
                 num_text=1,
                 merged_num=1):
        self.root = root_filename

        self.use_longcap = use_longcap
        self.use_synimg = use_synimg

        logging.debug(f'Loading csv data from {input_filename}, use long caption: {self.use_longcap}.')
        if self.use_longcap:
            (images,
             caption_raw_list,
             caption_shortIB_list,
             caption_longIB_list,
             caption_shortSV_list,
             caption_longSV_list,
             caption_shortLLA_list,
             caption_longLLA_list,
             dataset_names) = read_long_csv(input_filename.split(','), self.use_longcap)
        else: 
            images, caption_raw_list = read_csv(input_filename.split(','))
            (caption_shortIB_list, 
             caption_longIB_list, 
             caption_shortSV_list, 
             caption_longSV_list, 
             caption_shortLLA_list, 
             caption_longLLA_list) = None, None, None, None, None, None
            dataset_names = None
        self.labels = None
        self.images = images
        self.caption_raw_list = caption_raw_list
        self.caption_shortIB_list = caption_shortIB_list
        self.caption_longIB_list = caption_longIB_list
        self.caption_shortSV_list = caption_shortSV_list
        self.caption_longSV_list = caption_longSV_list
        self.caption_shortLLA_list = caption_shortLLA_list
        self.caption_longLLA_list = caption_longLLA_list  
        self.merged_num = merged_num
        self.dataset_names = dataset_names

        self.numclasses = 10000
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        self.num_text = num_text
    def __len__(self):
        return len(self.caption_raw_list)
    
    def split_caption(self, text):
        texts = re.split(r'\n|</s>|[.]',text)
        subcap = []
        for text_prompt in texts:
            text_prompt = text_prompt.strip()
            if len(text_prompt) != 0:
                subcap.append(text_prompt)
        del texts
        return subcap
        
    def fetch_file(self, root_dir, filename):
        """Shortcut to reader's `fetch_file()`."""
        with open(os.path.join(root_dir, filename), 'rb') as f:
            try:
                return Image.open(f).convert('RGB')
            except:
                print(root_dir, filename)

    def draw_numbers(self, n, k=4):
        population = list(range(0, n))
        if n >= k: return random.sample(population, k)
        else: return random.choices(population, k=k)

    def __getitem__(self, idx):
        images = self.fetch_file(self.root, str(self.images[idx]))
        images = self.transforms(images)
        if self.use_synimg:
            images_2 = self.fetch_file(f'/input_ssd/datasets/{self.dataset_names[idx]}/rawcaption_turbo_4step/', str(self.images[idx]).replace(ori_paths[self.dataset_names[idx]],''))
            try:
                images_1 = self.fetch_file(f'/input_ssd/datasets/{self.dataset_names[idx]}/instuctblip_shortcaption_turbo_4step/', str(self.images[idx]).replace(ori_paths[self.dataset_names[idx]],''))
            except:
                print(str(self.images[idx]))
                images_1 = images_2
            images_1 = self.transforms(images_1)
            images_2 = self.transforms(images_2)
        else:
            images_1 = None
            images_2 = None

        if self.use_longcap:
            # self.caption_shortIB_list = caption_shortIB_list
            # self.caption_longIB_list = caption_longIB_list
            # self.caption_shortSV_list = caption_shortSV_list
            # self.caption_longSV_list = caption_longSV_list
            # self.caption_shortLLA_list = caption_shortLLA_list
            # self.caption_longLLA_list = caption_longLLA_list
            long_captions = self.split_caption(self.caption_shortIB_list[idx]) + \
                            self.split_caption(self.caption_longIB_list[idx]) + \
                            self.split_caption(self.caption_shortSV_list[idx]) + \
                            self.split_caption(self.caption_longSV_list[idx]) + \
                            self.split_caption(self.caption_shortLLA_list[idx]) + \
                            self.split_caption(self.caption_longLLA_list[idx])
                       

            if self.merged_num == 1:
                captions = [str(self.caption_raw_list[idx])] + [long_captions[num] for num in self.draw_numbers(len(long_captions)-1,self.num_text)]
            else:
                captions = [str(self.caption_raw_list[idx])]
                for num in self.draw_numbers(len(long_captions)-self.merged_num-1,self.num_text):
                    captions.append('. '.join(long_captions[num:num+self.merged_num]))


            texts = self.tokenize(captions)

            if self.use_synimg:
                return images, images_1, images_2, texts, 0
            else:
                return images, texts, 0

        else:
            texts = self.tokenize([str(self.caption_raw_list[idx])])[0]

            if self.labels:
                labels = sum(F.one_hot(torch.tensor(self.labels[self.images[idx]]),
                                    num_classes=self.numclasses))
                return images, texts, labels
            else:
                return images, texts, 0



def read_txt(txt_file, serialize=True):

    txt_list = []
    dataname_index = []
    index2name = []
    for index, txt_path in enumerate(txt_file):
        # txt_path = '/input_ssd/zkc/excel_caption_merge/cc3m_3long_3short_1raw_imagepath.txt'
        index2name.append(txt_path.split('/')[-1].split('_')[0])
        with open(txt_path, 'r') as f:
            line = f.readline() # skip the first row
            while line:
                line = f.readline().strip()
                if line:
                    txt_list.append(line)
                    dataname_index.append(index)
            f.close()
    
    if serialize:
        return _TorchSerializedList(txt_list),_TorchSerializedList(dataname_index),index2name
    return txt_list,dataname_index,index2name


class TxtDataset(Dataset):
    def __init__(self,
                 input_filename,
                 transforms,
                 img_key,
                 caption_key,
                 sep="\t",
                 tokenizer=None,
                 root_filename='./datasets/',
                 meta_nouns=None,
                 use_longcap=False,
                 use_synimg=False,
                 num_text=1,
                 merged_num=1,
                 json_root=None):
        self.root = root_filename
        self.json_root = json_root
        self.use_longcap = use_longcap
        self.use_synimg = use_synimg

        logging.debug(f'Loading csv data from {input_filename}, use long caption: {self.use_longcap}.')
        json_list,dataname_index,index2name = read_txt(input_filename.split(','), self.use_longcap)

        self.index2name = index2name
        self.dataname2prefix =  {
                                'cc3m':'bacba1e325b67e1aa48e7ff40f7e80bf/CC3M/images/',
                                'cc12m':'bacba1e325b67e1aa48e7ff40f7e80bf/CC12M/images',
                                'yfcc15m':'bacba1e325b67e1aa48e7ff40f7e80bf/YFCC15M/images/',
    													  'laion':'bacba1e325b67e1aa48e7ff40f7e80bf/Laion/laion20m/images/'
                            }
        self.dataname_index = dataname_index
        self.json_list = json_list

        self.merged_num = merged_num
        self.numclasses = 10000
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        self.num_text = num_text
    def __len__(self):
        return len(self.json_list)
    
    def split_caption(self, text):
        texts = re.split(r'\n|</s>|[.]',text)
        subcap = []
        for text_prompt in texts:
            text_prompt = text_prompt.strip()
            if len(text_prompt) != 0:
                subcap.append(text_prompt)
        del texts
        return subcap
        
    def fetch_file(self, root_dir, filename):
        """Shortcut to reader's `fetch_file()`."""
        with open(os.path.join(root_dir, filename), 'rb') as f:
            try:
                return Image.open(f).convert('RGB')
            except:
                print(root_dir, filename)

    def draw_numbers(self, n, k=4):
        population = list(range(0, n))
        if n >= k: return random.sample(population, k)
        else: return random.choices(population, k=k)

    def __getitem__(self, idx):
        dataname = self.index2name[self.dataname_index[idx]]
        imagepath_prefix = self.dataname2prefix[dataname]

        json_file = os.path.join(f'{self.json_root}/{dataname}/json_meta_{dataname}_3long_3short_1raw_captions',self.json_list[idx])
        
        with open(json_file, 'r') as file:
            json_data = json.load(file)
        
            image_path = os.path.join(imagepath_prefix, json_data['ImagePath'])

            images = self.fetch_file(self.root, image_path)
            images = self.transforms(images)
            images_1 = None
            images_2 = None

            if self.use_longcap:
                long_captions = self.split_caption(json_data["shortIB_captions"]) + \
                                self.split_caption(json_data["longIB_captions"]) + \
                                self.split_caption(json_data["shortLLA_captions"]) + \
                                self.split_caption(json_data["longLLA_captions"]) + \
                                self.split_caption(json_data["shortSV_captions"]) + \
                                self.split_caption(json_data["longSV_captions"])
                        

                if self.merged_num == 1:
                    captions = [json_data['raw_caption']] \
                        + [long_captions[num] for num in self.draw_numbers(len(long_captions)-1,self.num_text)]
                else:
                    captions = [json_data['raw_caption']] 
                    for num in self.draw_numbers(len(long_captions)-self.merged_num-1,self.num_text):
                        captions.append('. '.join(long_captions[num:num+self.merged_num]))


                texts = self.tokenize(captions)

                if self.use_synimg:
                    return images, images_1, images_2, texts, 0
                else:
                    return images, texts, 0

            else:
                texts = self.tokenize([str(json_data['raw_caption'])])[0]

                return images, texts, 0


class JsonDataset(Dataset):
    def __init__(self,
                 input_filename,
                 transforms,
                 img_key,
                 caption_key,
                 sep="\t",
                 tokenizer=None,
                 root_filename='./datasets/',
                 meta_nouns=None,
                 use_longcap=False,
                 use_synimg=False,
                 num_text=1,
                 merged_num=1,
                 split_json_size=1):
        self.root = root_filename

        self.use_longcap = use_longcap
        self.use_synimg = use_synimg
        logging.debug(f'Loading json data from {input_filename}, use long caption: {self.use_longcap}.')
        self.json_list = read_txt(input_filename.split(','))
        self.merged_num = merged_num
        self.num_text = num_text
        seed=1234
        self.json_length = len(self.json_list)
        self.rng = np.random.default_rng(seed)
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        self.split_json_size = split_json_size

    def __len__(self):
        return self.json_length * 1024 // self.split_json_size
    
    def split_caption(self, text):
        texts = re.split(r'\n|</s>|[.]',text)
        subcap = []
        for text_prompt in texts:
            text_prompt = text_prompt.strip()
            if len(text_prompt) != 0:
                subcap.append(text_prompt)
        del texts
        return subcap
        
    def fetch_file(self, root_dir, filename):
        """Shortcut to reader's `fetch_file()`."""
        with open(os.path.join(root_dir, filename), 'rb') as f:
            try:
                return Image.open(f).convert('RGB')
            except:
                print(root_dir, filename)

    def draw_numbers(self, n, k=4):
        population = list(range(0, n))
        if n >= k: return random.sample(population, k)
        else: return random.choices(population, k=k)

    def __getitem__(self, idx):

        # real_idx = idx * self.split_json_size //1024
        json_file = self.json_list[idx % self.json_length]
        images_list = []
        texts_list = []
        with open(json_file, 'r') as file:
            data = json.load(file)
            # for meta_info_index in self.draw_numbers(len(data), self.split_json_size):
            for meta_info_index in self.rng.choice(len(data), self.split_json_size, replace=False):
                meta_info = data[meta_info_index]
                image_path = meta_info['ImagePath']
                images = self.fetch_file(self.root, image_path)
                images = self.transforms(images)

                if self.use_longcap:
                    long_captions = self.split_caption(meta_info["IBCaptionShort"]) + \
                                    self.split_caption(meta_info["IBCaptionLong"]) + \
                                    self.split_caption(meta_info["LLACaptionShort"]) + \
                                    self.split_caption(meta_info["LLACaptionLong"]) + \
                                    self.split_caption(meta_info["SVCaptionShort"]) + \
                                    self.split_caption(meta_info["SVCaptionLong"])
                    if self.merged_num == 1:
                        captions = [meta_info['RawCaption']] + \
                                [long_captions[num] for num in self.draw_numbers(len(long_captions)-1,self.num_text)]
                    else:
                        captions = [meta_info['RawCaption']]
                        for num in self.draw_numbers(len(long_captions)-self.merged_num-1,self.num_text):
                            captions.append('. '.join(long_captions[num:num+self.merged_num]))
                    texts = self.tokenize(captions)
                else:
                    texts = self.tokenize([str(meta_info['RawCaption'])])[0]
                images_list.append(images)
                texts_list.append(texts)
                

        return torch.stack(images_list,0), torch.stack(texts_list,0), 0

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def get_dci(args, preprocess_fns, subtest, tokenizer):
    subtests = {
        "all_swaps": {
            "load_base_image": True,
            "load_subcaptions": True,
            "negative_source":  'swaps',
            "negative_strategy": "first",
            "caption_bag_size": 0,
        },
        "all_swaps_pick5": {
            "load_base_image": True,
            "load_subcaptions": True,
            "negative_source":  'swaps',
            "negative_strategy": "first",
            "caption_bag_size": 5,
        },
        "base_swaps": {
            "load_base_image": True,
            "load_subcaptions": False,
            "negative_source":  'swaps',
            "negative_strategy": "first",
            "caption_bag_size": 0,
        },
        "all_hardest": {
            "load_base_image": True,
            "load_subcaptions": True,
            "negative_source":  'any',
            "negative_strategy": "hardest",
            "caption_bag_size": 0,
        },
    }
    _, preprocess_val = preprocess_fns
    dataset = get_clip_ready_ds(split='test', **subtests[subtest])
    dataset.processor = preprocess_val
    dataset.tokenizer = tokenizer

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=args.workers,
        shuffle=False,
        sampler=None
    )

    return DataInfo(dataloader=dataloader, sampler=None)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_json_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        root_filename=args.root_filename,
        meta_nouns=args.meta_nouns,
        use_longcap=args.use_longcap,
        use_synimg=args.use_synimg,
        num_text=args.num_text,
        merged_num=args.merged_num,
        split_json_size=args.split_json_size
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size//args.split_json_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_txt_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = TxtDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        root_filename=args.root_filename,
        meta_nouns=args.meta_nouns,
        use_longcap=args.use_longcap,
        use_synimg=args.use_synimg,
        num_text=args.num_text,
        merged_num=args.merged_num,
        json_root=args.json_root
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        root_filename=args.root_filename,
        meta_nouns=args.meta_nouns,
        use_longcap=args.use_longcap,
        use_synimg=args.use_synimg,
        num_text=args.num_text,
        merged_num=args.merged_num
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "txt":
        return get_txt_dataset
    elif dataset_type == "json":
        return get_json_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    if args.dci is not None:
        get_dci_config(args.dci)
        for key in ['all_swaps', 'all_swaps_pick5', 'base_swaps', 'all_hardest']:
            data["dci_"+key] = get_dci(args, preprocess_fns, key, tokenizer)

    return data
