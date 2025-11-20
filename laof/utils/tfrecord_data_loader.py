import tensorflow as tf
import torch
import numpy as np
from pathlib import Path
from typing import List
import doy
from tensordict import TensorDict

def parse_tf_example(example_proto):
    """解析TFRecord中的Example"""
    feature_description = {
        'obs': tf.io.FixedLenFeature([], tf.string),
        'ta': tf.io.FixedLenFeature([], tf.string),
        'done': tf.io.FixedLenFeature([], tf.string),
        'rewards': tf.io.FixedLenFeature([], tf.string),
        'ep_returns': tf.io.FixedLenFeature([], tf.string),
        'obs_of': tf.io.FixedLenFeature([], tf.string),
        'obs_sam': tf.io.FixedLenFeature([], tf.string),
        'mask_nums': tf.io.FixedLenFeature([], tf.string),  # 添加mask_nums字段
    }
    
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    # 反序列化tensor
    parsed_example = {}
    parsed_example['obs'] = tf.io.parse_tensor(example['obs'], out_type=tf.uint8)
    parsed_example['ta'] = tf.io.parse_tensor(example['ta'], out_type=tf.int64)
    parsed_example['done'] = tf.io.parse_tensor(example['done'], out_type=tf.bool)
    parsed_example['rewards'] = tf.io.parse_tensor(example['rewards'], out_type=tf.float32)
    parsed_example['ep_returns'] = tf.io.parse_tensor(example['ep_returns'], out_type=tf.float32)
    parsed_example['obs_of'] = tf.io.parse_tensor(example['obs_of'], out_type=tf.uint8)
    parsed_example['obs_sam'] = tf.io.parse_tensor(example['obs_sam'], out_type=tf.uint8)
    parsed_example['mask_nums'] = tf.io.parse_tensor(example['mask_nums'], out_type=tf.int8)  # 解析mask_nums
    
    return parsed_example

def tf_to_torch_batch_func(tf_batch):
    """将TensorFlow batch转换为PyTorch TensorDict格式的独立函数"""
    torch_batch = {}
    
    for key, value in tf_batch.items():
        # 转换为numpy然后到torch
        numpy_value = value.numpy()
        torch_value = torch.from_numpy(numpy_value)
        
        # 数据类型转换和归一化
        if key in ['obs', 'obs_of', 'obs_sam']:
            torch_value = torch_value.float() / 255.0 - 0.5  # 归一化到[-0.5, 0.5]
        elif key == 'ta':
            torch_value = torch_value.long()
        elif key == 'done':
            torch_value = torch_value.bool()
        elif key in ['rewards', 'ep_returns']:
            torch_value = torch_value.float()
        elif key == 'mask_nums':
            torch_value = torch_value.to(torch.int8)  # 保持int8类型
        
        torch_batch[key] = torch_value
    
    # 转换为TensorDict格式
    batch_size = torch_batch['obs'].shape[0]
    tensor_dict = TensorDict(torch_batch, batch_size=torch.Size([batch_size]))
    
    return tensor_dict

def get_tfrecord_files(data_root: str, env_name: str, split: str = 'train') -> List[Path]:
    """获取指定split的所有TFRecord文件路径"""
    data_root = Path(data_root)
    tfrecord_dir = data_root / "expert_data" / "opticalflow_rlds" / env_name
    
    if not tfrecord_dir.exists():
        raise FileNotFoundError(f"TFRecord directory not found: {tfrecord_dir}")
    
    if split == 'train':
        # 查找train的分片文件
        pattern = f"{split}-*-of-*.tfrecord"
        files = sorted(list(tfrecord_dir.glob(pattern)))
        if not files:
            # 如果没找到分片文件，尝试单文件
            single_file = tfrecord_dir / f"{split}.tfrecord"
            if single_file.exists():
                files = [single_file]
    else:
        # test数据通常是单文件
        files = [tfrecord_dir / f"{split}.tfrecord"]
        files = [f for f in files if f.exists()]
    
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for {split} split in {tfrecord_dir}")
    
    return files

def get_tfrecord_files_with_ratio(data_root: str, env_name: str, split: str = 'train', ratio: float = 1.0, use_remaining: bool = False) -> List[Path]:
    """获取指定split的TFRecord文件路径，支持按比例分割"""
    data_root = Path(data_root)
    tfrecord_dir = data_root / "expert_data" / "opticalflow_rlds" / env_name
    
    if not tfrecord_dir.exists():
        raise FileNotFoundError(f"TFRecord directory not found: {tfrecord_dir}")
    
    if split == 'train':
        # 查找train的分片文件
        pattern = f"{split}-*-of-*.tfrecord"
        all_files = sorted(list(tfrecord_dir.glob(pattern)))
        
        if not all_files:
            # 如果没找到分片文件，尝试单文件
            single_file = tfrecord_dir / f"{split}.tfrecord"
            if single_file.exists():
                all_files = [single_file]
        
        if not all_files:
            raise FileNotFoundError(f"No TFRecord files found for {split} split in {tfrecord_dir}")
        
        # 根据比例分割文件
        total_files = len(all_files)
        split_point = int(total_files * ratio)
        
        if use_remaining:
            # 返回剩余的文件 (从split_point到末尾)
            files = all_files[split_point:]
        else:
            # 返回前ratio部分的文件 (从0到split_point)
            files = all_files[:split_point]
            
        return files
    else:
        # test数据通常是单文件
        files = [tfrecord_dir / f"{split}.tfrecord"]
        files = [f for f in files if f.exists()]
        
        if not files:
            raise FileNotFoundError(f"No TFRecord files found for {split} split in {tfrecord_dir}")
        
        return files

class TFRecordDataIterator:
    """TFRecord数据迭代器"""
    
    def __init__(self, tfrecord_paths: List[Path], batch_size: int = 128, 
                 shuffle_buffer: int = 10000, infinite: bool = True):
        self.tfrecord_paths = [str(p) for p in tfrecord_paths]
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.infinite = infinite
        self.dataset = self._create_dataset()
        self.iterator = iter(self.dataset)
        
    def _create_dataset(self):
        """创建TensorFlow数据集"""
        # 创建数据集，自动并行读取多个文件
        dataset = tf.data.TFRecordDataset(self.tfrecord_paths, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.map(parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)
        
        # 设置无限重复（用于训练）
        if self.infinite:
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def tf_to_torch_batch(self, tf_batch):
        """将TensorFlow batch转换为PyTorch TensorDict格式"""
        return tf_to_torch_batch_func(tf_batch)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        tf_batch = next(self.iterator)
        torch_batch = self.tf_to_torch_batch(tf_batch)
        return torch_batch

class TFRecordDataset:
    """TFRecord数据集，用于eval_loader，支持直接迭代"""
    
    def __init__(self, tfrecord_paths: List[Path], batch_size: int = 128, 
                 shuffle_buffer: int = 1000, infinite: bool = False):
        self.tfrecord_paths = [str(p) for p in tfrecord_paths]
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.infinite = infinite
        self.dataset = self._create_dataset()
        
    def _create_dataset(self):
        """创建TensorFlow数据集"""
        # 创建数据集，自动并行读取多个文件
        dataset = tf.data.TFRecordDataset(self.tfrecord_paths, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.map(parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)
        
        # 设置无限重复（用于训练）
        if self.infinite:
            dataset = dataset.repeat()
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def __iter__(self):
        """返回数据集迭代器"""
        for tf_batch in self.dataset:
            torch_batch = tf_to_torch_batch_func(tf_batch)
            yield torch_batch

class TFRecordDataLoader:
    """TFRecord数据加载器"""
    
    def __init__(self, data_root: str, env_name: str, split: str = "train"):
        self.data_root = data_root
        self.env_name = env_name
        self.split = split
        self.tfrecord_paths = get_tfrecord_files(data_root, env_name, split)
        
        print(f"Found {len(self.tfrecord_paths)} TFRecord files for {split} split:")
        for path in self.tfrecord_paths:
            print(f"  {path}")
    
    def get_iter(self, batch_size: int = 128, infinite: bool = True, shuffle_buffer: int = 10000):
        """创建数据迭代器"""
        return TFRecordDataIterator(
            tfrecord_paths=self.tfrecord_paths,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            infinite=infinite
        )
    
    def get_dataset(self, batch_size: int = 128, infinite: bool = False, shuffle_buffer: int = 1000):
        """创建数据集（用于eval_loader）"""
        return TFRecordDataset(
            tfrecord_paths=self.tfrecord_paths,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            infinite=infinite
        )

class TFRecordDataLoaderWithRatio:
    """支持按比例分割的TFRecord数据加载器"""
    
    def __init__(self, data_root: str, env_name: str, split: str = "train", ratio: float = 1.0, use_remaining: bool = False):
        self.data_root = data_root
        self.env_name = env_name
        self.split = split
        self.ratio = ratio
        self.use_remaining = use_remaining
        self.tfrecord_paths = get_tfrecord_files_with_ratio(data_root, env_name, split, ratio, use_remaining)
        
        # 打印加载的文件信息
        if split == "train":
            file_type = "remaining" if use_remaining else "ratio"
            print(f"Loaded {len(self.tfrecord_paths)} TFRecord files for {split} split ({file_type}={ratio:.2f}):")
            if len(self.tfrecord_paths) <= 5:
                for path in self.tfrecord_paths:
                    print(f"  {path.name}")
            else:
                print(f"  {self.tfrecord_paths[0].name} to {self.tfrecord_paths[-1].name}")
    
    def get_iter(self, batch_size: int = 128, infinite: bool = True, shuffle_buffer: int = 10000):
        """创建数据迭代器"""
        return TFRecordDataIterator(
            tfrecord_paths=self.tfrecord_paths,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            infinite=infinite
        )
    
    def get_dataset(self, batch_size: int = 128, infinite: bool = False, shuffle_buffer: int = 1000):
        """创建数据集（用于eval_loader）"""
        return TFRecordDataset(
            tfrecord_paths=self.tfrecord_paths,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            infinite=infinite
        )

def load(env_name: str, data_root: str = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo"):
    """加载TFRecord数据，返回train和test数据加载器"""
    train_loader = TFRecordDataLoader(data_root, env_name, split="train")
    test_loader = TFRecordDataLoader(data_root, env_name, split="test")
    eval_loader = TFRecordDataLoader(data_root, env_name, split="test").get_dataset()
    return train_loader, test_loader, eval_loader


def load_noaction(env_name: str, data_root: str = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo", ratio=1.0):
    """加载TFRecord数据，返回train和test数据加载器"""
    train_loader = TFRecordDataLoader(data_root, f"{env_name}_{ratio}", split="train")
    train_noaction_loader = TFRecordDataLoader(data_root, f"{env_name}_{ratio}_noaction", split="train")
    test_loader = TFRecordDataLoader(data_root, f"{env_name}_{ratio}", split="test")
    eval_loader = TFRecordDataLoader(data_root, f"{env_name}_{ratio}", split="test").get_dataset()
    return train_loader, train_noaction_loader, test_loader, eval_loader

def load_ratiodata(env_name: str, data_root: str = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo", ratio=1.0):
    """加载TFRecord数据，按比例分割训练数据"""
    # train_loader加载前ratio比例的数据文件
    train_loader = TFRecordDataLoaderWithRatio(data_root, env_name, split="train", ratio=ratio, use_remaining=False)
    
    # train_noaction_loader加载剩余的数据文件
    train_noaction_loader = TFRecordDataLoaderWithRatio(data_root, env_name, split="train", ratio=ratio, use_remaining=True)
    
    # test数据加载器保持不变
    test_loader = TFRecordDataLoader(data_root, env_name, split="test")
    eval_loader = TFRecordDataLoader(data_root, env_name, split="test").get_dataset()
    
    return train_loader, train_noaction_loader, test_loader, eval_loader