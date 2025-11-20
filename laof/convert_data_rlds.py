import os
import h5py
import numpy as np
import tensorflow as tf
import random
from pathlib import Path
import argparse
from tqdm import tqdm

class HDF5ToTFRecordConverter:
    def __init__(self, data_root, env_name, add_time_horizon=1):
        self.data_root = Path(data_root)
        self.env_name = env_name
        self.add_time_horizon = add_time_horizon
        self.data_path = self.data_root / "expert_data" / "opticalflow" / env_name
        
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
    
    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))
    
    def load_demo_from_hdf5(self, file_path, demo_key):
        """从HDF5文件中加载单个demo数据"""
        with h5py.File(file_path, 'r') as f:
            demo_group = f[demo_key]
            
            data = {
                'obs': np.array(demo_group['obs']),
                'ta': np.array(demo_group['ta']),
                'done': np.array(demo_group['done']),
                'rewards': np.array(demo_group['rewards']),
                'ep_returns': np.array(demo_group['ep_returns']),
                'obs_of': np.array(demo_group['obs_of']),
                'obs_sam': np.array(demo_group['obs_sam'])
            }
            
        return data
    
    def process_trajectory_data(self, data):
        """处理轨迹数据，添加时间维度，返回独立的时间步样本列表"""
        if self.add_time_horizon != 1:
            raise NotImplementedError("Currently only supports ADD_TIME_HORIZON=1")
        
        # 获取数据长度
        seq_len = data['obs'].shape[0]
        
        if seq_len < 3:
            return []  # 数据太短，无法创建3帧组合
        
        # 处理obs: 将通道维度移动到前面 (H, W, C) -> (C, H, W)
        obs_original = data['obs']  # (T, 64, 64, 3)
        obs_transposed = np.transpose(obs_original, (0, 3, 1, 2))  # (T, 3, 64, 64)
        
        # 处理obs_of和obs_sam: 同样移动通道维度，格式与obs保持一致
        obs_of_original = data['obs_of']  # (T, 64, 64, 3)
        obs_of_transposed = np.transpose(obs_of_original, (0, 3, 1, 2))  # (T, 3, 64, 64)
        
        obs_sam_original = data['obs_sam']  # (T, 64, 64, 3)
        obs_sam_transposed = np.transpose(obs_sam_original, (0, 3, 1, 2))  # (T, 3, 64, 64)
        
        # 创建独立的时间步样本
        samples = []
        for i in range(seq_len - 2):
            sample = {
                'obs': np.stack([
                    obs_transposed[i],      # 过去帧
                    obs_transposed[i+1],    # 当前帧
                    obs_transposed[i+2]     # 未来帧
                ], axis=0),  # (3, 3, 64, 64)
                
                'obs_of': np.stack([
                    obs_of_transposed[i],   # 过去帧
                    obs_of_transposed[i+1], # 当前帧
                    obs_of_transposed[i+2]  # 未来帧
                ], axis=0),  # (3, 3, 64, 64) - 与obs格式一致
                
                'obs_sam': np.stack([
                    obs_sam_transposed[i],   # 过去帧
                    obs_sam_transposed[i+1], # 当前帧
                    obs_sam_transposed[i+2]  # 未来帧
                ], axis=0),  # (3, 3, 64, 64) - 与obs格式一致
                
                'ta': np.array([
                    data['ta'][i],
                    data['ta'][i+1],
                    data['ta'][i+2]
                ]),  # (3,)
                
                'done': np.array([
                    data['done'][i],
                    data['done'][i+1],
                    data['done'][i+2]
                ]),  # (3,)
                
                'rewards': np.array([
                    data['rewards'][i],
                    data['rewards'][i+1],
                    data['rewards'][i+2]
                ]),  # (3,)
                
                'ep_returns': data['ep_returns'][i+1]  # 标量
            }
            samples.append(sample)
        
        return samples
    
    def create_tf_example(self, sample):
        """创建TensorFlow Example"""
        feature = {
            'obs': self._bytes_feature(tf.io.serialize_tensor(sample['obs'].astype(np.uint8))),
            'ta': self._bytes_feature(tf.io.serialize_tensor(sample['ta'].astype(np.int64))),
            'done': self._bytes_feature(tf.io.serialize_tensor(sample['done'].astype(bool))),
            'rewards': self._bytes_feature(tf.io.serialize_tensor(sample['rewards'].astype(np.float32))),
            'ep_returns': self._bytes_feature(tf.io.serialize_tensor(np.array(sample['ep_returns']).astype(np.float32))),
            'obs_of': self._bytes_feature(tf.io.serialize_tensor(sample['obs_of'].astype(np.uint8))),
            'obs_sam': self._bytes_feature(tf.io.serialize_tensor(sample['obs_sam'].astype(np.uint8))),
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def get_all_demos(self, split):
        """获取所有demo的路径和键名"""
        split_path = self.data_path / split
        demo_list = []
        
        # 获取所有hdf5文件
        hdf5_files = sorted([f for f in split_path.glob("*.hdf5")])
        
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, 'r') as f:
                # 获取所有demo键
                demo_keys = [key for key in f.keys() if key.startswith('demo')]
                for demo_key in demo_keys:
                    demo_list.append((hdf5_file, demo_key))
        
        return demo_list
    
    def convert_split(self, split, output_dir, num_shards=None, shuffle=True):
        """转换一个数据分割（train或test）"""
        print(f"Converting {split} split...")
        
        # 获取所有demo
        demo_list = self.get_all_demos(split)
        print(f"Found {len(demo_list)} demos in {split} split")
        
        if shuffle:
            random.shuffle(demo_list)
        
        # 确定分片数量
        if num_shards is None:
            num_shards = 10 if split == 'train' else 1
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if num_shards == 1:
            # 单文件模式
            output_paths = [output_dir / f"{split}.tfrecord"]
            demo_shards = [demo_list]
        else:
            # 多文件模式（分成多个shard）
            output_paths = [output_dir / f"{split}-{i:05d}-of-{num_shards:05d}.tfrecord" 
                          for i in range(num_shards)]
            
            # 将demo分配到不同的shard
            demos_per_shard = len(demo_list) // num_shards
            demo_shards = []
            
            for i in range(num_shards):
                start_idx = i * demos_per_shard
                if i == num_shards - 1:
                    # 最后一个shard包含剩余的所有demo
                    end_idx = len(demo_list)
                else:
                    end_idx = (i + 1) * demos_per_shard
                demo_shards.append(demo_list[start_idx:end_idx])
        
        total_successful_demos = 0
        total_samples = 0
        
        # 写入每个shard
        for shard_idx, (output_path, shard_demos) in enumerate(zip(output_paths, demo_shards)):
            print(f"Writing shard {shard_idx + 1}/{num_shards}: {output_path}")
            
            with tf.io.TFRecordWriter(str(output_path)) as writer:
                successful_demos = 0
                shard_samples = 0
                
                for hdf5_file, demo_key in tqdm(shard_demos, desc=f"Processing shard {shard_idx + 1}"):
                    try:
                        # 加载demo数据
                        demo_data = self.load_demo_from_hdf5(hdf5_file, demo_key)
                        
                        # 处理轨迹数据，获取样本列表
                        samples = self.process_trajectory_data(demo_data)
                        
                        if not samples:
                            continue
                        
                        # 为每个样本创建TF Example并写入
                        for sample in samples:
                            example = self.create_tf_example(sample)
                            writer.write(example.SerializeToString())
                        
                        successful_demos += 1
                        shard_samples += len(samples)
                        
                    except Exception as e:
                        print(f"Error processing {hdf5_file}:{demo_key}: {e}")
                        continue
                
                print(f"Shard {shard_idx + 1}: {successful_demos} demos, {shard_samples} samples")
                total_successful_demos += successful_demos
                total_samples += shard_samples
        
        print(f"Successfully converted {total_successful_demos} demos with {total_samples} samples to {num_shards} file(s)")
        return output_paths
    
    def convert_all(self, output_dir=None, train_shards=10, test_shards=1, shuffle=True):
        """转换所有数据"""
        # 如果没有指定输出目录，使用默认的opticalflow_rlds路径
        if output_dir is None:
            output_dir = self.data_root / "expert_data" / "opticalflow_rlds" / self.env_name
        
        print(f"Converting data from {self.data_path}")
        print(f"Output directory: {output_dir}")
        print(f"Train shards: {train_shards}, Test shards: {test_shards}")
        
        # 转换train和test数据
        for split in ['train', 'test']:
            split_path = self.data_path / split
            if split_path.exists():
                num_shards = train_shards if split == 'train' else test_shards
                self.convert_split(split, output_dir, num_shards, shuffle)
            else:
                print(f"Warning: {split} directory not found at {split_path}")

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
    
    return parsed_example

def create_dataset(tfrecord_paths, batch_size=128, shuffle_buffer=1000):
    """创建TensorFlow数据集，支持多个TFRecord文件"""
    if isinstance(tfrecord_paths, (str, Path)):
        tfrecord_paths = [str(tfrecord_paths)]
    else:
        tfrecord_paths = [str(p) for p in tfrecord_paths]
    
    # 创建数据集，自动并行读取多个文件
    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    dataset = dataset.map(parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_tfrecord_files(output_dir, split='train'):
    """获取指定split的所有TFRecord文件路径"""
    output_dir = Path(output_dir)
    
    if split == 'train':
        # 查找train的分片文件
        pattern = f"{split}-*-of-*.tfrecord"
        files = sorted(list(output_dir.glob(pattern)))
        if not files:
            # 如果没找到分片文件，尝试单文件
            single_file = output_dir / f"{split}.tfrecord"
            if single_file.exists():
                files = [single_file]
    else:
        # test数据通常是单文件
        files = [output_dir / f"{split}.tfrecord"]
        files = [f for f in files if f.exists()]
    
    return files

def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 data to TFRecord format')
    parser.add_argument('--data_root', type=str, default='/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo',
                        help='Root directory of the data')
    parser.add_argument('--env_name', type=str, required=True,
                        help='Environment name')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for TFRecord files (default: data_root/expert_data/opticalflow_rlds/env_name)')
    parser.add_argument('--train_shards', type=int, default=10,
                        help='Number of shards for training data (default: 10)')
    parser.add_argument('--test_shards', type=int, default=1,
                        help='Number of shards for test data (default: 1)')
    parser.add_argument('--add_time_horizon', type=int, default=1,
                        help='Time horizon to add (currently only supports 1)')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Disable shuffling of data')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = HDF5ToTFRecordConverter(
        data_root=args.data_root,
        env_name=args.env_name,
        add_time_horizon=args.add_time_horizon
    )
    
    # 设置输出目录
    if args.output_dir is None:
        # 使用默认路径: data_root/expert_data/opticalflow_rlds/env_name
        output_dir = Path(args.data_root) / "expert_data" / "opticalflow_rlds" / args.env_name
    else:
        output_dir = args.output_dir
    print(f"Output directory: {output_dir}")
    
    # 执行转换
    converter.convert_all(
        output_dir=output_dir, 
        train_shards=args.train_shards,
        test_shards=args.test_shards,
        shuffle=not args.no_shuffle
    )
    
    print("Conversion completed!")
    
    # 测试加载
    train_tfrecords = get_tfrecord_files(output_dir, 'train')
    if train_tfrecords:
        print(f"\nTesting dataset loading with {len(train_tfrecords)} train files...")
        dataset = create_dataset(train_tfrecords, batch_size=128)
        
        for batch in dataset.take(1):
            print("Sample batch shapes:")
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")
        print("Dataset loading test successful!")

if __name__ == "__main__":
    main()