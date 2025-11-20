import h5py
import numpy as np
from pathlib import Path
import shutil
import re
import argparse

def get_demo_keys(hdf5_file_path):
    """获取HDF5文件中所有的demo键"""
    demo_keys = []
    
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # 获取所有以'demo'开头的键
            all_keys = list(f.keys())
            demo_keys = [key for key in all_keys if key.startswith('demo')]
            
            # 按demo编号排序
            def extract_demo_number(demo_key):
                match = re.match(r'demo(\d+)', demo_key)
                return int(match.group(1)) if match else 0
            
            demo_keys.sort(key=extract_demo_number)
            
    except Exception as e:
        print(f"Error reading {hdf5_file_path}: {e}")
        return []
    
    return demo_keys

def copy_demos_to_new_file(source_file, target_file, demo_keys_to_copy):
    """将指定的demo键复制到新文件"""
    
    try:
        with h5py.File(source_file, 'r') as src:
            with h5py.File(target_file, 'w') as dst:
                
                # 复制非demo数据（如果有的话）
                for key in src.keys():
                    if not key.startswith('demo'):
                        src.copy(key, dst)
                
                # 复制指定的demo数据
                for demo_key in demo_keys_to_copy:
                    if demo_key in src:
                        src.copy(demo_key, dst)
                        print(f"  Copied {demo_key}")
                    else:
                        print(f"  Warning: {demo_key} not found in source file")
        
        return True
        
    except Exception as e:
        print(f"Error copying demos: {e}")
        return False

def split_hdf5_dataset(source_file_path, nums, output_dir=None):
    """分割HDF5数据集
    
    Args:
        source_file_path: 源文件路径
        nums: 前nums个demo放入第一个文件
        output_dir: 输出目录，如果为None则使用源文件所在目录
    """
    
    source_path = Path(source_file_path)
    
    if not source_path.exists():
        print(f"Error: Source file not found: {source_file_path}")
        return False
    
    # 设置输出目录
    if output_dir is None:
        output_dir = source_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置输出文件路径
    base_name = source_path.stem  # 获取不带扩展名的文件名
    file1_path = output_dir / f"00.hdf5"
    file2_path = output_dir / f"01.hdf5"
    
    print(f"Splitting {source_file_path}")
    print(f"Output files:")
    print(f"  First {nums} demos -> {file1_path}")
    print(f"  Remaining demos -> {file2_path}")
    
    # 获取所有demo键
    demo_keys = get_demo_keys(source_path)
    
    if not demo_keys:
        print("No demo data found in source file")
        return False
    
    print(f"Found {len(demo_keys)} demos: {demo_keys}")
    
    # 验证nums参数
    if nums <= 0:
        print(f"Error: nums must be positive, got {nums}")
        return False
    
    if nums >= len(demo_keys):
        print(f"Warning: nums ({nums}) >= total demos ({len(demo_keys)})")
        print("All demos will be in the first file, second file will be empty")
    
    # 分割demo键
    first_demos = demo_keys[:nums]
    remaining_demos = demo_keys[nums:]
    
    print(f"\nFirst file will contain {len(first_demos)} demos: {first_demos}")
    print(f"Second file will contain {len(remaining_demos)} demos: {remaining_demos}")
    
    # 创建第一个文件
    print(f"\nCreating {file1_path}...")
    if not copy_demos_to_new_file(source_path, file1_path, first_demos):
        print(f"Failed to create {file1_path}")
        return False
    
    # 创建第二个文件
    print(f"\nCreating {file2_path}...")
    if not copy_demos_to_new_file(source_path, file2_path, remaining_demos):
        print(f"Failed to create {file2_path}")
        return False
    
    print(f"\nSplit completed successfully!")
    print(f"Files created:")
    print(f"  {file1_path} - {len(first_demos)} demos")
    print(f"  {file2_path} - {len(remaining_demos)} demos")
    
    return True

def verify_split_files(file1_path, file2_path, original_file_path):
    """验证分割后的文件"""
    print(f"\nVerifying split files...")
    
    # 获取各文件的demo数量
    original_demos = get_demo_keys(original_file_path)
    file1_demos = get_demo_keys(file1_path)
    file2_demos = get_demo_keys(file2_path)
    
    print(f"Original file: {len(original_demos)} demos")
    print(f"File 1: {len(file1_demos)} demos")
    print(f"File 2: {len(file2_demos)} demos")
    
    # 检查总数是否匹配
    total_split = len(file1_demos) + len(file2_demos)
    if total_split == len(original_demos):
        print("✓ Total demo count matches")
    else:
        print(f"✗ Demo count mismatch: {total_split} vs {len(original_demos)}")
    
    # 检查是否有重复
    all_split_demos = set(file1_demos + file2_demos)
    if len(all_split_demos) == total_split:
        print("✓ No duplicate demos")
    else:
        print("✗ Duplicate demos found")
    
    return total_split == len(original_demos) and len(all_split_demos) == total_split

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Split HDF5 dataset into two files')
    parser.add_argument('source_file', help='Source HDF5 file path')
    parser.add_argument('nums', type=int, help='Number of demos for the first file')
    parser.add_argument('--output_dir', help='Output directory (default: same as source)')
    parser.add_argument('--verify', action='store_true', help='Verify the split files')
    
    args = parser.parse_args()
    
    # 执行分割
    success = split_hdf5_dataset(args.source_file, args.nums, args.output_dir)
    
    if success and args.verify:
        # 验证分割结果
        source_path = Path(args.source_file)
        output_dir = Path(args.output_dir) if args.output_dir else source_path.parent
        
        file1_path = output_dir / "00.hdf5"
        file2_path = output_dir / "01.hdf5"
        
        verify_split_files(file1_path, file2_path, source_path)

if __name__ == "__main__":
    # 如果直接运行脚本，使用默认参数
    source_file = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/leaper/train/0.hdf5"
    # source_file = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/bigfish/train/2.hdf5"
    # source_file = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/heist/train/5.hdf5"
    # source_file = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/chaser/train/3.hdf5"
    
    # 获取用户输入
    try:
        nums = int(input("Enter the number of demos for the first file (00.hdf5): "))
        
        print(f"\nWill split {source_file}")
        print(f"First {nums} demos -> 00.hdf5")
        print(f"Remaining demos -> 01.hdf5")
        
        confirm = input("Continue? (y/N): ")
        if confirm.lower() in ['y', 'yes']:
            success = split_hdf5_dataset(source_file, nums)
            
            if success:
                # 询问是否验证
                verify = input("Verify the split files? (y/N): ")
                if verify.lower() in ['y', 'yes']:
                    source_path = Path(source_file)
                    output_dir = source_path.parent
                    
                    file1_path = output_dir / "00.hdf5"
                    file2_path = output_dir / "01.hdf5"
                    
                    verify_split_files(file1_path, file2_path, source_path)
        else:
            print("Operation cancelled.")
            
    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")