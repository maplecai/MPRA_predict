import numpy as np
import h5py


def load_h5(file_dir: str):
    with h5py.File(file_dir, 'r') as f:
        keys = list(f.keys())
        if len(keys) == 1:
            data = f[keys[0]][:]
        else:
            print('h5 file has more than one key')
            print(f'keys: {keys}')
            data = f
    return data

def save_h5(file_dir: str, data) -> None:
    with h5py.File(file_dir, 'w') as f:
        f.create_dataset('data', data=data)
    return

import h5py
import numpy as np




import h5py
import numpy as np


class HDF5Writer:
    def __init__(self, file_path, dataset_name, data_shape, max_samples=None, chunk_size=100, dtype="float32", compression="gzip"):
        """
        HDF5 增量写入工具
        Args:
            file_path (str): HDF5 文件路径
            dataset_name (str): 数据集名称
            data_shape (tuple): 单个样本的形状，例如 (2048, 305)
            max_samples (int, optional): 最大样本数（None 表示动态增长）
            chunk_size (int): 每块的 batch 大小
            dtype (str): 数据类型
            compression (str): 压缩方式，可为 None/gzip/lzf
        """
        self.file = h5py.File(file_path, "a")
        self.dataset_name = dataset_name
        self.data_shape = data_shape
        self.max_samples = max_samples
        self.chunk_size = chunk_size

        if dataset_name not in self.file:
            if max_samples is None:
                dset = self.file.create_dataset(
                    dataset_name,
                    shape=(0,) + data_shape,
                    maxshape=(None,) + data_shape,
                    dtype=dtype,
                    chunks=(chunk_size,) + data_shape,
                    compression=compression
                )
            else:
                dset = self.file.create_dataset(
                    dataset_name,
                    shape=(max_samples,) + data_shape,
                    dtype=dtype,
                    chunks=(chunk_size,) + data_shape,
                    compression=compression
                )
        else:
            dset = self.file[dataset_name]

        self.dset = dset
        self.index = dset.shape[0] if max_samples is None else 0

    def append(self, batch):
        """追加写入 batch 数据"""
        batch = np.asarray(batch, dtype=self.dset.dtype)
        n = batch.shape[0]

        if self.max_samples is None:
            self.dset.resize(self.index + n, axis=0)
        elif self.index + n > self.max_samples:
            raise ValueError("超过最大样本数！")

        self.dset[self.index:self.index+n, ...] = batch
        self.index += n
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()










class HDF5MultiWriter:
    def __init__(self, file_path, chunk_size=100, dtype="float32", compression="gzip"):
        """
        HDF5 多数据集增量写入工具
        Args:
            file_path (str): HDF5 文件路径
            chunk_size (int): 每块的 batch 大小
            dtype (str): 默认数据类型
            compression (str): 压缩方式，可为 None/gzip/lzf
        """
        self.file = h5py.File(file_path, "a")
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.compression = compression
        self.datasets = {}  # 存储 dataset_name -> (dataset对象, 当前index, max_samples)

    def create_dataset(self, dataset_name, data_shape, max_samples=None):
        """
        创建一个新的数据集
        Args:
            dataset_name (str): 数据集名称
            data_shape (tuple): 单个样本的形状
            max_samples (int, optional): 最大样本数（None 表示动态增长）
        """
        if dataset_name in self.file:
            dset = self.file[dataset_name]
            index = dset.shape[0] if max_samples is None else 0
        else:
            if max_samples is None:
                dset = self.file.create_dataset(
                    dataset_name,
                    shape=(0,) + data_shape,
                    maxshape=(None,) + data_shape,
                    dtype=self.dtype,
                    chunks=(self.chunk_size,) + data_shape,
                    compression=self.compression
                )
                index = 0
            else:
                dset = self.file.create_dataset(
                    dataset_name,
                    shape=(max_samples,) + data_shape,
                    dtype=self.dtype,
                    chunks=(self.chunk_size,) + data_shape,
                    compression=self.compression
                )
                index = 0
        self.datasets[dataset_name] = {"dset": dset, "index": index, "max_samples": max_samples}

    def append(self, dataset_name, batch):
        """追加写入 batch 数据到指定数据集"""
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 {dataset_name} 尚未创建，请先调用 create_dataset()")

        info = self.datasets[dataset_name]
        dset, index, max_samples = info["dset"], info["index"], info["max_samples"]

        batch = np.asarray(batch, dtype=dset.dtype)
        n = batch.shape[0]

        if max_samples is None:
            dset.resize(index + n, axis=0)
        elif index + n > max_samples:
            raise ValueError(f"数据集 {dataset_name} 超过最大样本数！")

        dset[index:index+n, ...] = batch
        info["index"] += n
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()








def pad_seq(seq: str, padded_length: int, padding_method:str ='N', padding_position: str='both_sides', given_left_seq: str=None, given_right_seq: str=None, genome=None) -> str:
    seq_len = len(seq)
    if seq_len > padded_length:
        raise ValueError(f'seq_len={seq_len} > padded_length={padded_length}')
    padding_len = padded_length - seq_len

    if padding_position == 'both_sides':
        left_len = padding_len // 2
        right_len = padding_len - left_len
    elif padding_position == 'left':
        left_len = padding_len
        right_len = 0
    elif padding_position == 'right':
        left_len = 0
        right_len = padding_len
    elif padding_position.isdigit():
        left_len = int(padding_position)
        right_len = padding_len - left_len
    elif padding_position == 'random':
        left_len = np.random.randint(0, padding_len)
        right_len = padding_len - left_len
    else:
        raise ValueError('padding_postition must be "both_sides", "left", "right" or a integer')

    if padding_method == 'N':
        left_seq = 'N' * left_len
        right_seq = 'N' * right_len

    elif padding_method == 'random':
        bases = np.array(['A', 'C', 'G', 'T'])
        left_seq = ''.join(bases[np.random.randint(0, 4, left_len)])
        right_seq = ''.join(bases[np.random.randint(0, 4, right_len)])

    elif padding_method == 'genome':
        left_seq = random_genome_seq(genome, left_len) if left_len > 0 else ''
        right_seq = random_genome_seq(genome, right_len) if right_len > 0 else ''

    elif padding_method == 'repeat':
        if left_len == 0:
            left_seq = ''
        else:
            repeats_needed = left_len // seq_len + 1
            repeated_seq = seq * repeats_needed
            left_seq = repeated_seq[-left_len:]
        if right_len == 0:
            right_seq = ''
        else:
            repeats_needed = right_len // seq_len + 1
            repeated_seq = seq * repeats_needed
            right_seq = repeated_seq[:right_len]

    elif padding_method == 'given':
        if len(given_left_seq) < left_len or len(given_right_seq) < right_len:
            raise ValueError('given_left_seq and given_right_seq must be at least as long as the padding length')
        left_seq = given_left_seq[-left_len:] if left_len > 0 else ''
        right_seq = given_right_seq[:right_len] if right_len > 0 else ''

    elif padding_method == 'given+N':
        if left_len == 0:
            left_seq = ''
        elif len(given_left_seq) < left_len:
            left_seq = 'N' * (left_len - len(given_left_seq)) + given_left_seq
        else:
            left_seq = given_left_seq[-left_len:]
        if right_len == 0:
            right_seq = ''
        elif len(given_right_seq) < right_len:
            right_seq =  given_right_seq + 'N' * (right_len - len(given_right_seq))
        else:
            right_seq = given_right_seq[:right_len]
        
    else:
        raise ValueError('padding_method must be "N", "random", or "given"')
    
    padded_seq = ''.join([left_seq, seq, right_seq])
    return padded_seq
