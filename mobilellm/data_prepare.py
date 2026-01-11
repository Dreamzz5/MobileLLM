import os

import numpy as np
import torch

from .utils import StandardScaler, print_log, vrange


class STLLMDataset(torch.utils.data.Dataset):
    """支持embedding的数据集类，训练集及其他数据集均使用此格式"""

    def __init__(self, x_data, y_data, embeddings=None):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.embeddings = torch.FloatTensor(embeddings) if embeddings is not None else None

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return self.x_data[idx], self.y_data[idx], self.embeddings[idx]
        else:
            return self.x_data[idx], self.y_data[idx]


def dict_collate_fn(batch):
    """
    自定义collate函数，将元组格式转换为字典格式
    支持带和不带embeddings的情况
    """
    if len(batch[0]) == 3:  # 有embeddings
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        embeddings_batch = torch.stack([item[2] for item in batch])
        return {"x": x_batch, "y": y_batch, "embeddings": embeddings_batch}
    else:  # 没有embeddings
        x_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        return {"x": x_batch, "y": y_batch}


def get_dataloaders(
    data_dir,
    data_format="index",  # "index" 或 "stllm" 格式
    batch_size=64,
    log=None,
):
    """
    统一数据加载函数，所有数据集处理均以训练集（train）为基准：
    1. 标准化参数完全基于训练集统计特征
    2. 训练集主导控所有数据的预处理逻辑
    3. 验证/测试集严格遵循训练集的处理标准
    """
    # --------------------------
    # 1. 加载训练集（核心）
    # --------------------------
    # 加载stllm格式的训练数据（核心）
    train_data = np.load(os.path.join(data_dir, "train.npz"))
    x_train = train_data["x"].astype(np.float32)
    y_train = train_data["y"].astype(np.float32)

    # 基于训练集格式加载验证/测试集
    val_data = np.load(os.path.join(data_dir, "val.npz"))
    test_data = np.load(os.path.join(data_dir, "test.npz"))
    x_val = val_data["x"].astype(np.float32)
    y_val = val_data["y"].astype(np.float32)
    x_test = test_data["x"].astype(np.float32)
    y_test = test_data["y"].astype(np.float32)

    train_embeddings = None
    val_embeddings = None
    test_embeddings = None

    if data_format == "stllm":
        # 分别加载训练、验证、测试集的embedding
        train_embedding_path = os.path.join(data_dir, "train_embeddings.npz")
        val_embedding_path = os.path.join(data_dir, "val_embeddings.npz")
        test_embedding_path = os.path.join(data_dir, "test_embeddings.npz")

        if os.path.exists(train_embedding_path):
            train_embedding_data = np.load(train_embedding_path)
            train_embeddings = (
                train_embedding_data["embeddings"].astype(np.float32).transpose((0, 2, 1))
            )
            print_log(f"训练集embedding加载完成: {train_embeddings.shape}", log=log)

        if os.path.exists(val_embedding_path):
            val_embedding_data = np.load(val_embedding_path)
            val_embeddings = (
                val_embedding_data["embeddings"].astype(np.float32).transpose((0, 2, 1))
            )
            print_log(f"验证集embedding加载完成: {val_embeddings.shape}", log=log)

        if os.path.exists(test_embedding_path):
            test_embedding_data = np.load(test_embedding_path)
            test_embeddings = (
                test_embedding_data["embeddings"].astype(np.float32).transpose((0, 2, 1))
            )
            print_log(f"测试集embedding加载完成: {test_embeddings.shape}", log=log)

    # 特征数量以训练集为准（排除时间特征）
    num_features = x_train.shape[-1] - 2

    # --------------------------
    # 2. 以训练集为基准进行标准化
    # --------------------------
    scalers = []
    print_log(f"以训练集为基准进行特征标准化（共{num_features}个特征）", log=log)
    for feature_idx in range(num_features):
        # 仅用训练集计算均值和标准差（核心）
        scaler = StandardScaler(
            mean=x_train[..., feature_idx].mean(), std=x_train[..., feature_idx].std()
        )
        scalers.append(scaler)

        # 所有数据集（包括验证/测试）都使用训练集的scaler
        x_train[..., feature_idx] = scaler.transform(x_train[..., feature_idx])
        x_val[..., feature_idx] = scaler.transform(x_val[..., feature_idx])
        x_test[..., feature_idx] = scaler.transform(x_test[..., feature_idx])

    # --------------------------
    # 3. 以训练集为模板创建数据集
    # --------------------------
    print_log(f"训练集主导的数据加载完成:", log=log)
    print_log(f"训练集:  x-{x_train.shape}  y-{y_train.shape}", log=log)
    print_log(f"验证集:  x-{x_val.shape}  y-{y_val.shape}", log=log)
    print_log(f"测试集:  x-{x_test.shape}  y-{y_test.shape}", log=log)

    # 分别创建带embedding的数据集
    if train_embeddings is not None:
        trainset = STLLMDataset(x_train, y_train, train_embeddings)
        valset = (
            STLLMDataset(x_val, y_val, val_embeddings)
            if val_embeddings is not None
            else torch.utils.data.TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
        )
        testset = (
            STLLMDataset(x_test, y_test, test_embeddings)
            if test_embeddings is not None
            else torch.utils.data.TensorDataset(
                torch.FloatTensor(x_test), torch.FloatTensor(y_test)
            )
        )
        print_log(f"训练集使用embedding，验证/测试集根据可用性决定", log=log)
    else:
        trainset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_train), torch.FloatTensor(y_train)
        )
        valset = torch.utils.data.TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
        testset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_test), torch.FloatTensor(y_test)
        )

    # --------------------------
    # 4. 创建数据加载器（训练集可打乱）
    # --------------------------
    trainset_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dict_collate_fn,  # 训练集打乱，使用字典格式
    )
    valset_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dict_collate_fn,  # 验证/测试集不打乱，使用字典格式
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, collate_fn=dict_collate_fn  # 使用字典格式
    )

    return trainset_loader, valset_loader, testset_loader, scalers
