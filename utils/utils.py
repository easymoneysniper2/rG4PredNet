import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from termcolor import cprint

def log_print(text, color=None, on_color=None, attrs=None):
        if cprint is not None:
            cprint(text, color=color, on_color=on_color, attrs=attrs)
        else:
            print(text)

def read_csv(path):
    # 使用pandas的read_csv函数读取path指定的csv文件，header=None表示文件没有标题行
    df = pd.read_csv(path, sep='\t', header=None)

    df = df.loc[df[0] != "Name"]

    # 定义列索引
    NAME_COL = 0
    SEQ_COL = 1
    STR_COL = 2
    LABEL_COL = 3

    # 提取所需列并转换为numpy数组
    names = df[NAME_COL].to_numpy()
    sequences = df[SEQ_COL].to_numpy()
    structs = df[STR_COL].to_numpy()
    targets = df[LABEL_COL].to_numpy().astype(np.float32).reshape(-1, 1)

    return names,sequences, structs, targets

def read_csv_with_name(path):
    # load sequences
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.loc[df[0] != "Type"]

    Type = 0
    loc = 1
    Seq = 2
    Str = 3
    Score = 4
    label = 5

    name = df[loc].to_numpy()
    sequences = df[Seq].to_numpy()
    structs = df[Str].to_numpy()
    targets = df[label].to_numpy().astype(np.float32).reshape(-1, 1)
    return name, sequences, structs, targets

def convert_one_hot(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""
    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        # print(index)
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2,index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3,index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)
    one_hot_seq = np.array(one_hot_seq)
    return one_hot_seq



def split_dataset(data1, data2, targets, test_frac=0.2, random_seed=None):
    """
    将数据集划分为训练集和测试集。

    参数:
    data1 (numpy.ndarray): 第一个数据集。
    data2 (numpy.ndarray): 第二个数据集。
    targets (numpy.ndarray): 目标标签。
    test_frac (float): 测试集所占比例，默认为0.2。
    random_seed (int, optional): 随机种子，默认为None。

    返回:
    tuple: 包含训练集和测试集的元组，每个元素都是一个包含data1, data2和targets的列表。
    """
    if not (0 <= test_frac <= 1):
        raise ValueError("test_frac should be between 0 and 1")

    if random_seed is not None:
        np.random.seed(random_seed)

    # 将targets中小于0.5的索引赋值给neg_indices
    neg_indices = np.where(targets < 0.5)[0]
    # 将targets中大于等于0.5的索引赋值给pos_indices
    pos_indices = np.where(targets >= 0.5)[0]

    # 计算正、负样本在测试集中的数量
    n_neg_test = int(len(neg_indices) * test_frac)
    n_pos_test = int(len(pos_indices) * test_frac)

    # 生成两个随机排列，长度分别为正、负样本的数量
    shuffled_neg_indices = np.random.permutation(len(neg_indices))
    shuffled_pos_indices = np.random.permutation(len(pos_indices))

    def create_dataset(pos_indices, neg_indices, n_pos, n_neg):
        X1 = np.concatenate((data1[pos_indices[:n_pos]], data1[neg_indices[:n_neg]]))
        X2 = np.concatenate((data2[pos_indices[:n_pos]], data2[neg_indices[:n_neg]]))
        Y = np.concatenate((targets[pos_indices[:n_pos]], targets[neg_indices[:n_neg]]))
        return [X1, X2, Y]

    # 生成训练集部分
    train_pos_indices = pos_indices[shuffled_pos_indices[n_pos_test:]]
    train_neg_indices = neg_indices[shuffled_neg_indices[n_neg_test:]]
    train = create_dataset(train_pos_indices, train_neg_indices, len(pos_indices) - n_pos_test, len(neg_indices) - n_neg_test)

    # 生成测试集部分
    test_pos_indices = pos_indices[shuffled_pos_indices[:n_pos_test]]
    test_neg_indices = neg_indices[shuffled_neg_indices[:n_neg_test]]
    test = create_dataset(test_pos_indices, test_neg_indices, n_pos_test, n_neg_test)

    return train, test


def k_fold_split(data1, data2, targets, folds=10, random_seed=None):
    """
    将数据集划分为k折交叉验证的数据集。

    参数:
    data1 (numpy.ndarray): 第一个数据集。
    data2 (numpy.ndarray): 第二个数据集。
    targets (numpy.ndarray): 目标标签。
    folds (int): 折数，默认为10。
    random_seed (int, optional): 随机种子，默认为None。

    返回:
    list: 包含每折训练集和验证集的列表，每个元素都是一个包含data1, data2和targets的元组。
    """
    if folds <= 1:
        raise ValueError("folds should be greater than 1")

    if random_seed is not None:
        np.random.seed(random_seed)

    # 将targets中小于0.5的索引赋值给neg_indices
    neg_indices = np.where(targets < 0.5)[0]
    # 将targets中大于等于0.5的索引赋值给pos_indices
    pos_indices = np.where(targets >= 0.5)[0]

    # 生成两个随机排列，长度分别为正、负样本的数量
    shuffled_neg_indices = np.random.permutation(neg_indices)
    shuffled_pos_indices = np.random.permutation(pos_indices)

    # 将正负样本的索引分别分割成folds份
    neg_folds = np.array_split(shuffled_neg_indices, folds)
    pos_folds = np.array_split(shuffled_pos_indices, folds)

    def create_dataset(pos_indices, neg_indices):
        X1 = np.concatenate((data1[pos_indices], data1[neg_indices]))
        X2 = np.concatenate((data2[pos_indices], data2[neg_indices]))
        Y = np.concatenate((targets[pos_indices], targets[neg_indices]))
        return [X1, X2, Y]

    datasets = []
    for i in range(folds):
        # 验证集
        val_pos_indices = pos_folds[i]
        val_neg_indices = neg_folds[i]
        val = create_dataset(val_pos_indices, val_neg_indices)

        # 训练集
        train_pos_indices = np.concatenate([pos_folds[j] for j in range(folds) if j != i])
        train_neg_indices = np.concatenate([neg_folds[j] for j in range(folds) if j != i])
        train = create_dataset(train_pos_indices, train_neg_indices)

        datasets.append((train, val))

    return datasets


def param_num(model):
    params = list(model.parameters())
    num_param0 = sum(p.numel() for p in params)
    num_param1 = sum(p.numel() for p in params if p.requires_grad)
    
    print("===========================")
    print(f"Total params: {num_param0}")
    print(f"Trainable params: {num_param1}")
    print(f"Non-trainable params: {num_param0 - num_param1}")
    print("===========================")


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            #如果提供了后续调度器，使用它来计算学习率
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


if __name__ == "__main__":
    path = "../test.tsv"
    sequences, structs, targets = read_csv(path)
    print(sequences[0])
    print(structs[0])
    print(targets[0])