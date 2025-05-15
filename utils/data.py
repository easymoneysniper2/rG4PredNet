import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    自定义数据集类，用于加载Ernie嵌入、结构信息、注意力图和标签。
    """
    def __init__(self, embedding, structure, label):
        self.embedding = embedding
        self.structure = structure
        self.label = label

    def __getitem__(self, index):
        """
        获取数据集中指定索引的数据。
        """
        embedding = self.embedding[index]
        structure = self.structure[index]
        label = self.label[index]

        return embedding, structure, label

    def __len__(self):
        """
        获取数据集的长度。
        """
        return len(self.label)