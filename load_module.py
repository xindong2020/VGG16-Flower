
import os
import sys
import shutil
import random
import time
import torch.utils.data as data
from torchvision import transforms, datasets


class Load:
    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def split(self, data_path, datasets_path):
        # 获取花种类
        flower_classes = [cla for cla in os.listdir(data_path) if ".txt" not in cla]

        self.mkdir(datasets_path)
        # 创建训练集文件夹
        train_path = datasets_path + "/train/"
        self.mkdir(train_path)
        for cla in flower_classes:
            self.mkdir(train_path + cla)

        # 创建测试集文件夹
        valid_path = datasets_path + "/valid/"
        self.mkdir(valid_path)
        for cla in flower_classes:
            self.mkdir(valid_path + cla)

        # 训练集与测试集划分比例 9：1
        split_rate = 0.9
        # 遍历 5 种花的全部图像并按比例分成训练集和验证集
        for cla in flower_classes:
            cla_path = data_path + '/' + cla + '/'
            cla_list = os.listdir(cla_path)     # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            num = len(cla_list)
            # sample() 函数返回一个从序列即列表、元组、字符串或集合中选择的特定长度的项目列表
            train_list = random.sample(cla_list, k=int(num * split_rate))
            for index, image in enumerate(cla_list):
                # train_list 中保存训练集 train 的图像名称
                if image in train_list:
                    image_path = cla_path + image
                    new_path = train_path + cla
                    shutil.move(image_path, new_path)   # 移动文件到新路径
                else:
                    image_path = cla_path + image
                    new_path = valid_path + cla
                    shutil.move(image_path, new_path)
                print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
                time.sleep(0.001)
        print("processing done!")

    def load_data(self, data_path, datasets_path, train_size, valid_size):

        self.split(data_path, datasets_path)

        num_workers = 0
        if not sys.platform.startswith("win"):
            num_workers = 8

        # 对图像的预处理
        data_transform = {"train": transforms.Compose([transforms.RandomResizedCrop([224, 224]),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                          "valid": transforms.Compose([transforms.Resize([224, 224]),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])}
        # 训练集
        train_datasets = datasets.ImageFolder(root=os.path.join(datasets_path, "train"), transform=data_transform["train"])
        print("Using {} train images".format(len(train_datasets)))
        train_loader = data.DataLoader(train_datasets, batch_size=train_size, shuffle=True, num_workers=num_workers)

        # 测试集
        valid_datasets = datasets.ImageFolder(root=os.path.join(datasets_path, "valid"), transform=data_transform["valid"])
        print("Using {} valid images".format(len(valid_datasets)))
        valid_loader = data.DataLoader(valid_datasets, batch_size=valid_size, shuffle=True, num_workers=num_workers)

        return train_loader, valid_loader
