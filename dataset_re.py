import cv2
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import os
import pandas as pd

major_name = ['가지', '거베라', '고추(청양)', '국화', '딸기', '멜론', '부추', '상추', '안개', '애플망고', '애호박', '오이', '장미',
              '토마토(일반)', '파프리카(볼로키)']
minor_name = ['개화기', '과비대성숙기', '과실비대기', '생육기', '수확기', '영양생장', '절화기', '정식기', '착과기', '착색기', '화아발달기', '화아분화기']

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=["file_Name", "pl_Name", "pl_Step", 'pl_Name+Step'],
                                      header=0)
        self.img_dir = img_dir
        self.transform = transform
        self.le_super = LabelEncoder()
        self.le_super = self.le_super.fit(major_name)
        self.le_super_class = self.le_super.classes_
        self.le_finer = LabelEncoder()
        self.le_finer = self.le_finer.fit(minor_name)
        self.le_finer_class = self.le_finer.classes_

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # image = decode_image(image)
        super_class = self.le_super.transform([self.img_labels.iloc[idx, 1]])
        finer_class = self.le_finer.transform([self.img_labels.iloc[idx, 2]])
        if self.transform:
            image = self.transform(image=image)["image"]
            # image = self.transform(image)
        return image, super_class, finer_class, self.img_labels.iloc[idx, 0]  # , self.le_super_class, self.le_finer_class


# img_labels = pd.read_csv('/home/sunwoo/Desktop/Crop/classfication/test_label.csv',
#                          names=["file_Name", "pl_Name", "pl_Step", "pl_Name+Step"], header=0)
# print(img_labels['pl_Name'].value_counts())
# # print(len(img_labels['pl_Name']))
# print(img_labels['pl_Step'].value_counts())
# print(img_labels['pl_Name+Step'].value_counts())
# for i in range(len(img_labels['pl_Step'])):
#     img_labels['pl_Step'][i] = img_labels['pl_Step'][i].split(' ')[1]
# print(img_labels['pl_Step'].value_counts())
# le_finer = LabelEncoder()
# le_super = LabelEncoder()
# le_super = le_super.fit(img_labels['pl_Name'])
# le_finer.fit(img_labels['pl_Step'])
# le_finer_class = le_finer.classes_
# le_super_class = le_super.classes_
# print(le_finer_class)
# print(len(le_finer_class))
# print(le_super_class)
# print(len(le_super_class))
# print(le_finer.transform(['생육기']))
# print(le_super.transform(['딸기']))

#
# import os
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data.dataset import Dataset
# from time import time
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# # N_CHANNELS = 3
# #
# dataset = CustomImageDataset('/home/sw/바탕화면/crop/classfication/test_label.csv','/home/sw/바탕화면/crop/test_image/', transform=transform)
# full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=os.cpu_count())
#
# mean = torch.zeros(1)
# std = torch.zeros(1)
# print('==> Computing mean and std..')
# for inputs, _, _ in full_loader:
#     temp_mean = torch.mean(inputs)
#     temp_std = torch.std(inputs)
#     mean += temp_mean
#     std += temp_std
# mean.div_(len(dataset))
# std.div_(len(dataset))
# print(mean, std)
