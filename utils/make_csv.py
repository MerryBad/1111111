import json
import csv

import os
import pandas as pd
import json
import numpy as np
import re
import random
major_name = ['가지', '거베라', '고추(청양)', '국화', '딸기', '멜론', '부추', '상추', '안개', '애플망고', '애호박', '오이', '장미',
              '토마토(일반)', '파프리카(볼로키)']
name = ['가지 개화기', '가지 수확기', '가지 영양생장', '거베라 수확기', '거베라 영양생장',
        '거베라 정식기', '고추(청양) 개화기', '고추(청양) 생육기', '고추(청양) 수확기', '고추(청양) 정식기',
        '국화 생육기', '국화 영양생장', '국화 정식기', '국화 화아발달기', '국화 화아분화기',
        '딸기 개화기', '딸기 꽃눈분화기', '딸기 수확기', '멜론 개화기', '멜론 과실비대기',
        '멜론 영양생장', '멜론 정식기', '부추 생육기', '부추 영양생장', '상추 생육기',
        '상추 수확기', '상추 영양생장', '상추 착과기', '안개 개화기', '안개 수확기',
        '안개 영양생장', '애플망고 개화기', '애플망고 화아분화기', '애호박 생육기', '애호박 수확기',
        '애호박 영양생장', '애호박 착과기', '오이 개화기', '오이 영양생장', '장미 영양생장',
        '장미 절화기', '토마토(일반) 생육기', '토마토(일반) 수확기', '토마토(일반) 정식기', '토마토(일반) 착과기',
        '파프리카(볼로키) 과비대성숙기', '파프리카(볼로키) 과실비대기', '파프리카(볼로키) 수확기', '파프리카(볼로키) 착과기', '파프리카(볼로키) 착색기']

# df = pd.read_csv('E:/test_label.csv', header=0, encoding='cp949')
# all_label = pd.read_csv('all_label.csv', encoding='cp949')
# temp=pd.DataFrame(columns=["file_Name", "pl_Name", 'pl_Type', 'pl_Step'])
#
# for name in all_label:
#     is_it = df['file_Name'] == name
#     temp = pd.concat([temp, df[is_it]])
# temp.to_csv('data_label.csv')

# data = pd.read_csv('data_label.csv', header=0)
# is_it = data['pl_Name'] == '가지'
# print(data[is_it]['file_Name'].to_list())

# a.drop(columns='ind', inplace=True,axis=1)
# a.to_csv('data_label2.csv',header=["file_Name", "pl_Name", 'pl_Type', 'pl_Step'],index=False)


# print(data['pl_Name'].value_counts())  # 15
# print(data['pl_Step'].value_counts())  # 50
#
# for i in range(len(data['pl_Step'])):
#     data['pl_Step'][i] = data['pl_Step'][i].split(' ')[1]
#
# print(data['pl_Step'].value_counts())  # 50


# df = pd.read_csv('data_label.csv')
# df.set_index("file_Name", inplace=True)
# df.to_csv('data_label2.csv')



#     f.writerow(["file_Name", "pl_Name", 'pl_Type', 'pl_Step']

# df = pd.read_csv('E:/test_label.csv', header=0, encoding='cp949')
# is_it = df['pl_Step'] == name[0]
# all_label = [] # df에 넣어서 df 새로 만들기?
# each_label = {} # 이미지 꺼내와서 전처리하기
# for i in name:
#     is_it = df['pl_Step'] == i
#     temp = random.sample(df[is_it]['file_Name'].to_list(), 400)
#     each_label[i.split()[0]] = temp
#     all_label.extend(temp)
#
# with open('Each_label.csv', 'w', newline='') as f:
#     write = csv.writer(f)
#     write.writerow(each_label.keys())
#     write.writerow(each_label.values())
#
# with open('all_label.csv', 'w', newline='') as f:
#     write = csv.writer(f)
#     write.writerow(all_label)

p = re.compile('[0-9]+_[0-9]+_[0-9]+')
path = '/home/sw/바탕화면/crop/train_label/'

# path_label = 'D:/dataset/label/'
file_List = os.listdir(path)

# with open('../train_label.csv', 'w', newline='') as output:
#     f = csv.writer(output)
#     f.writerow(["file_Name", "pl_Name", 'pl_Step', 'pl_Name+Step'])
#     for i in file_List:
#         for line in open((path + i), "r", encoding='utf-8'):
#             try:
#                 df = pd.Series(json.loads(line))
#                 if df.values[0][0]['pl_step'] == '영양생장기':
#                     os.remove(path + i)
#                     continue
#                 f.writerow([p.findall(path + i)[0].strip("'") + '.jpg', df.values[0][0]['pl_name'],
#                             df.values[0][0]['pl_step'], df.values[0][0]['pl_name'] + ' ' + df.values[0][0]['pl_step']])
#             except:
#                 print(path+i)
#                 exit()

data = pd.read_csv('../train_label.csv', header=0)
# print(data['pl_Name'].value_counts())  # 15
print(data['pl_Step'].value_counts())  # 14
print(data['pl_Name+Step'].value_counts())  # 52

# # train image>label
# # test image<label
# path_image = '/home/sw/바탕화면/crop/train_image/'
# path_label = '/home/sw/바탕화면/crop/train_label/'
# image_List = os.listdir(path_image)
# label_List = os.listdir(path_label)
# count = 0
#
# for i in range(len(image_List)):
#     image_List[i] = image_List[i].strip(".jpg")
# for j in range(len(label_List)):
#     label_List[j] = label_List[j].strip(".json")
#
# for i in image_List:
#     if i in label_List:
#         pass
#     else:
#         print(i)
#         os.remove(path_image+i+".jpg")

# for i in image_List:
#     # print(path_label+i.strip(".JPG")+'.json')
#     # print(path_label+i.upper().strip(".JPG"))
#     # count += 1
#     # print(str(count) + '/' + str(len(image_List)))
#     #
#     # try:
#     #     sh.copy(path_label + i.upper().strip(".JPG") + '.json',
#     #             'E:/test_label2/' + i.upper().strip(".JPG") + '.json')
#     # except:
#     #     #     os.remove(path_image+i)
#     #     pass
#     for j in label_List:
#         if i.strip(".JPG") in j.strip(".JSON"):
#             break
#         else:
#             print(i.strip(".jpg"))
#             print(j.strip(".json"))
#             s
#             os.remove(path_image + i)
#             s

# import tarfile
#
# path = 'D:/시설 작물 개체 이미지/거베라/'
# des = 'D:/시설 작물 개체 이미지/거베라/'
# fname = os.listdir(path)
# # print(path+fname[0])
# # print(des+fname[0])
# # s
# for i in range(len(fname)):
#     ap = tarfile.open(path + fname[i])
#     ap.extractall(des)
#     ap.close()


#
#
# data = pd.read_csv('E:/test_label1.csv', header=0)
#
# for i in range(len(data)):
#     if data.iloc[i, 3] == '멜론 영양생장기' or data.iloc[i, 3] == '토마토(일반) 영양생장':
#         try:
#             os.remove(path_image+data.iloc[i, 0])
#             os.remove(path_label+data.iloc[i, 0][:-4]+'.json')
#         except:
#             pass
