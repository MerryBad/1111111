import glob
import shutil
import re
import os


# #
# p=re.compile('[0-9]+_[0-9]+_[0-9]+')
# path = 'D:/dataset/Training'
#
# targetPattern = r"/home/sunwoo/Documents/Dataset/val_image/*.jpg"
# file_List = glob.glob(targetPattern) # 전체이름
# file_name = []
#
# for i in file_List:
#     print(p.findall(path+i))
#     s
#     file_name.append(p.findall(path+i)[0].strip("'")) #숫자만 나옴
# for i in range(len(file_List)):
#     # sh.move(file_List[i], path + 'train/' + file_name[i] + '.JPG')
#     print(file_List[i])
#     print(path + 'train/' + file_name[i] + '.JPG')

# path = '/home/sunwoo/Documents/Dataset/123/'
#
# folder_path = []
# for filename in os.listdir(path):
#     temp = os.path.join(path, filename)
#     if os.path.isdir(temp):
#         folder_path.append(temp)
# for i in range(len(folder_path)):
#     # print(folder_path[i])
#     file_list = os.listdir(folder_path[i])
#     # print(file_list)
#     for j in range(len(file_list)):
#         sh.move(folder_path[i]+'/'+file_list[j], '/home/sunwoo/Documents/Dataset/val_image/'+file_list[j])
#     #     print(folder_path[i]+'/'+file_list[j])
#     #     print('/home/sunwoo/Documents/Dataset/val_image/'+file_list[j])
#
#
# print('file move success.')

import os
import shutil
import time


def read_all_file(path):
    output = os.listdir(path)
    file_list = []

    for i in output:
        if os.path.isdir(path + "/" + i):
            file_list.extend(read_all_file(path + "/" + i))
        elif os.path.isfile(path + "/" + i):
            file_list.append(path + "/" + i)

    return file_list


def copy_all_file(file_list, new_path):
    for src_path in file_list:
        file = src_path.split("/")[-1]
        shutil.copyfile(src_path, new_path + "/" + file)
        print("파일 {} 작업 완료".format(file))  # 작업한 파일명 출력


start_time = time.time()  # 작업 시작 시간

src_path = "/home/sw/바탕화면/crop/train_2"  # 기존 폴더 경로
new_path = "/home/sw/바탕화면/crop/train_image"  # 옮길 폴더 경로

file_list = read_all_file(src_path)
copy_all_file(file_list, new_path)

print("=" * 40)
print("러닝 타임 : {}".format(time.time() - start_time))  # 총 소요시간 계산