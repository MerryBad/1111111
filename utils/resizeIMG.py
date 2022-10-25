from PIL import Image
import os
import pandas as pd

image_path = r'D:/시설 작물 개체 이미지/Validation/image/'  # 원본 이미지 경로
data_path = r'E:/test/'  # 저장할 이미지 경로
# data = pd.read_csv('data_label.csv', header=0)
# is_it = data['pl_Name'] == '상추'
# data_list = data[is_it]['file_Name'].to_list()
data_list = os.listdir(image_path)
# print(image_path)
# print(data_path)
# print(data_list)
# print(len(data_list))
count = 0
# # 모든 이미지 resize 후 저장하기
for name in data_list:
    # 이미지 열기
    if name in os.listdir(data_path):
        pass
    else:
        try:
            im = Image.open(image_path + name)

            # 이미지 resize
            im = im.resize((256, 256))

            # 이미지 JPG로 저장
            im = im.convert('RGB')

            names = data_path + name
            im.save(names[:-4] + '.jpg')
        except:
            print(image_path + name)
            exit()
    count += 1
    print(count, '/', len(data_list))

print('end ::: ')

# data = pd.read_csv('E:/test_label1.csv', header=0)
#
# for i in range(len(data)):
#     if data.iloc[i, 3] == '멜론 영양생장기' or data.iloc[i, 3] == '토마토(일반) 영양생장':
#         try:
#             os.remove(path_image+data.iloc[i, 0])
#             os.remove(path_label+data.iloc[i, 0][:-4]+'.json')
#         except:
#             pass
