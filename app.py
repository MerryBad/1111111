import io
import json
import os
import time
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from models import vgg_attention
import torch
import pandas as pd
<<<<<<< HEAD
#from test_attention import *
=======
from test_attention import *
>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/H2GTRM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

app = Flask(__name__)
<<<<<<< HEAD
model_path = 'C:/Users/sinb1/1111111/static/vgg_mm16_add_attention-141-best.pth'
model_pred = vgg_attention.vgg16_bn(12, 15)
model_pred.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
=======
model_path = 'E:/web/static/vgg_mm16_add_attention-141-best.pth'
model_pred = vgg_attention.vgg16_bn(12, 15)
model_pred.load_state_dict(torch.load(model_path))
>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44
model_pred.eval()  # autograd 끄기

# TODO          0       1         2         3       4     5      6       7
major_name = ['가지', '거베라', '고추(청양)', '국화', '딸기', '멜론', '부추', '상추',
              '안개', '애플망고', '애호박', '오이', '장미', '토마토(일반)', '파프리카(볼로키)']
#               8       9        10      11     12        13              14

# TODO           0          1           2          3       4        5         6       7       8         9         10         11
minor_name = ['개화기', '과비대성숙기', '과실비대기', '생육기', '수확기', '영양생장', '절화기', '정식기', '착과기', '착색기', '화아발달기', '화아분화기']

# TODO        0         1           2              3                 4            5           6          7
merge = [[0, 4, 5], [4, 5, 7], [0, 4, 3, 7], [7, 3, 10, 11, 5], [0, 11, 4], [2, 7, 5, 0], [5, 3], [4, 3, 8, 5],
         [5, 0, 4], [11, 0], [5, 8, 4, 3], [0, 5], [5, 6], [3, 4, 8, 7, 15], [8, 9, 1, 4, 2]]


# TODO        8          9           10,      11      12         13                 14


# Transform input into the form our model expects
def transform_image(infile):
    mean = 0.4255
    std = 0.2234
    input_transforms = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)  # 이미지 파일 열기
    timg = my_transforms(image)  # PIL 이미지를 적절한 모양의 PyTorch 텐서로 변환
    timg.unsqueeze_(0)  # PyTorch 모델은 배치 입력을 예상하므로 1짜리 배치를 만듦
    return timg


# Get a prediction
def get_prediction(input_tensor):
    start_time = time.time()
    major, minor = model_pred(input_tensor)
    end_time = time.time()
    log_time = end_time - start_time
    log_time = '추론시간 : ' + str(log_time)[:5] + 's'
    _, top1_preds_major = major.max(1)
    _, top3_preds_major = major.topk(3, 1, largest=True, sorted=True)
    _, top1_preds_minor = minor.max(1)
    _, top3_preds_minor = minor.topk(3, 1, largest=True, sorted=True)
    count = 0

    for i in range(3):
        print(top3_preds_minor[0][i], merge[top1_preds_major[0]])
        if top3_preds_minor[0][i] in merge[top1_preds_major[0]]:  # 작물의 생육 단계에 포함되있으면
            break  # 패스
        else:
            count += 1
            print(str(count) + "th predicted Growth stage is't in crop type")

    if count == 3:
        print('Prediction fail')
        top1_preds_minor = top3_preds_minor[0][count - 1]
    m = torch.nn.Softmax(dim=1)
    major = m(major)
    top3_major = []
    top3_major_name = []
    for i in range(3):
        top3_major.append(major.detach().numpy()[0][top3_preds_major[0][i]] * 100)
        top3_major_name.append(major_name[top3_preds_major[0][i]])
    minor = m(minor)
    top3_minor = []
    top3_minor_name = []

    for i in range(3):
        top3_minor.append(minor.detach().numpy()[0][top3_preds_minor[0][i]] * 100)
        top3_minor_name.append(minor_name[top3_preds_minor[0][i]])
    image_name = time.strftime("%Y%m%d-%H%M%S")
    plt.clf()
    plt.bar(top3_major_name, top3_major)
<<<<<<< HEAD
    plt.savefig('C:/Users/sinb1/1111111/static/IMG/' + image_name + 'major.jpg')
    plt.clf()
    plt.bar(top3_minor_name, top3_minor)
    plt.savefig('C:/Users/sinb1/1111111/static/IMG/' + image_name + 'minor.jpg')
    return major_name[top1_preds_major], minor_name[top1_preds_minor], 'IMG/' + image_name + 'major.jpg', 'IMG/' + image_name + 'minor.jpg ', log_time


app.config['UPLOAD_FOLDER'] = 'C:/Users/sinb1/1111111/static/IMG'
=======
    plt.savefig('E:/web/static/IMG/' + image_name + 'major.jpg')
    plt.clf()
    plt.bar(top3_minor_name, top3_minor)
    plt.savefig('E:/web/static/IMG/' + image_name + 'minor.jpg')
    return major_name[top1_preds_major], minor_name[top1_preds_minor], 'IMG/' + image_name + 'major.jpg', 'IMG/' + image_name + 'minor.jpg ', log_time


app.config['UPLOAD_FOLDER'] = 'E:/web/static/IMG'
>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/model')
def model():
    return render_template('model.html')


@app.route('/gradcam_home')
def gradcam_home():
    return render_template('gradcam_home.html')

<<<<<<< HEAD
#
# @app.route('/upload_grad', methods=['POST'])
# def upload_grad():
#     file = request.files['file']
#     filename = file.filename
#     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#     img_src = 'E:/web/static/IMG/' + filename
#
#     img = transform_image(img_src)
#     major, minor, _, _, log_time= get_prediction(img)
#     major = '예측 작물 종류 : ' + major
#     minor = '생육 단계 : ' + minor
#     start_time_gard = time.time()
#     grad_major_image = grad_major(img_src)
#     grad_minor_image = grad_minor(img_src)
#     end_time_grad = time.time()
#     log_time_grad = end_time_grad - start_time_gard
#     log_time_grad = 'Grad-CAM 실행 시간 : ' + str(log_time_grad)[:5] + 's'
#
#     filename = 'IMG/' + filename
#     return render_template('gradcam_home.html', filename=filename, major=major, minor=minor,
#                            grad_major_image=grad_major_image,
#                            grad_minor_image=grad_minor_image, time=log_time, time_grad=log_time_grad)
#
=======

@app.route('/upload_grad', methods=['POST'])
def upload_grad():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img_src = 'E:/web/static/IMG/' + filename

    img = transform_image(img_src)
    major, minor, _, _, log_time= get_prediction(img)
    major = '예측 작물 종류 : ' + major
    minor = '생육 단계 : ' + minor
    start_time_gard = time.time()
    grad_major_image = grad_major(img_src)
    grad_minor_image = grad_minor(img_src)
    end_time_grad = time.time()
    log_time_grad = end_time_grad - start_time_gard
    log_time_grad = 'Grad-CAM 실행 시간 : ' + str(log_time_grad)[:5] + 's'

    filename = 'IMG/' + filename
    return render_template('gradcam_home.html', filename=filename, major=major, minor=minor,
                           grad_major_image=grad_major_image,
                           grad_minor_image=grad_minor_image, time=log_time, time_grad=log_time_grad)

>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44

@app.route('/upload_model', methods=['POST'])
def upload_model():
    file = request.files['file']
    filename = file.filename
    print(filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    real_major, real_minor = '', ''
<<<<<<< HEAD
    real_name = pd.read_csv('C:/Users/sinb1/1111111/static/test_label.csv', names=["file_Name", "pl_Name", "pl_Step", 'pl_Name+Step'],
=======
    real_name = pd.read_csv('E:/web/static/test_label.csv', names=["file_Name", "pl_Name", "pl_Step", 'pl_Name+Step'],
>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44
                            header=0)

    find = real_name["file_Name"]
    try:
        find = find[find == filename].index[0]
    except:
        find = False
    if find:
        real_major = '예측 작물 종류 : ' + real_name["pl_Name"][find]
        real_minor = '생육 단계 : ' + real_name["pl_Step"][find]
    else:
        real_major = "테스트 데이터가 아닙니다."

<<<<<<< HEAD
    img_src = 'C:/Users/sinb1/1111111/static/IMG/' + filename
=======
    img_src = 'E:/web/static/IMG/' + filename
>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44
    img = transform_image(img_src)
    pred_major, pred_minor, major_filename, minor_filename, log_time = get_prediction(img)
    pred_major = '실제 작물 종류 : ' + pred_major
    pred_minor = '생육 단계 : ' + pred_minor
    filename = 'IMG/' + filename
    print(major_filename)

    # 그래프 만들기
    return render_template('model.html', filename=filename, pred_major=pred_major, pred_minor=pred_minor,
                           time=log_time, real_major=real_major, real_minor=real_minor, major_filename=major_filename,
                           minor_filename=minor_filename)


if __name__ == '__main__':
<<<<<<< HEAD
    app.run('192.168.56.1', port=5000, debug=True)
=======
    app.run('113.198.138.224', port=5000, debug=True)
>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44
