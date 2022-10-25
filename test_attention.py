import datetime
import os.path
import time
import warnings
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations
<<<<<<< HEAD
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image
=======
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
>>>>>>> 8a4ccccf1449c53c33e3f49fb764fb52584aae44
import numpy as np
from PIL import Image
import torch
import torchvision
from models import vgg_attention, vgg_attention_major, vgg_attention_minor

warnings.filterwarnings(action='ignore')

NUM_CLS_MAJOR = 15
NUM_CLS_MINOR = 12
major_name = ['가지', '거베라', '고추(청양)', '국화', '딸기', '멜론', '부추', '상추', '안개', '애플망고', '애호박', '오이', '장미',
              '토마토(일반)', '파프리카(볼로키)']
minor_name = ['개화기', '과비대성숙기', '과실비대기', '생육기', '수확기', '영양생장', '절화기', '정식기', '착과기', '착색기', '화아발달기', '화아분화기']

model_path = 'E:/web/static/vgg_mm16_add_attention-141-best.pth'
mean = 0.4255
std = 0.2234
transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=mean, std=std, p=1.0),
        albumentations.pytorch.transforms.ToTensorV2()
    ])


# def grad_major(img, path):
#     model = vgg_attention_major.vgg16_bn(12, 15)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     target_layers = [model.major_branch[-2]]
#     rgb_img = Image.open(path).convert('RGB')
#     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
#     grayscale_cam = cam(input_tensor=img.float())
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#     Image.fromarray(visualization, 'RGB').save('E:/web/static/major.png')
#
#
# def grad_minor(img, path):
#     model = vgg_attention_minor.vgg16_bn(12, 15)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     target_layers = [model.minor_branch[-2]]
#     rgb_img = Image.open(path).convert('RGB')
#     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
#     grayscale_cam = cam(input_tensor=img.float())
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#     Image.fromarray(visualization, 'RGB').save('E:/web/static/minor.png')



model_minor = vgg_attention_minor.vgg16_bn(12, 15)
model_minor.load_state_dict(torch.load(model_path))
model_minor.eval()

target_minor_layers = [model_minor.minor_branch[-2]]
model_major = vgg_attention_major.vgg16_bn(12, 15)
model_major.load_state_dict(torch.load(model_path))
model_major.eval()

target_major_layers = [model_major.major_branch[-2]]


def grad_major(path):
    image_name = time.strftime("%Y%m%d-%H%M%S")

    rgb_img = Image.open(path).convert('RGB')
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()

    cam = GradCAM(model=model_major, target_layers=target_major_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization, 'RGB').save('E:/web/static/IMG/'+image_name+'major.jpg')
    name_1 = 'IMG/'+image_name+'major.jpg'
    return name_1

def grad_minor(path):
    image_name = time.strftime("%Y%m%d-%H%M%S")

    rgb_img = Image.open(path).convert('RGB')
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()

    cam = GradCAM(model=model_minor, target_layers=target_minor_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization, 'RGB').save('E:/web/static/IMG/'+image_name+'_minor.jpg')
    name_2 = 'IMG/'+image_name+'_minor.jpg'
    return name_2

# def pred(path):
#     model = vgg_attention.vgg16_bn(12, 15)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     rgb_img = Image.open(path).convert('RGB')
#     rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
#     input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()
#
#     major, minor = model(input_tensor)
#     _, top1_preds_major = major.max(1)
#     _, top3_preds_major = major.topk(3, 1, largest=True, sorted=True)
#     _, top1_preds_minor = minor.max(1)
#     _, top3_preds_minor = minor.topk(3, 1, largest=True, sorted=True)
#
#     print('작물 종류 : ', major_name[major])
#     print('생육 단계 : ', minor_name[minor])


# image_path = '/home/sunwoo/Desktop/data/test_image/19_20201208_1548725.jpg'
# grad_major(image_path)
# grad_minor(image_path)
# pred(image_path)
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
# grayscale_cam = cam(input_tensor=input_tensor)
# grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
# # Image.fromarray(visualization, 'RGB').save('grad_cam/'+datetime.datetime.now().strftime('%H%M%S')+'_major.png')
# Image.fromarray(visualization, 'RGB').save('grad_cam/' + datetime.datetime.now().strftime('%H%M%S') + '_minor.png')
