import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import cv2
import numpy as np
# print(torch.__version__)
# print(torchvision.__version__)
# print(cv2.__version__)
# print(np.__version__)

"""
preprocess() 함수 : 이미지를 Tensor로 바꿔주고, Normalization
@parameters
img : 전처리할 대상 이미지
"""
def preprocess(img):
    print(f"imput img shape : {img.shape}")
    trf = T.Compose([
                    T.ToTensor(),
                    # T.CenterCrop(IMG_SIZE),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225]
                    ),
    ])
    input_img = trf(img).unsqueeze(0)
    # GPU 사용 가능시, GPU 메모리에 적재
    if device == 'cuda':
        input_img = input_img.to(device)
    return input_img


"""
seg_map() 함수 : 배경과 사람의 class에 채색
@parameters
img : 전처리할 대상 이미지
n_classes : COCO dataset 기준 21개의 class가 존재
"""
def seg_map(img, n_classes=21):
    color_idx = 0
    # 색상을 닮을 변수
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    print(rgb.shape)
    for i in (0,15): # 사람과 배경의 경우만 색칠
        idx = img == i
        rgb[idx] = COLORS[color_idx]
        color_idx = color_idx + 1

    return rgb


"""
shape_estimation() 함수
@parameters
filename : shape 추출하기 위한 파일 이름
size : 변경할 파일의 사이즈 (default: 파일의 기본 사이즈)
"""
def shape_estimation(filename,img_size=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using.."+device)
    # 원하는 색상 지정하는 부분
    COLORS = np.array([
    (255, 255, 255), # background -> 하얀색
    (192, 128, 128), # person
    ])
    IMG_SIZE = img_size
    count = 0
    cap = cv2.VideoCapture(filename)
    assert cap.isOpened()

    nof_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, img = cap.read()
    print(f"Video resolution : {img.shape}")



# test code.
if __name__ == "__main__":
    shape_estimation('./testvid.mp4')