import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import cv2
import numpy as np
import os
import time

# print(torchvision.__version__)
# print(cv2.__version__)
# print(np.__version__)

"""
preprocess() 함수 : 이미지를 Tensor로 바꿔주고, Normalization
@parameters
img : 전처리할 대상 이미지
"""


def preprocess(img):
    # print(f"imput img shape : {img.shape}")
    trf = T.Compose(
        [
            T.ToTensor(),
            # T.CenterCrop(IMG_SIZE),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_img = trf(img).unsqueeze(0)
    # GPU 사용 가능시, GPU 메모리에 적재
    return input_img


"""
seg_map() 함수 : 배경과 사람의 class에 채색
@parameters
img : 전처리할 대상 이미지
COLORS : 배경과 사람 색칠할 색상 정보
n_classes : COCO dataset 기준 21개의 class가 존재
"""


def seg_map(img, COLORS, n_classes=21):
    color_idx = 0
    # 색상을 담을 변수
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in (0, 15):  # 사람과 배경의 경우만 색칠
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


def shape_estimation(filename, img_size=480):
    # 출력 파일 경로 지정
    t = time.time()
    output_path = os.path.dirname(filename)
    output_filename = os.path.basename(filename).split(".")[0] + "_shape.mp4"
    output_filename = os.path.join(output_path, output_filename)

    # 모델 가져오기
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능 여부

    print("Using.." + device)
    if device == "cpu":
        print(f"Can use CPUs -> {torch.get_num_threads()}")
        torch.set_num_threads(torch.get_num_threads())
    COLORS = np.array([(255, 255, 255), (192, 128, 128)])  # background -> 하얀색  # person
    IMG_SIZE = img_size

    # Video 가져오기
    cap = cv2.VideoCapture(filename)
    assert cap.isOpened()
    width = int(cap.get(3))
    height = int(cap.get(4))
    # nof_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # frame_count = 0

    # video writer 구성(MP4), image shape (height, width, channel)
    video_format = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        output_filename,
        video_format,
        cap.get(cv2.CAP_PROP_FPS),
        (IMG_SIZE, int(height * IMG_SIZE / width)),
    )
    # print(f"Video resolution : {img.shape}")

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        # 읽어온 video 해상도 조정
        img = cv2.resize(img, (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 모델에 맞게 data 전처리 후 video write
        input_tensor = preprocess(img)
        if device == "cuda":  # CUDA에 모델과 data 올림
            input_tensor = input_tensor.to(device)
            model = model.to(device)
        out = model(input_tensor)["out"][0]  # out.shape (21,height, width)
        out = out.argmax(0).detach().cpu().numpy()  # 사물 detect
        out = seg_map(out, COLORS)  # 채색
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out_video.write(out)
        # print("\r Frame : ", int(nof_frame), frame_count)
        # frame_count = frame_count + 1
    cap.release()
    out_video.release()
    print(f"total time : {(time.time() - t)/60.} min.")  # 경과시간


# test code.
# if __name__ == "__main__":
#     shape_estimation('./testvid.mp4')
