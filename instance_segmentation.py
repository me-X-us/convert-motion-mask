import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import numpy as np
import cv2
import time
import os


"""
preprocess : Data preprocessing method
@params
img : image to process
"""


def preprocess(img):
    trf = T.Compose([T.ToTensor()])
    input_img = trf(img).unsqueeze(0)
    return input_img


"""
seg_map : make segmentation image
@params
img : input image
COLORS : background and human shape colors
"""


def seg_map(img, COLORS):
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    idx = img >= 0.5
    rgb[idx] = COLORS[1]
    rgb[idx == False] = COLORS[0]
    return rgb


"""
shape_estimation
OUTPUT Video :  name -> input_shape.mp4, 
                codec -> vp09, 
                cotainer -> mp4
return : video fps (str type)
@params
filename : Video data's name
img_size : resoluion, default = 480
"""


def shape_estimation(filename, img_size=480):
    t = time.time()
    output_path = os.path.dirname(filename)
    output_filename = os.path.basename(filename).split(".")[0] + "_shape.mp4"
    output_filename = os.path.join(output_path, output_filename)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using.." + device)
    if device == "cpu":
        print(f"Can use CPUs -> {torch.get_num_threads()}")
        torch.set_num_threads(torch.get_num_threads())
    COLORS = np.array(
        [(255, 255, 255), (0, 255, 255)]
    )  # background -> white  # person -> cyan
    IMG_SIZE = img_size

    # read video data
    cap = cv2.VideoCapture(filename)
    assert cap.isOpened()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # fps, total frame, video_length(float type)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # nof_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # video_length = nof_frame / video_fps
    # frame_count = 0

    # video writer (MP4), image shape (height, width, channel)
    video_format = cv2.VideoWriter_fourcc(*"vp09")
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
        # video resize
        img = cv2.resize(img, (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_tensor = preprocess(img)
        if device == "cuda":
            input_tensor = input_tensor.to(device)
            model = model.to(device)
        out = model(input_tensor)[0]["masks"][0]  # out.shape (1,height, width)
        out = out.squeeze().detach().cpu().numpy()

        out = seg_map(out, COLORS)  # color map
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out_video.write(out)
        # print("\r Frame : ", int(nof_frame), frame_count)
        # frame_count = frame_count + 1
    cap.release()
    out_video.release()
    print(f"total time : {(time.time() - t)/60.} min.")  # total time

    return str(video_fps)


# test code
# if __name__ == "__main__":
#     shape_estimation('./testvid.mp4')
