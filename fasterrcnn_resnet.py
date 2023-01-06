import numpy as np
import torchvision.models.detection
import torch
from torchvision.utils import draw_bounding_boxes
import cv2 as cv

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # load ResNet50 with pretrain parameters
model.eval()  # set model to evaluation

capture = cv.VideoCapture(0)  # open the camera of your computer
cv.namedWindow("camera", 0)  # create a window
while True:
    """grab each frame of image"""
    ret, frame = capture.read()  # frame is ndarray with 3 channels
    if not ret:  # if unable to load image
        break

    frame = np.transpose(frame, (2, 0, 1))  # transpose the axis to C,H,W
    img = torch.from_numpy(frame)  # convert ndarray to tensor
    img_float32 = img.to(torch.float32)/255  # change the type of data, and normalization
    batch = [img_float32]  # process image as a batch
    prediction = model(batch)[0]  # predict
    box = draw_bounding_boxes(img, prediction["boxes"], colors=(0, 0, 255),
                              width=4, font_size=30)  # draw boxes
    im = box.detach().numpy()  # convert to numpy
    im = np.transpose(im, (1, 2, 0))  # transpose the axis to H, W, C
    if cv.waitKey(50) == 27:  # press ESC to quit
        break
    cv.imshow("camera", im)  # display the outcome

cv.waitKey(0)
cv.destroyAllWindows()
