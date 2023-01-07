import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
# print(net)

img = cv2.imread("data/blackman.jpeg", -1)
img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))  # convert the img to tensor
img_tensor = img_tensor.unsqueeze(0)  # normalization
img_tensor = img_tensor.to(dtype=torch.float32)  # change the type of data
output = net(img_tensor)  # make prediction
if net.n_classes > 1:
    mask = output.argmax(dim=1)  # 在预测维度上寻找最大值填入mask
else:
    mask = torch.sigmoid(output) > 0.5
outcome = mask[0].long().squeeze().numpy()
plt.imshow(outcome)
plt.show()

