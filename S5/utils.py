iimport matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_images(train_loader):
	batch_data, batch_label = next(iter(train_loader)) 

	fig = plt.figure()

	for i in range(12):
  		plt.subplot(3,4,i+1)
  		plt.tight_layout()
  		plt.imshow(batch_data[i].squeeze(0), cmap='gray')
  		plt.title(batch_label[i].item())
  		plt.xticks([])
  		plt.yticks([])
