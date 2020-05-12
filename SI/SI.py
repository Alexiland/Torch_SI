# import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import kornia
import torch



def image_complexity(input_images):
  gray_images = kornia.color.rgb_to_grayscale(input_images)
  sobeled_images = kornia.filters.sobel(gray_images)
  complexity = sobeled_images.mean(dim=(2,3))
  complexity = complexity.squeeze()
  return complexity




# if __name__ == '__main__':
#     img = cv2.imread('/Users/alex/Desktop/Rice/Sophomore2/Research/AdaSeg/pytorch-cifar-master/SI/dave.png',
#                      cv2.IMREAD_GRAYSCALE)
    # img = cv2.GaussianBlur(img, (3, 3), 0)

    # print(image_complexity(torch.from_numpy(img)))
    # show(calculate_sobel(img)[0], calculate_sobel(img)[1], img)
