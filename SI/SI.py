import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import kornia
import torch

"""
Leverages cv2.Sobel to calculate sobelx and sobely matrix of given image
"""
def calculate_sobel(img):
    scale = 1
    delta = 0
    # calculate sobel x
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # calculate soble y
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    # return sobelx, sobely
    return abs_grad_x, abs_grad_y
"""
Calculate SI_r on each pixel with corresponding sobelx and sobely value
"""
def calculate_SI_pixel(sobelx_pixel, sobely_pixel):
    return math.sqrt(math.pow(sobelx_pixel, 2) + math.pow(sobely_pixel, 2))

"""
Implement mean of SI calculation
"""
def SI_mean(sobelx, sobely):
    sum = 0
    # flatten numpy.ndarray
    sobelx.flatten()
    sobely.flatten()
    print(len(sobelx))
    print(len(sobelx[0]))

    # use numpy api

    for i in range(len(sobelx) - 1):
        for j in range(len(sobelx[0]) - 1):
            sum += calculate_SI_pixel(sobelx[i][j], sobely[i][j])
            j += 1
        i += 1
    sum /= len(sobelx) * len(sobelx[0])
    return sum

def SI_rms(sobelx, sobely):
    sum = 0
    # flatten numpy.ndarray
    sobelx.flatten()
    sobely.flatten()
    for i in range(len(sobelx) - 1):
        for j in range(len(sobelx[0]) - 1):
            # sum of square of SI_r
            sum += math.pow(calculate_SI_pixel(sobelx[i][j], sobely[i][j]), 2)
            j += 1
        i += 1
    sum /= len(sobelx) * len(sobelx[0])
    sum = math.sqrt(sum)
    return sum

"""
What if (\sum SI_r)^2 is way larger than \sum SI_r^2?
"""
def SI_stdev(sobelx, sobely):
    si_mean_square = math.pow(SI_mean(sobelx, sobely), 2)
    sum = 0
    sobelx.flatten()
    sobely.flatten()
    for i in range(len(sobelx) - 1):
        for j in range(len(sobelx[0]) - 1):
            # sum of square of SI_r
            sum += math.pow(calculate_SI_pixel(sobelx[i][j], sobely[i][j]), 2)
            j += 1
        i += 1
    print(sum)
    sum /= len(sobelx) * len(sobelx[0])
    print(sum)
    sum = sum - si_mean_square
    print(sum)
    print(si_mean_square)
    sum = math.sqrt(sum)
    return sum
"""
Wrapper of using SI calculation
"""
def SI_combined(img):
    print(SI_mean(calculate_sobel(img)[0], calculate_sobel(img)[1]))
    print(SI_rms(calculate_sobel(img)[0], calculate_sobel(img)[1]))
    print(SI_stdev(calculate_sobel(img)[0], calculate_sobel(img)[1]))
    return SI_mean(calculate_sobel(img)[0], calculate_sobel(img)[1])


def show(sobelx, sobely, img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    grad = cv2.addWeighted(calculate_sobel(img)[0], 0.5, calculate_sobel(img)[1], 0.5, 0)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(grad, cmap='gray')
    plt.title('addWeight'), plt.xticks([]), plt.yticks([])

    plt.show()


def image_complexity(input_images):
  gray_images = kornia.color.rgb_to_grayscale(input_images)
  sobeled_images = kornia.filters.sobel(gray_images)
  complexity = sobeled_images.mean(dim=(2,3))
  complexity = complexity.squeeze()
  return complexity

if __name__ == '__main__':
    img = cv2.imread('/Users/alex/Desktop/Rice/Sophomore2/Research/AdaSeg/pytorch-cifar-master/SI/dave.png',
                     cv2.IMREAD_GRAYSCALE)
    # img = cv2.GaussianBlur(img, (3, 3), 0)

    # print(image_complexity(torch.from_numpy(img)))
    # show(calculate_sobel(img)[0], calculate_sobel(img)[1], img)
