import numpy as np
import cv2

im_one = cv2.imread('originals/hillary.jpg')
im_two = cv2.imread('originals/trump.jpg')
im_one = cv2.resize(im_one, (275,183))
print(f'Hillary shape = {im_one.shape}')
print(f'Trump shape = {im_two.shape}')
dst = cv2.addWeighted(im_one, 0.3, im_two, 0.7, 0)

cv2.imwrite('results/blended2.jpg',dst)
