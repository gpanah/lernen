import cv2
import numpy as np
img = cv2.imread('originals/checkers.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blocksize=2
ksize=3
k=0.01
 
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,blocksize,ksize,k)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[255,0,0]

cv2.imwrite(f'results/corners-block{blocksize}-ksize{ksize}-k{k}.jpg', img)
