# from PIL import Image, ExifTags, ImageDraw
# img = Image.open('originals/qb.jpg').convert('LA')
# img.save('results/grayscale.png')

import cv2

img = cv2.imread('originals/hillary.jpg')
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('results/grayscalecv2.jpg', grayscale)
