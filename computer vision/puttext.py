import cv2
img = cv2.imread('originals/qb.jpg')


cv2.putText(img, 'Serenity Now', (10,500), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255),2, cv2.LINE_AA)
cv2.imwrite('results/puttext.jpg', img)
