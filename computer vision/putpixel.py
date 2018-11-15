from PIL import Image, ExifTags, ImageDraw
img = Image.open('originals/qb.jpg')

x = 0
while x < 2000:
  y = 0
  while y < 1000:
    img.putpixel((x,y),(255,0,0))
    y += 1
  x += 7

img.save('results/putpixel.jpg', 'JPEG')
