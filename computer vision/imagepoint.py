from PIL import Image, ExifTags, ImageDraw
img = Image.open('originals/qb.jpg')

lookupTable = list(range(255,-1,-1)) + list(range(255,-1,-1)) + list(range(255,-1,-1))
newImg = img.point(lookupTable)

newImg.save('results/imagepoint.jpg')
