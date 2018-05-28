import glob

from PIL import Image

i = 0
images = glob.glob("*.jpg")
# Crops all images in directory, where it is executed
for image in images:
    img = Image.open(image)
    area = (300, 0, 880, 720)
    cropped_img = img.crop(area)
    filename = repr(i) + ".jpg"
    cropped_img.save(filename)
    i += 1
