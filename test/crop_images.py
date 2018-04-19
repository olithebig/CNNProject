#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:34:49 2017

@author: work
"""
import glob
from PIL import Image
#img = Image.open("2.jpg")
#area = (1100, 0, 1920, 1080)
#cropped_img = img.crop(area)
#cropped_img.show()

i=0
images=glob.glob("*.jpg")
for image in images:
    img = Image.open(image)
    area = (1050, 0, 1920, 1080)
    cropped_img = img.crop(area)
    filename = repr(i) + ".jpg"
    cropped_img.save(filename)
    i += 1
