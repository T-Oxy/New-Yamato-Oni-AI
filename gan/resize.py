# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np

#パラメータ
#======================================
#画像を保存してあるフォルダ名
f = 'all/'
#リサイズした画像を保存するフォルダ名
f_resize = 'all_resize/'
#リサイズ後のサイズ
size = 128
#======================================

#処理
#======================================
if not os.path.isdir(f_resize):
    os.makedirs(f_resize)
files = os.listdir(f)
for file in files:
    img = Image.open(f + file).convert("RGBA"); img.close

    tmp = np.array(img)

    mask = tmp[:,:,3] < 240
    tmp[mask, 0] = 255
    tmp[mask, 1] = 255
    tmp[mask, 2] = 255

    img = Image.fromarray(tmp[:,:,0:3])

    width, height = img.size
    if width == height:
        tmp = img
    elif width > height:
        tmp = Image.new('RGB', (width, width), (255, 255, 255))
        tmp.paste(img, (0, (width - height) // 2))
    else:
        tmp = Image.new('RGB', (height, height), (255, 255, 255))
        tmp.paste(img, ((height - width) // 2, 0))
    img_resize = tmp.resize((size, size), Image.BICUBIC)
    img_resize.save(f_resize + file)
    print("リサイズ完了")
print()
#======================================
