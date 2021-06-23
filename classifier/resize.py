import os
import glob
from PIL import Image

name = ['demon','human']
a = 0
for a in range (2):
    files = glob.glob(name[a] + '/*.jpg')
    b = 0
    for f in files:
        b += 1
        img = Image.open(f)
        img_resize = img.resize((128, 128))
        ftitle, fext = os.path.splitext(f)
        img_resize.save(name[a] + '_resize/' + str(b) + fext)

    a += 1
