# import better_exceptions
import os
import glob
from PIL import Image

files = glob.glob('all/*.jpg')
a = 0
for f in files:
    a += 1
    img = Image.open(f)
    img_resize = img.resize((128, 128))
    ftitle, fext = os.path.splitext(f)
    img_resize.save('all_resize/' + str(a) + '_(128x128)' + fext)
#    print(a, end=", ")
