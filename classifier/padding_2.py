import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def draw_images(generator, x, dir_name, index):
    # 出力ファイルの設定
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=save_name, save_format='jpeg')
    
    # 1つの入力画像から3枚分拡張する事を指定
    for i in range(30):
        bach = g.next()

# 出力先フォルダの指定
output_dir = "./img2/demon"
#output_dir = "./img2/human"

if not(os.path.exists(output_dir)):
    os.mkdir(output_dir)

#拡張する画像の読み込み
images = glob.glob(os.path.join("./img/demon_resize","*.jpg"))
#images = glob.glob(os.path.join("./img/human_resize","*.jpg"))


#ImageDataGeneratorを定義
datagen = ImageDataGenerator(rotation_range = 30,
                             width_shift_range = 20,
                             shear_range = 0,
                             height_shift_range = 0,
                             zoom_range = 0.1,
                             horizontal_flip = True,
                             fill_mode = "nearest",
                             channel_shift_range = 40)

#読み込んだ画像を順に拡張
for i in range(len(images)):
    img = load_img(images[i])
    img = img.resize((128,128))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)
    draw_images(datagen,x,output_dir,i)
