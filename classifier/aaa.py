from PIL import Image
import os, glob
import numpy as np
import random, math

# 画像が保存されているディレクトリのパス
root_dir = "./img_test"
# 画像が保存されているフォルダ名
categories = ["demon_test","human_test"]

X = [] # 画像データ
Y = [] # ラベルデータ

# フォルダごとに分けられたファイルを収集
#（categoriesのidxと、画像のファイルパスが紐づいたリストを生成）
allfiles = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpeg")
    for f in files:
        allfiles.append((idx, f))

for cat, fname in allfiles:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((128,128))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

x = np.array(X)
y = np.array(Y)

np.save("./test/d_or_h_data_test_X.npy", x)
np.save("./test/d_or_h_data_test_Y.npy", y)

# モデルの精度を測る
eval_X = np.load("./test/d_or_h_data_test_X.npy")
eval_Y = np.load("./test/d_or_h_data_test_Y.npy")

from keras.utils import np_utils
# データの整数値を2値クラスの行列に変換
eval_Y = np_utils.to_categorical(eval_Y, 2)

score = model.model.evaluate(x=eval_X, y=eval_Y)

print('loss=', score[0])
print('accuracy=', score[1])

from sklearn.metrics import confusion_matrix

eval_X = np.load("./test/d_or_h_data_test_X.npy")
eval_Y = np.load("./test/d_or_h_data_test_Y.npy")

predict_classes = model.model.predict_classes(eval_X)
print(confusion_matrix(eval_Y, predict_classes))
