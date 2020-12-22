import os
import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(-1, 28, 28, 1).astype(np.float32)
    im = im / 255.0
    return im

model_path = '.\my_model.h5'
mnist_model = tf.keras.models.load_model(model_path)
result_path='number_test.txt'
test_data_path='.\test_data'
resultfile=open(result_path,'a')

for root, dirs, files in os.walk(test_data_path):
    for f in files:
        img = load_image(os.path.join(root,f))
        pred_results = mnist_model.predict(img)
        lab = np.argsort(pred_results)
        resultfile.write(f+' %d\n'%lab[0][-1])
        print(f+' %d'%lab[0][-1])
resultfile.close()



