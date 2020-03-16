from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, cv2
from tqdm import tqdm
from random import shuffle

image_dir = 'train_image_default/'
test_dir = 'test_image_default/'
model_dir = 'default.model'
npz_dir = 'default.npz'

batch_size = 1000

image_size = 128
class_names = {0:'cat',1:'dog',2:'flower'}

image_data = []
label_data = []
###################################
classify_images = []

with np.load(npz_dir) as data:
    train_images = data['image_data']
    train_labels = data['label_data']
    test_images = data['image_data']
    test_labels = data['label_data']

train_images = tf.keras.utils.normalize(train_images, axis = 1) #makes data value 0-1
test_images = tf.keras.utils.normalize(test_images, axis = 1) #makes data value 0-1

shuffle(test_images)

"""for image in os.listdir(test_dir):
    image = cv2.imread(test_dir + "/" + image)
    image = cv2.resize(image, (image_size, image_size))
    image = np.asarray(image)
    classify_images.append(image)"""

predict = 0
for images in test_images:
    plt.imshow(test_images[predict], cmap = plt.cm.binary)
    plt.show()

    model = tf.keras.models.load_model(model_dir)
    predictions = model.predict([test_images])
    predict_class = class_names[np.argmax(predictions[predict])]
    print("\nprediction: ", predict_class)
    predict += 1
