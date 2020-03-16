from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from tqdm import tqdm
from random import shuffle

image_dir = 'train_image_default/'
test_dir = 'test_image_default/'
model_dir = 'default.model'
npz_dir = 'default.npz'

train_size = 500 #how many images trained per folder

image_size = 128
class_names = {0:'cat',1:'dog',2:'flower'}

def create_training_data():
    image_data = []
    label_data = []

    for folder in os.listdir(image_dir):
        counter = 0
        for image in tqdm(os.listdir(image_dir + folder)):
            try:
                counter += 1
                image = cv2.imread(image_dir + folder + "/" + image)
                image = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.asarray(image)
                image_data.append(image)
                label_data.append(int(folder))
                if counter > train_size:
                    break
            except:
                pass

    #print(image_data[0], "\n", label_data[0])
    np.savez(npz_dir, image_data = image_data, label_data = label_data)

def training_model():
    with np.load(npz_dir) as data:
        train_images = data['image_data']
        train_labels = data['label_data']
        test_images = data['image_data']
        test_labels = data['label_data']

    train_images = tf.keras.utils.normalize(train_images, axis = 1) #makes data value 0-1
    test_images = tf.keras.utils.normalize(test_images, axis = 1) #makes data value 0-1

    """train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_images), tf.constant(train_labels)))
    test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_images), tf.constant(test_labels)))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 1000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)"""

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (image_size, image_size, 3))) #flattens image, 3:rgb/1:gray
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #amount of neurons
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #probability distr

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x = train_images, y = train_labels, epochs = 50)
    #model.fit(train_dataset, epochs = 100)

    val_loss, val_acc = model.evaluate(test_images, test_labels)
    print("\n", val_loss, val_acc)
    model.save(model_dir)

def image_classifier():
    classify_images = []

    for image in os.listdir(test_dir):
        image = cv2.imread(test_dir + "/" + image)
        image = cv2.resize(image, (image_size, image_size))
        image = np.asarray(image)
        classify_images.append(image)

    predict = 0
    for images in classify_images:
        plt.imshow(classify_images[predict], cmap = plt.cm.binary)
        plt.show()
        model = tf.keras.models.load_model(model_dir)
        predictions = model.predict([classify_images])
        predict_class = class_names[np.argmax(predictions[predict])]
        print("\nprediction: ", predict_class)
        predict += 1

def main():
    flag = 1
    while flag == 1:
        run = int(input("\ncreate training data : 1\nrun training model   : 2\nclassify images      : 3\nquit                 : 0\n"))
        if run == 1:
            create_training_data()
        elif run == 2:
            training_model()
        elif run == 3:
            image_classifier()
        elif run == 0:
            flag = 0
        else:
            print("error: try again")
main()
