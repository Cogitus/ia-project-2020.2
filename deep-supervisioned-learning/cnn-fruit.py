# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# talvez tentar melhorar o código usando isso: https://keras.io/api/preprocessing/image/

# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/

import os
import glob
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn.metrics import classification_report


# dimensions of our images.
img_width, img_height = 320, 258

path = os.getcwd()
train_data_dir = path+'/deep-supervisioned-learning/data/train'
validation_data_dir = path+'/deep-supervisioned-learning/data/validation'
nb_train_samples = 1800
nb_validation_samples = 200


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    # input_shape = (img_height, img_width, 3)

EPOCHS = 4
batch_size = 32
training = False
mymodel = 3
file_name = "cnn_fullsizeimg_model"+str(mymodel)+"_"+str(EPOCHS)+"epochs_"+str(batch_size)+"batch_datagen.h5"

if(mymodel == 1):
    model = Sequential()
    model.add(Conv2D(100, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(60, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(3, activation='softmax'))

if(mymodel == 2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

if(mymodel == 3):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

if(mymodel == 4):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(3, activation='softmax'))

if(training):
    print(model.summary())

    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # this is the augmentation configuration we will use for training
    # train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                                        height_shift_range=0.2, shear_range=0.15, horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size)

    model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

    model.save_weights('deep-supervisioned-learning/'+file_name)

if(not training):
    model.load_weights('deep-supervisioned-learning/'+file_name)
    #model.load_weights('deep-supervisioned-learning/cnn_fullsizeimg_model2_fitgenerator_2epochs_32batch.h5')

    # from matplotlib import pyplot as plt
    # plt.imshow(np_im, interpolation='nearest')
    # plt.show() 

    
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    next_batch = validation_generator.next()
    imgs = next_batch[0]
    labels = next_batch[1]

    for i in range(18):
        next_batch = validation_generator.next()
        imgs = np.append(imgs, next_batch[0], axis=0)
        labels = np.append(labels, next_batch[1], axis=0)

    predictions = model.predict(imgs)

    pred_argmax = np.argmax(predictions, axis=1)
    labels_argmax = np.argmax(labels, axis=1)
    print(classification_report(labels_argmax, pred_argmax, target_names=["carambola", "pear", "plum"]))
    

    """
    test_imgs_carambola = []
    test_imgs_pear = []
    test_imgs_plum = []

    carambola_set = glob.glob (validation_data_dir +'/carambola/*.png')
    pear_set = glob.glob (validation_data_dir +'/pear/*.png')
    plum_set = glob.glob (validation_data_dir +'/plum/*.png')

    for img_path in carambola_set:
        image = Image.open(img_path)
        np_im = np.array(image)
        test_imgs_carambola.append(np_im)

    for img_path in pear_set:
        image = Image.open(img_path)
        np_im = np.array(image)
        test_imgs_pear.append(np_im)

    for img_path in plum_set:
        image = Image.open(img_path)
        np_im = np.array(image)
        test_imgs_plum.append(np_im)

    np_test_imgs_carambola = np.array(test_imgs_carambola)
    np_test_imgs_pear = np.array(test_imgs_pear)
    np_test_imgs_plum = np.array(test_imgs_plum)

    np_test_imgs_carambola = np_test_imgs_carambola/255
    np_test_imgs_pear = np_test_imgs_pear/255
    np_test_imgs_plum = np_test_imgs_plum/255

    all_test_imgs = np.concatenate([np_test_imgs_carambola, np_test_imgs_pear])
    all_test_imgs = np.concatenate([all_test_imgs, np_test_imgs_plum])

    predictions = model.predict(all_test_imgs)
    pred_argmax = np.argmax(predictions, axis=1)

    carambola_labels = np.full(200, 0)
    pear_labels = np.full(200, 1)
    plum_labels = np.full(200, 2)
    labels =  np.concatenate([carambola_labels, pear_labels, plum_labels])

    print(classification_report(labels, pred_argmax, target_names=["carambola", "pear", "plum"]))
    """
