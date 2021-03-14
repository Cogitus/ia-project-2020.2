# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# talvez tentar melhorar o código usando isso: https://keras.io/api/preprocessing/image/

import os
import glob
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 320, 258

path = os.getcwd()
train_data_dir = path+'/deep-supervisioned-learning/data/train'
validation_data_dir = path+'/deep-supervisioned-learning/data/validation'
nb_train_samples = 1800
nb_validation_samples = 200
epochs = 8
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

training = True
mymodel = 1
file_name = "cnn_fullsize_model1_fit_3epochs.h5"

if(mymodel == 1):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    print(model.summary())

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


if(training):
    model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    # this is the augmentation configuration we will use for training
    # train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size)

    # trocar fir_generator por fit() 
    # aparentemente fit_generator foi deprecated e só utilizam fit

    model.fit(train_generator, epochs=3, validation_data=validation_generator)

    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=nb_train_samples // batch_size,
    #     epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=nb_validation_samples // batch_size)

    model.save_weights(file_name)

if(not training):
    model.load_weights('deep-supervisioned-learning/'+file_name)

    test_imgs_carambola = []
    test_imgs_pear = []
    test_imgs_plum = []

    carambola_set = glob.glob (validation_data_dir +'/carambola/*.png')
    pear_set = glob.glob (validation_data_dir +'/pear/*.png')
    plum_set = glob.glob (validation_data_dir +'/plum/*.png')

    from matplotlib import pyplot as plt
    # plt.imshow(np_im, interpolation='nearest')
    # plt.show()
    

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

    all_test_imgs = np.concatenate([np_test_imgs_carambola, np_test_imgs_pear])
    all_test_imgs = np.concatenate([all_test_imgs, np_test_imgs_plum])

    # plt.imshow(all_test_imgs[48], interpolation='nearest')
    # plt.show()

    # plt.imshow(all_test_imgs[293], interpolation='nearest')
    # plt.show()

    # plt.imshow(all_test_imgs[528], interpolation='nearest')
    # plt.show()

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    predictions = model.predict(all_test_imgs)

    correct_carambola=0
    correct_pear=0
    correct_plum=0

    for i in range(len(validation_generator.labels)):
        # print(np.argmax(predictions[i]))
        # print(validation_generator.labels[i])
        
        if(np.argmax(predictions[i]) == 0 and validation_generator.labels[i] == 0):
            correct_carambola += 1
        if(np.argmax(predictions[i]) == 1 and validation_generator.labels[i] == 1):
            correct_pear += 1
        if(np.argmax(predictions[i]) == 2 and validation_generator.labels[i] == 2):
            correct_plum += 1

    acc_carambola = correct_carambola / 200
    acc_pear = correct_pear / 200
    acc_plum = correct_plum / 200

    print("Acerto carambola: "+str(acc_carambola*100)+"%")
    print("Acerto pear: "+str(acc_pear*100)+"%")
    print("Acerto plum: "+str(acc_plum*100)+"%")

