import os
import glob
import numpy as np
from PIL import Image

# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
# from keras import backend as K

path = os.getcwd()
train_data_dir = path+'/deep-supervisioned-learning/data/train'
validation_data_dir = path+'/deep-supervisioned-learning/data/validation'

train_labels = np.concatenate([np.full(1800, 0), np.full(1800, 1), np.full(1800, 2)])
test_labels = np.concatenate([np.full(200, 0), np.full(200, 1), np.full(200, 2)])


def create_nparray_with_images(data_dir):
    img_list = []
    img_dirs = glob.glob(data_dir)
    for img_path in img_dirs:
        image = Image.open(img_path)
        np_im = np.array(image)
        img_list.append(np_im)
    
    return np.array(img_list)

train_imgs_carambola = create_nparray_with_images(train_data_dir+'/carambola/*.png')
train_imgs_pear = create_nparray_with_images(train_data_dir+'/pear/*.png')
train_imgs_plum = create_nparray_with_images(train_data_dir+'/plum/*.png')

test_imgs_carambola = create_nparray_with_images(validation_data_dir+'/carambola/*.png')
test_imgs_pear = create_nparray_with_images(validation_data_dir+'/pear/*.png')
test_imgs_plum = create_nparray_with_images(validation_data_dir+'/plum/*.png')

train_images = np.concatenate([train_imgs_carambola, train_imgs_pear, train_imgs_plum])
test_images = np.concatenate([test_imgs_carambola, test_imgs_pear, test_imgs_plum])


"""
Código para mostrar imagens e as labels correspondentes
"""
# from matplotlib import pyplot as plt
# plt.imshow(train_images[285], interpolation='nearest')
# print(train_labels[285])
# plt.show()
# plt.imshow(train_images[2512], interpolation='nearest')
# print(train_labels[2512])
# plt.show()
# plt.imshow(train_images[3895], interpolation='nearest')
# print(train_labels[3895])
# plt.show()

# plt.imshow(test_images[59], interpolation='nearest')
# print(test_labels[59])
# plt.show()
# plt.imshow(test_images[269], interpolation='nearest')
# print(test_labels[269])
# plt.show()
# plt.imshow(test_images[455], interpolation='nearest')
# print(test_labels[455])
# plt.show()

img_width, img_height = 320, 258
input_shape = (img_width, img_height, 3)

# Build the model.
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

# Compile the model.
model.compile(
'adam',
loss='categorical_crossentropy',
metrics=['accuracy'],
)

# Train the model.
model.fit(
train_images,
to_categorical(train_labels),
epochs=5,
validation_data=(test_images, to_categorical(test_labels)),
)

model.save_weights('cnn-fruit-handcrafted-5layers-5epochs.h5')
