# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# talvez tentar melhorar o código usando isso: https://keras.io/api/preprocessing/image/

# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/

import os
import glob
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model

from keras_visualizer import visualizer

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# dimensions of our images.
img_width, img_height = 320, 258

path = os.getcwd()
train_data_dir = path+'/deep-supervisioned-learning/data/train'
validation_data_dir = path+'/deep-supervisioned-learning/data/validation'
nb_train_samples = 1800
nb_validation_samples = 200


EPOCHS = 4
batch_size = 32
training = False
generate_graphs = False
mymodel = 3
file_name = "cnn_fullsizeimg_model"+str(mymodel)+"_"+str(EPOCHS)+"epochs_"+str(batch_size)+"batch_datagen.h5"
# Modelo bom
# file_name = "cnn_fullsizeimg_model3_4epochs_32batch_datagen(old).h5"

if(not generate_graphs):

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
        # input_shape = (img_height, img_width, 3)

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

        # visualizer(model, format='png', view=True)

        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    if(mymodel == 4 ):
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

        history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)

        np.save('deep-supervisioned-learning/'+file_name[:-3]+'.npy', history.history)

        model.save_weights('deep-supervisioned-learning/'+file_name)


    if(not training):
        model.load_weights('deep-supervisioned-learning/'+file_name)
        #model.load_weights('deep-supervisioned-learning/cnn_fullsizeimg_model2_fitgenerator_2epochs_32batch.h5')

        # plt.imshow(np_im, interpolation='nearest')
        # plt.show() 

        print(model.summary())
        
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
        confusion_mtx = confusion_matrix(labels_argmax, pred_argmax)

        plt.figure(figsize=(10,8))
        sns.heatmap(confusion_mtx, annot=True, fmt="d")
        plt.show()
        
        """
        plt.imshow(imgs[0], interpolation='nearest')
        plt.show()
        # print("label img: "+str(labels[0]))
    
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(imgs[0].reshape(1,img_height,img_width,3))
        
        def display_activation(activations, col_size, row_size, act_index): 
            activation = activations[act_index]
            activation_index=0
            fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
            for row in range(0,row_size):
                for col in range(0,col_size):
                    ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
                    activation_index += 1

        display_activation(activations, 8, 4, 0)
        display_activation(activations, 8, 4, 2)

        display_activation(activations, 8, 4, 3)
        display_activation(activations, 8, 4, 5)
        plt.show()
        """

if(generate_graphs):
        history = np.load('deep-supervisioned-learning/'+file_name[:-3]+'.npy', allow_pickle='TRUE').item()
        
        sns.set_theme(style="whitegrid")
        sns.set_palette(sns.color_palette("bright"))
        

        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = history['accuracy']
        val_acc = history['val_accuracy']
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


        # sns.lineplot(x=epochs, y=loss, label='Training loss')
        # sns.lineplot(x=epochs, y=val_loss, label='Validation loss')
        # plt.show()