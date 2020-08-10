
# Importing the libraries
from keras.layers import Input
#from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Convolution2D, Conv2D
#from keras.layers.convolutional import MaxPooling2D
from keras.layers import Reshape,Bidirectional, BatchNormalization, AveragePooling2D, MaxPooling2D, concatenate,TimeDistributed, LSTM, MaxPooling1D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
import pandas as pd
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.layers.wrappers import TimeDistributed
from keras import backend as K
import tensorflow as tf
from keras.utils.data_utils import get_file
import numpy as np 
from keras.layers import InputLayer



traindf=pd.read_csv("Action_Test.csv",dtype=str)
valid_datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.05)
test_datagen =  ImageDataGenerator(rescale=1./255.,validation_split=0.15)
train_datagen = ImageDataGenerator(
            rescale = 1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

train_generator=train_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="Action_images",
    x_col='Image_cropped',
    y_col='Emotion',
    subset="training",
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(299, 299))

valid_generator=valid_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="Action_images",
    x_col="Image_cropped",
    y_col="Emotion",
    subset="validation",
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(299, 299))

test_generator=test_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="Action_images",
    x_col="Image_cropped",
    y_col="Emotion",
    subset="validation",
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(299, 299))





"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""

TH_BACKEND_TH_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_th_dim_ordering_th_kernels.h5"
TH_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_th_kernels.h5"
TF_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_tf_kernels.h5"
TF_BACKEND_TH_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_th_dim_ordering_tf_kernels.h5"


def merge(inputs, mode, concat_axis=-1):
    return concatenate(inputs)


def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def inception_stem(input):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv_block(x, 32, 3, 3, border_mode='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')
##########################################################################################
    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, border_mode='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, border_mode='valid')

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)
    return x


def inception_A(input):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
    return m


def inception_B(input):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=channel_axis)
    return m


def inception_C(input):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    c1 = conv_block(input, 256, 1, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)
    c2 = merge([c2_1, c2_2], mode='concat', concat_axis=channel_axis)

    c3 = conv_block(input, 384, 1, 1)
    c3 = conv_block(c3, 448, 3, 1)
    c3 = conv_block(c3, 512, 1, 3)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)
    c3 = merge([c3_1, c3_2], mode='concat', concat_axis=channel_axis)

    c4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    m = merge([c1, c2, c3, c4], mode='concat', concat_axis=channel_axis)
    return m


def reduction_A(input):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    return m


def reduction_B(input):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 256, 1, 1)
    r2 = conv_block(r2, 256, 1, 7)
    r2 = conv_block(r2, 320, 7, 1)
    r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    return m


def create_inception_v4(nb_classes=1001, load_weights=True):
    '''
    Creates a inception v4 network
    :param nb_classes: number of classes.txt
    :return: Keras Model with 1 input and 1 output
    '''

    if K.image_data_format() == 'th':
        init = Input((3, 299, 299))
    else:
        init = Input((299, 299, 3))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, out, name='Inception-v4')

    if load_weights:
        if K.backend() == "theano":
            if K.image_data_format() == "th":
                weights = get_file('inception_v4_weights_th_dim_ordering_th_kernels.h5', TH_BACKEND_TH_DIM_ORDERING,
                                   cache_subdir='models')
            else:
                weights = get_file('inception_v4_weights_tf_dim_ordering_th_kernels.h5', TH_BACKEND_TF_DIM_ORDERING,
                                   cache_subdir='models')
        else:
            if K.image_data_format() == "th":
                weights = get_file('inception_v4_weights_th_dim_ordering_tf_kernels.h5', TF_BACKEND_TH_DIM_ORDERING,
                                   cache_subdir='models')
            else:
                weights = get_file('inception_v4_weights_tf_dim_ordering_tf_kernels.h5', TH_BACKEND_TF_DIM_ORDERING,
                                   cache_subdir='models')

        model.load_weights(weights)
        print("Model weights loaded.")

    return model



# In[23]:




if __name__ == "__main__":

    inception_v4 = create_inception_v4(load_weights=True)


print("Layers:",len(inception_v4.layers))
print(inception_v4.input)
inception_v4.summary()



# # Loading the weights

n1= Reshape((1,1001))
n2=Bidirectional(LSTM(512))
n20=Reshape((1,1024))
n21=LSTM(128)
n211=Dropout(0.5)
n22=Activation('relu')
n222=Reshape((1,128))
n23=Flatten()
n3=Dense(5, activation = 'softmax', name='my_dense_1')



inp1 = inception_v4.get_layer('input_1').input
out1= inception_v4.get_layer('dense_1').output
model1=Model(inp1,out1)

inp2=model1.input
out2=n1(model1.output)
model2=Model(inp2,out2)
inp3=model2.input

out3=n2(model2.output)
model3=Model(inp3,out3)
inp4=model3.input
out30=n20(model3.output)
model30=Model(inp4,out30)
inp40=model30.input

out21=n21(model30.output)
model21=Model(inp40,out21)
inp31=model21.input

out211=n211(model21.output)
model211=Model(inp31,out211)
inp311=model211.input

out22=n22(model211.output)
model22=Model(inp311,out22)
inp32=model22.input
model22.summary()

out222=n222(model22.output)
model222=Model(inp32,out222)
inp322=model222.input
model222.summary()

out23=n23(model222.output)
model23=Model(inp322,out23)
inp33=model23.input


out4=n3(model23.output)
model5=Model(inp33,out4)
model5.summary()

model5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model5.load_weights("/Action/weights/weights_2-improvement-11-0.79.hdf5")
print("Loaded model from disk")

scores=model5.evaluate_generator(test_generator, len(test_generator))
print("%s: %.2f%%" % (model5.metrics_names[1], scores[1]*100))
print(test_generator.class_indices)
prediction= model5.predict_generator(test_generator,len(test_generator))
print("****Predictions:****")
print(prediction)
print(type(prediction))
print (test_generator.filenames)
#accuracy
accuracy_score(test_generator.classes, predicted_class_indices)


