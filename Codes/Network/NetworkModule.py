import numpy as np
import keras
from keras import models
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D

from Network.Config import Configure
from Network.Config import Methods


def Image_Module():
    input = Input(shape=(Configure.image_size[0], Configure.image_size[1], 3))
    x = Conv2D(16, (3, 3), activation='relu')(input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2),(2,2))(x)
    x = Conv2D(65, (3, 3), activation='relu')(x)
    x = Conv2D(65, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2),(2,2))(x)
    x = Flatten()(x)
    z = Dense(64, activation="relu")(x)
    return Model(inputs=input, outputs=z)   # input matrix(w,h,3) -> output vector(64)


def PoseInfo_Module():
    input = Input(shape=(1*4 + 2*4,))
    z = Dense(256, activation="relu")(input)
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(64, activation="relu")(z)

    return Model(inputs=input, outputs=z)  # input vector(12,) -> output vector(64)

#############################
#############################
#########  GAZENET  #########
#############################
#############################
class GazeNet:

    @staticmethod
    def BuildNetwork():
        # print("Build GazeNet!!!!!")
        #
        # input1 = Input(shape=(Configure.image_size[1], Configure.image_size[0], 3))
        # input2 = Input(shape=(1 * 4 + 2 * 4,))
        #
        # Module1 = Image_Module()  #
        # Module2 = PoseInfo_Module()
        #
        #
        # result1 = Module1(input1)
        # result2 = Module2(input2)
        #
        # z = keras.layers.Concatenate()([result1, result2])  #128
        #
        # z = Dense(64, activation="relu")(z)
        # z = Dropout(0.5)(z)
        # z = Dense(256, activation="relu")(z)
        # z = Dropout(0.5)(z)
        # z = Dense(2, activation="linear")(z)  # 激活函数：LINEAR。 原因是因为我们的task是一个regression问题
        #
        # return Model(inputs=(input1, input2), outputs = z)

        print("Build GazeNet!!!!!")

        input1 = Input(shape=(Configure.image_size[1], Configure.image_size[0], 3))
        input2 = Input(shape=(1 * 4 + 2 * 4,))

        Module1 = Image_Module()  #
        Module2 = PoseInfo_Module()


        result1 = Module1(input1)
        result2 = Module2(input2)

        z = keras.layers.Concatenate()([result1, result2])  #128

        z = Dense(256, activation="relu")(z)
        #z = Dropout(0.5)(z)
        z = Dense(64, activation="relu")(z)
        #z = Dropout(0.5)(z)
        z = Dense(2, activation="linear")(z)  # 激活函数：LINEAR。 原因是因为我们的task是一个regression问题

        return Model(inputs=(input1, input2), outputs = z)