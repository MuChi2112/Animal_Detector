# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1 # channel的維度
 
    # if we are using "channels first", update the input shape
    # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
                
        model.add(Conv2D(64, (3, 3), input_shape=(height, width, 3), activation='relu', padding='same', name='Conv1-1'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1-2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling1'))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv2-1'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv2-2'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling2'))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv3-1'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv3-2'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv3-3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling3'))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv4-1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv4-2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv4-3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling4'))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv5-1'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv5-2'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv5-3'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling5'))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu', name='FC1'))
        model.add(Dense(4096, activation='relu', name='FC2'))
        model.add(Dense(classes, activation='softmax', name='FC3'))

        # Compile the model (you can choose an optimizer and loss function accordingly)
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # return the constructed network architecture
        return model

# model = SmallerVGGNet.build(96, 96, 3, 3)
# model.summary()