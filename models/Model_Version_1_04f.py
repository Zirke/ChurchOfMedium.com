import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Input
from tensorflow.keras import Model
from tensorflow_core.python.layers.core import Dropout


class Model_Version_1_04f(tf.keras.Model):
    def __init__(self):
        super(Model_Version_1_04f, self).__init__()
        # first convolutional layer
        self.conv1 = Conv2D(32,  # filters
                            (3, 3),  # Kernel size
                            strides=(2, 2),  # Stride
                            padding='same',  # Same refers to same padding as previous layer.
                            data_format=None,
                            # It should be defined if the dimensions are structured in non standard approach
                            dilation_rate=(1, 1),  # how dilated the picture is
                            activation='relu',  # Activation function
                            use_bias=True,  # Enable bias
                            kernel_initializer='glorot_uniform',  # initialiser of filters
                            bias_initializer='zeros',  # initialisation of bias
                            kernel_regularizer=None,  #
                            bias_regularizer=None,  #
                            activity_regularizer=None,  #
                            kernel_constraint=None,  #
                            bias_constraint=None,  #
                            input_shape=(299, 299, 1))  # shape of the input

        self.maxpol1 = MaxPooling2D(pool_size=(3, 3),  # pool size
                                   strides=(2, 2),  # stride size
                                   padding='same',  # padding
                                   data_format=None)  #

        self.dropout2 = tf.keras.layers.Dropout(rate=0.3,
                                                noise_shape=None,
                                                seed=None)

        self.conv2 = Conv2D(64,  # filters
                            (3, 3),  # Kernel size
                            strides=(1, 1),  # Stride
                            padding='same',  # Same refers to same padding as previous layer.
                            data_format=None,
                            # It should be defined if the dimensions are structured in non standard approach
                            dilation_rate=(1, 1),  # how dilated the picture is
                            activation='relu',  # Activation function
                            use_bias=True,  # Enable bias
                            kernel_initializer='glorot_uniform',  # initialiser of filters
                            bias_initializer='zeros',  # initialisation of bias
                            kernel_regularizer=None,  #
                            bias_regularizer=None,  #
                            activity_regularizer=None,  #
                            kernel_constraint=None,  #
                            bias_constraint=None,  #
                            input_shape=(299, 299, 1))  # shape of the input

        self.maxpol2 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                   strides=(2, 2),  # stride size
                                   padding='same',  # padding
                                   data_format=None)  #


        self.flatten = Flatten()

        # Dense is a fully connected layer
        self.d1 = Dense(1024,  # Amount of neurons
                        activation='relu',  # Activation function
                        use_bias=True,  # bias is enabled
                        bias_initializer='zeros',  # initialisation of bias
                        bias_regularizer=None,  # regularize biases
                        activity_regularizer=None,  #
                        bias_constraint=None)  #

        self.d2 = Dense(512,  # Amount of neurons
                        activation='relu',  # Activation function
                        use_bias=True,  # bias is enabled
                        bias_initializer='zeros',  # initialisation of bias
                        bias_regularizer=None,  # regularize biases
                        activity_regularizer=None,  #
                        bias_constraint=None)  #

        self.d3 = Dense(5,  # Amount of neurons
                        activation='softmax',  # Activation function
                        use_bias=True,  # bias is enabled
                        bias_initializer='zeros',  # initialisation of bias
                        bias_regularizer=None,  # regularize biases
                        activity_regularizer=None,  #
                        bias_constraint=None)  #

    # Call method should include all layers from model.
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpol1(x)
        x = self.conv2(x)
        x = self.maxpol2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    def model(self):
        x = Input(shape=(299, 299, 1))
        return Model(inputs=[x], outputs=self.call(x))

