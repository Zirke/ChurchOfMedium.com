import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Input, Dropout


class Model_Version_1_07c(tf.keras.Model):
    def __init__(self):
        super(Model_Version_1_07c, self).__init__()

        # first convolutional layer
        self.conv_layer_1 = Conv2D(32,  # filters
                                   (3, 3),  # Kernel size
                                   strides=(1, 1),  # Stride
                                   padding='same',  # Same refers to same padding as previous layer.
                                   data_format=None,
                                   # It should be defined if the dimensions are structured in non standard approach
                                   dilation_rate=(1, 1),  # how dilated the picture is
                                   activation='relu',  # Activation function
                                   use_bias=True,  # Enable bias
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                   # initialiser of filters
                                   bias_initializer='zeros',  # initialisation of bias
                                   kernel_regularizer=None,
                                   bias_regularizer=None,  #
                                   activity_regularizer=None,  #
                                   kernel_constraint=None,  #
                                   bias_constraint=None,  #
                                   )
        self.ave_pool_1 = AveragePooling2D(pool_size=(2,2),
                                           strides=(2,2),
                                           padding='valid',
                                           data_format=None)

        self.max_pool_1 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                       strides=(2, 2),  # stride size
                                       padding='valid',  # padding
                                       data_format=None)

        self.conv_layer_2 = Conv2D(64,  # filters
                                   (3, 3),  # Kernel size
                                   strides=(1, 1),  # Stride
                                   padding='same',  # Same refers to same padding as previous layer.
                                   data_format=None,
                                   # It should be defined if the dimensions are structured in non standard approach
                                   dilation_rate=(1, 1),  # how dilated the picture is
                                   activation='relu',  # Activation function
                                   use_bias=True,  # Enable bias
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                   # initialiser of filters
                                   bias_initializer='zeros',  # initialisation of bias
                                   kernel_regularizer=None,
                                   bias_regularizer=None,  #
                                   activity_regularizer=None,  #
                                   kernel_constraint=None,  #
                                   bias_constraint=None,  #
                                   )

        self.max_pool_2 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                       strides=(2, 2),  # stride size
                                       padding='valid',  # padding
                                       data_format=None)

        self.ave_pool_2 = AveragePooling2D(pool_size=(2, 2),
                                           strides=(2, 2),
                                           padding='valid',
                                           data_format=None)

        self.conv_layer_3 = Conv2D(128,  # filters
                                   (3, 3),  # Kernel size
                                   strides=(1, 1),  # Stride
                                   padding='same',  # Same refers to same padding as previous layer.
                                   data_format=None,
                                   # It should be defined if the dimensions are structured in non standard approach
                                   dilation_rate=(1, 1),  # how dilated the picture is
                                   activation='relu',  # Activation function
                                   use_bias=True,  # Enable bias
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                   # initialiser of filters
                                   bias_initializer='zeros',  # initialisation of bias
                                   kernel_regularizer=None,
                                   bias_regularizer=None,  #
                                   activity_regularizer=None,  #
                                   kernel_constraint=None,  #
                                   bias_constraint=None,  #
                                   )

        self.ave_pool_3 = AveragePooling2D(pool_size=(2, 2),
                                           strides=(2, 2),
                                           padding='valid',
                                           data_format=None)
        self.max_pool_3 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                       strides=(2, 2),  # stride size
                                       padding='valid',  # padding
                                       data_format=None)

        self.conv_layer_4 = Conv2D(128,  # filters
                                   (3, 3),  # Kernel size
                                   strides=(1, 1),  # Stride
                                   padding='same',  # Same refers to same padding as previous layer.
                                   data_format=None,
                                   # It should be defined if the dimensions are structured in non standard approach
                                   dilation_rate=(1, 1),  # how dilated the picture is
                                   activation='relu',  # Activation function
                                   use_bias=True,  # Enable bias
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                   # initialiser of filters
                                   bias_initializer='zeros',  # initialisation of bias
                                   kernel_regularizer=None,
                                   bias_regularizer=None,  #
                                   activity_regularizer=None,  #
                                   kernel_constraint=None,  #
                                   bias_constraint=None,  #
                                   )
        self.ave_pool_4 = AveragePooling2D(pool_size=(2, 2),
                                           strides=(2, 2),
                                           padding='valid',
                                           data_format=None)

        self.max_pool_4 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                       strides=(2, 2),  # stride size
                                       padding='valid',  # padding
                                       data_format=None)
        self.flatten = Flatten()

        self.fc_layer_1 = Dense(512,  # Amount of neurons
                                activation='relu',  # Activation function
                                use_bias=True,  # bias is enabled
                                #kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                bias_initializer='zeros',  # initialisation of bias
                                bias_regularizer=None,  # regularize biases
                                activity_regularizer=None,  #
                                bias_constraint=None)  #

        # Dense is a fully connected layer
        self.fc_layer_2 = Dense(512,  # Amount of neurons
                                activation='relu',  # Activation function
                                use_bias=True,  # bias is enabled
                                #kernel_regularizer=tf.keras.regularizers.l1(0.01),
                                bias_initializer='zeros',  # initialisation of bias
                                bias_regularizer=None,  # regularize biases
                                activity_regularizer=None,  #
                                bias_constraint=None)  #

        self.output_layer = Dense(5,  # Amount of neurons
                                  activation='softmax',  # Activation function
                                  use_bias=True,  # bias is enabled
                                  bias_initializer='zeros',  # initialisation of bias
                                  bias_regularizer=None,  # regularize biases
                                  activity_regularizer=None,  #
                                  bias_constraint=None)  #

    # Call method should include all layers from model.
    def call(self, x):
        x = self.conv_layer_1(x)
        x = self.ave_pool_1(x)
        x = self.conv_layer_2(x)
        x = self.ave_pool_2(x)
        x = self.conv_layer_3(x)
        x = self.ave_pool_3(x)
        x = self.conv_layer_4(x)
        x = self.ave_pool_4(x)
        x = self.flatten(x)
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        return self.output_layer(x)

    def model(self):
        x = Input(shape=(299, 299, 1))
        return Model(inputs=[x], outputs=self.call(x))
