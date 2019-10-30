import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Input, Dropout


class Model_Version_2_02c(tf.keras.Model):
    def __init__(self):
        super(Model_Version_2_02c, self).__init__()

        # first convolutional layer
        self.conv1 = Conv2D(16,  # filters
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
                            kernel_regularizer=tf.keras.regularizers.l1(0.001),
                            bias_regularizer=None,  #
                            activity_regularizer=None,  #
                            kernel_constraint=None,  #
                            bias_constraint=None,  #
                            )

        self.maxpool1 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                     strides=(2, 2),  # stride size
                                     padding='same',  # padding
                                     data_format=None)  #

        # Second convolutional layer
        self.conv2 = Conv2D(32,  # filters
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
                            kernel_regularizer=tf.keras.regularizers.l1(0.001),
                            bias_regularizer=None,  #
                            activity_regularizer=None,  #
                            kernel_constraint=None,  #
                            bias_constraint=None,  #
                            )

        # 3rd convolutional layer
        self.conv3 = Conv2D(64,  # filters
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
                            )

        self.conv4 = Conv2D(128,  # filters
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
                            )

        self.con5 = Conv2D(256,  # filters
                           (3, 3),  # Kernel size
                           strides=(1, 1),  # Stride
                           padding='valid',  # Same refers to same padding as previous layer.
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
                           )

        self.conv6 = Conv2D(512,  # filters
                            (3, 3),  # Kernel size
                            strides=(1, 1),  # Stride
                            padding='valid',  # Same refers to same padding as previous layer.
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
                            )

        self.drop = Dropout(rate=0.35,
                            seed=None,
                            noise_shape=None)

        self.flatten = Flatten()

        self.d1 = Dense(1024,  # Amount of neurons
                        activation='relu',  # Activation function
                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                        use_bias=True,  # bias is enabled
                        bias_initializer='zeros',  # initialisation of bias
                        bias_regularizer=None,  # regularize biases
                        activity_regularizer=None,  #
                        bias_constraint=None)  #

        # Dense is a fully connected layer
        self.d2 = Dense(1024,  # Amount of neurons
                        activation='relu',  # Activation function
                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                        use_bias=True,  # bias is enabled
                        bias_initializer='zeros',  # initialisation of bias
                        bias_regularizer=None,  # regularize biases
                        activity_regularizer=None,  #
                        bias_constraint=None)  #

        self.d3 = Dense(1,  # Amount of neurons
                        activation='sigmoid',  # Activation function
                        use_bias=True,  # bias is enabled
                        bias_initializer='zeros',  # initialisation of bias
                        bias_regularizer=None,  # regularize biases
                        activity_regularizer=None,  #
                        bias_constraint=None)  #

    # Call method should include all layers from model.
    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool1(x)
        x = self.con5(x)
        x = self.maxpool1(x)
        x = self.conv6(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop(x)
        x = self.d2(x)
        x = self.drop(x)
        return self.d3(x)

    def model(self):
        x = Input(shape=(299, 299, 1))
        return Model(inputs=[x], outputs=self.call(x))