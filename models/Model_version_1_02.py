import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Input


class Model_Version_1_02(tf.keras.Model):
    def __init__(self):
        super(Model_Version_1_02, self).__init__()

        # Adding 2 layers of padding on each side, temporarily making the image 303x303
        self.padding = ZeroPadding2D(padding=(2, 2),  # Amount of Zeros on each side of the input
                                     data_format='channels_last',  # format of batch, height, width, channels
                                     )
        # first convolutional layer
        self.conv1 = Conv2D(64,  # filters
                            (3, 3),  # Kernel sizew
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

        self.maxpool1 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                     strides=(2, 2),  # stride size
                                     padding='same',  # padding
                                     data_format=None)  #

        # Second convolutional layer
        self.conv2 = Conv2D(128,  # filters
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

        # 3rd convolutional layer
        self.conv3 = Conv2D(128,  # filters
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

        self.maxpool2 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                     strides=(2, 2),  # stride size
                                     padding='same',  # padding
                                     data_format=None)  #

        # 4th convolutional layer
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

        # 5th convolutional layer
        self.conv5 = Conv2D(256,  # filters
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
        # 6th convolutional layer
        self.conv6 = Conv2D(256,  # filters
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

        self.maxpool3 = MaxPooling2D(pool_size=(2, 2),  # pool size
                                     strides=(2, 2),  # stride size
                                     padding='same',  # padding
                                     data_format=None)  #

        self.flatten = Flatten()

        # Dense is a fully connected layer
        self.d1 = Dense(256,  # Amount of neurons
                        activation='relu',  # Activation function
                        use_bias=True,  # bias is enabled
                        bias_initializer='zeros',  # initialisation of bias
                        bias_regularizer=None,  # regularize biases
                        activity_regularizer=None,  #
                        bias_constraint=None)  #

        # Dense is a fully connected layer
        self.d2 = Dense(256,  # Amount of neurons
                        activation='relu',  # Activation function
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
        x = self.padding(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    def model(self):
        x = Input(shape=(299, 299, 1))
        return Model(inputs=[x], outputs=self.call(x))
