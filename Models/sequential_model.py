import tensorflow as tf
from callback import *
from main import FILE_SIZE,TEST_SIZE,BATCH_SIZE

def create_model(batched_training_data,batched_val_data,callback, cp_callback, tb_callback):
    # initializing the callback
    callback = myCallback()
    tb_callback = tensorboard_callback("logs", 1)
    cp_callback = checkpoint_callback()

    # define the model for training
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(299, 299, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        batched_training_data,
        steps_per_epoch=FILE_SIZE // BATCH_SIZE,  # FILE_SIZE
        validation_data=batched_val_data,
        validation_steps=TEST_SIZE // BATCH_SIZE,  # TEST_SIZE
        epochs=5,
        verbose=1,  # verbose is the progress bar when training
        callbacks=[callback, cp_callback, tb_callback]
    )
    return model, history


# use in main to get output
#model, history = create_model(batched_training_data,batched_val_data,callback,cp_callback,tb_callback)
