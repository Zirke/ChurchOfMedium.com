import tensorflow as tf
class Confusion_matrix(tf.keras.metrics.Metric):
    def __init__(self, name= 'confusion', **kwargs):
        super(Confusion_matrix, self).__init__(name = name, **kwargs)
        self.cf_matrix = self.add_weight(name = 'cf_matrix', initializer = 'zeros')

    def update_state(self, y_true, y_pred, sample_weight = None):
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)
        # print(y_pred)

        c_f_matrix = tf.math.confusion_matrix(y_true,y_pred)
        print(c_f_matrix)
        self.cf_matrix = tf.math.add(c_f_matrix, self.cf_matrix)

    def result(self):
        return self.cf_matrix