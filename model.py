from tfinterface.supervised import SoftmaxClassifier
import tensorflow as tf

class Model(SoftmaxClassifier):

    def __init__(self, n_classes, *args, **kwargs):
        self.n_classes = n_classes

        self._initial_learning_rate = kwargs.pop("initial_learning_rate", 0.001)
        self._decay_steps = kwargs.pop("decay_steps", 200)
        self._decay_rate = kwargs.pop("decay_rate", 0.96)
        self._staircase = kwargs.pop("staircase", True)

        super(Model, self).__init__(*args, **kwargs)

    def get_labels(self, inputs):
        # one hot labels
        return tf.one_hot(inputs.labels, self.n_classes)

    def get_learning_rate(self, inputs):
        return tf.train.exponential_decay(
            self._initial_learning_rate,
            self.inputs.global_step,
            self._decay_steps,
            self._decay_rate,
            staircase = True
        )

    def get_logits(self, inputs):

        # cast
        net = tf.cast(self.inputs.features, tf.float32, "cast")
        # input is 32x32x3

        # conv layers
        net = tf.layers.conv2d(net, 16, [5, 5], activation=tf.nn.elu, name="elu_1", padding="same")
        # Parameters: 5*5*3*16 =
        net = tf.layers.conv2d(net, 32, [5, 5], activation=tf.nn.elu, name="elu_2", padding="same")
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, name="max_pool_1", padding="same")
        # Parameters: 5*5*16*32 =

        #net = tf.layers.conv2d(net, 32, [3, 3], activation=tf.nn.elu, name="elu_3", padding="same")
        net = tf.layers.conv2d(net, 64, [3, 3], activation=tf.nn.elu, name="elu_3_a", padding="same")
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, name="max_pool_2", padding="same")
        # Parameters: 3*3*32*64 =

        net = tf.layers.conv2d(net, 64, [3, 3], activation=tf.nn.elu, name="elu_4", padding="same")
        #net = tf.layers.conv2d(net, 64, [3, 3], activation=tf.nn.elu, strides=2, name="elu_4_a", padding="same")
        # Parameters: 3*3*64*64 =

        # flatten
        net = tf.contrib.layers.flatten(net)

        # dense layers
        net = tf.layers.dense(net, 256, activation=tf.nn.elu)
        #net = tf.nn.dropout(net, self.inputs.keep_prob)
        net = tf.layers.dropout(net, rate = 0.15, training = inputs.training)
        # Parameters: 8*8*64*256 =

        net = tf.layers.dense(net, 128, activation=tf.nn.elu)
        #net = tf.layers.dropout(net, rate = 0.2, training = inputs.training)
        #net = tf.nn.dropout(net, self.inputs.keep_prob)
        # Parameters: 256*128 =

        # output layer
        return tf.layers.dense(net, self.n_classes)
        # Parameters: 128*43 =

        # TOTAL PARAMS: 1,156,144
        # = 5*5*3*16 + 5*5*16*32 + 3*3*32*64 + 3*3*64*64 +8*8*64*256 + 256*128 + 128*43 =  1,156,144

    def get_summaries(self, inputs):
        return [
            tf.summary.scalar("learning_rate", self.learning_rate)
        ]
