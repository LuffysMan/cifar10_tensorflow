import tensorflow as tf


def _conv2d(input, input_filters, output_filters, kernel, strides=1, padding="same"):
    with tf.variable_scope('conv2d'):
        shape = [kernel, kernel, input_filters, output_filters]
        weights = tf.get_variable('weights', shape=shape, dtpyt=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(input, weights, strides=[1,strides, strides, 1], padding=padding, name='conv')
        return conv

class ResNet():
    def __init__(self, *args, **kwargs):
        self.training = ''

    def bn_relu(self, input):
        """Helper to build a BN -> relu block
        """
        with tf.variable_scope('bn_relu'):
            norm = tf.layers.batch_normalization(input, training=self.training, name='norm')
            relu = tf.nn.relu(norm, name='relu')
            return relu

    def conv_bn_relu(self, input, input_filters, output_filters, kernel, strides, padding='same'):
        """Helper to build a conv -> BN -> relu block
        """
        with tf.variable_scope('conv_bn_relu'):
            conv = _conv2d(input, input_filters, output_filters, kernel, strides, padding=padding)
            bn_relu = self.bn_relu(conv)
            return bn_relu
        
    def bn_relu_conv(self, input, input_filters, output_filters, kernel, strides, padding="same"):
        """Helper to build a BN -> relu -> conv block.
        This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        with tf.variable_scope('bn_relu_conv'):
            bn_relu = self.bn_relu(input)
            conv = _conv2d(bn_relu, input_filters, output_filters, kernel, strides=strides, padding=padding)
            return conv

    def basic_block(self, input, num_filters, strides=1, with_shortcut=False):
        """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        """
        with tf.variable_scope('basic_block'):
            conv = _conv2d(input, num_filters, kernel, strides, padding='same', name='conv')

            if with_shortcut:
                shortcut = self.conv2d(input, num_filters[0], num_filters[3], kernel=1, strides=strides)
                bn_shortcut = tf.layers.batch_normalization(shortcut, axis=3, training=self.training)
                residual = tf.nn.relu(bn_shortcut+bn3)
            else:
                residual = tf.nn.relu(input+bn3)
