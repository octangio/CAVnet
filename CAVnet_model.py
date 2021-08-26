import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU
from tensorflow.keras.layers import Conv2DTranspose


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              dilation_rate=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        dilation_rate=dilation_rate,
        name=conv_name)(x)
    x = BatchNormalization(name=bn_name)(x)
    x = LeakyReLU()(x)
    # x=ReLU()(x)
    return x


def feature_block(x):
    b1 = conv2d_bn(x, 64, 3, 3)
    b1 = conv2d_bn(b1, 64, 3, 3)
    b1 = conv2d_bn(b1, 64, 3, 3)
    b2 = conv2d_bn(b1, 64, 3, 3, dilation_rate=(2, 2))
    b2 = conv2d_bn(b2, 64, 3, 3, dilation_rate=(3, 3))
    b2 = conv2d_bn(b2, 64, 3, 3, dilation_rate=(5, 5))
    block = layers.concatenate([b1, b2])
    return block


def cavnet(input_shape):
    with tf.device('gpu:0'):
        with tf.name_scope('input'):
            input_img = layers.Input(shape=input_shape, name='input_angio')

        with tf.name_scope('conv1'):
            x1 = conv2d_bn(input_img, 128, 3, 3, name='conv1')

        with tf.name_scope('block1'):
            block1 = feature_block(x1)

        with tf.name_scope('block2'):
            x2 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block1)
            x2 = BatchNormalization()(x2)
            x2 = layers.LeakyReLU()(x2)
            block2 = feature_block(x2)

        with tf.name_scope('block3'):
            x3 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block2)
            x3 = BatchNormalization()(x3)
            x3 = layers.LeakyReLU()(x3)
            block3 = feature_block(x3)

        with tf.name_scope('block4'):
            x4 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block3)
            x4 = BatchNormalization()(x4)
            x4 = layers.LeakyReLU()(x4)
            block4 = feature_block(x4)

        with tf.name_scope('mid'):
            xm = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(block4)
            xm = layers.LeakyReLU()(xm)
            mid = feature_block(xm)
            mid = Conv2DTranspose(64, 2, strides=2)(mid)
    with tf.device('gpu:1'):
        with tf.name_scope('ublock4'):
            xu4 = layers.concatenate([mid, block4])
            ublock4 = feature_block(xu4)
            ublock4 = Conv2DTranspose(64, 2, strides=2)(ublock4)

        with tf.name_scope('ublock3'):
            xu3 = layers.concatenate([ublock4, block3])
            ublock3 = feature_block(xu3)
            ublock3 = Conv2DTranspose(64, 2, strides=2)(ublock3)

        with tf.name_scope('ublock2'):
            xu2 = layers.concatenate([ublock3, block2])
            ublock2 = feature_block(xu2)
            ublock2 = Conv2DTranspose(64, 2, strides=2)(ublock2)

        with tf.name_scope('ublock1'):
            xu1 = layers.concatenate([ublock2, block1])
            ublock1 = feature_block(xu1)

        with tf.name_scope('output'):
            xo = conv2d_bn(ublock1, 64, 3, 3)
            xo = Conv2D(4, [3, 3], padding='same')(xo)
            xo = BatchNormalization()(xo)
            output = layers.Activation('softmax')(xo)

        model = Model(inputs=input_img, outputs=output, name='CAVnet')
        model.summary()
        return model
