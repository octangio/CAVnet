import tensorflow as tf
from tensorflow.keras import backend, losses


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return tf.reshape(1 - numerator / denominator, (-1, 1, 1))


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    union = backend.sum(y_true_f + y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def CE_DL_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    ce_loss = losses.categorical_crossentropy(y_true, y_pred)
    dl_loss = dice_loss(y_true, y_pred)
    return ce_loss + dl_loss
