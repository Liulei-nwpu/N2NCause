import enum

import tensorflow as tf

from layers.similarity import manhattan_distance,euclidean_distance

_EPSILON = 1e-6


class LossType(enum.Enum):
    MSE = 0,
    MAE = 1,
    CROSS_ENTROPY = 2,
    CONTRASTIVE = 3,
    CONTRASTIVE_LECUN = 4,
    SIGMOID_CROSS = 5,




def get_loss_function(loss_type: str):
    if loss_type == LossType.MSE.name:
        return mse
    elif loss_type == LossType.MAE.name:
        return mae
    elif loss_type == LossType.CROSS_ENTROPY.name:
        return cross_entropy
    elif loss_type == LossType.CONTRASTIVE.name:
        return contrastive
    elif loss_type == LossType.CONTRASTIVE_LECUN.name:
        return contrastive_lecun
    elif loss_type == LossType.SIGMOID_CROSS.name:
        return sigmoid_cross
    else:
        raise AttributeError('{} loss type not supported.'.format(loss_type))


def contrastive(predictions, labels):
    contrastive_loss_minus = tf.to_float(labels) * _contrastive_plus(predictions)
    contrastive_loss_plus = (1.0 - tf.to_float(labels)) * _contrastive_minus(predictions)
    c_loss = tf.reduce_sum(contrastive_loss_plus + contrastive_loss_minus)
    return c_loss


def _contrastive_plus(model_energy):
    return 0.25 * tf.square(1.0 - tf.to_float(model_energy))


def _contrastive_minus(model_energy, margin=tf.constant(0.5)):
    mask = tf.to_float(tf.less(tf.to_float(model_energy), margin))
    return mask * tf.square(tf.to_float(model_energy))


def cross_entropy(predictions, labels):
    labels = tf.cast(labels, "float")
    predictions = tf.cast(predictions, "float")
    return tf.reduce_mean(
        -tf.reduce_sum(
            labels * tf.log(predictions + _EPSILON),
            axis=1
        )
    )

def mse(predictions, labels):
    return tf.losses.mean_squared_error(labels, predictions)

def mae(predictions, labels):
    # return tf.reduce_mean(tf.losses.absolute_difference(labels, predictions))
    maes = tf.losses.absolute_difference(labels, predictions)
    maes_loss = tf.reduce_sum(maes)
    return maes_loss

def log_loss(predictions, labels):
    logs = tf.loss.log_loss(labels, predictions)
    return tf.reduce_mean(logs)

def sigmoid_cross(predictions, labels):
    sigmoids = tf.nn.sigmoid_cross_entropy_with_logits(labels, predictions)
    sigmoids_loss = tf.reduce_mean(sigmoids)
    return sigmoids_loss

def contrastive_lecun(x1, x2, labels, predictions, margin=tf.constant(0.2), distance=euclidean_distance):
    temp_sim = tf.rint(predictions)
    correct_predictions = tf.equal(temp_sim, tf.to_float(labels))
    labels = tf.to_float(labels)
    correct_predictions = tf.to_float(correct_predictions)
    c_loss1 = correct_predictions * 0.5 * tf.square(distance(x1, x2)) + \
             (1 - correct_predictions) * 0.5 * tf.square(tf.maximum(0.0, margin - distance(x1, x2))) 
    c_loss2 = tf.losses.mean_squared_error(labels, predictions)
    # c_loss = labels * 0.5 * distance(x1, x2) * distance(x1, x2) + \
    #          (1 - labels) * 0.5 * tf.maximum(0.0, margin - distance(x1, x2)) * tf.maximum(0.0, margin - distance(x1, x2))
    return 0.4*tf.reduce_mean(tf.reduce_sum(c_loss1)) + 0.6 * c_loss2
