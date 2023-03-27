from requests import session
import tensorflow as tf

from layers.basics import optimize
from layers.similarity import manhattan_similarity
from layers import losses
import numpy as np


class BaseSiameseNet:
    
    def __init__(
            self,
            max_sequence_len,
            vocabulary_size,
            main_cfg,
            model_cfg,
    ):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_len])
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.sentences_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        
        self.debug = None
        self.debug_vars = dict()
        
        self.loss_function = losses.get_loss_function(main_cfg['PARAMS'].get('loss_function'))
        self.embedding_size = main_cfg['PARAMS'].getint('embedding_size') # get embedding size from config
        self.learning_rate = main_cfg['TRAINING'].getfloat('learning_rate') # get learning rate from config
        
        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings',
                                              [vocabulary_size, self.embedding_size])
            self.embedded_x1 = tf.gather(word_embeddings, self.x1)
            self.embedded_x2 = tf.gather(word_embeddings, self.x2)
        
        with tf.variable_scope('siamese'):
            #print("<-------------siamese layer is running ------------>")
            self.predictions, self.out1, self.out2 = self.siamese_layer(max_sequence_len, model_cfg)
        
        with tf.variable_scope('loss'):
            if self.loss_function == losses.contrastive_lecun:
                print("loss function is contrastive_lecun!!!")
                self.loss = losses.contrastive_lecun(self.out1, self.out2,self.labels,self.predictions)
            else:
                self.loss = self.loss_function(self.labels,self.predictions)
            self.opt = optimize(self.loss, self.learning_rate)
        
        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.to_float(self.labels))
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))
            self.f1  = self.F1(self.temp_sim, self.labels)
            self.r = self.Recall(self.temp_sim, self.labels)

        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("f1", self.f1)
            self.summary_op = tf.summary.merge_all()
    
    def siamese_layer(self, sequence_len, model_cfg):
        """Implementation of specific siamese layer"""
        raise NotImplementedError()
    @staticmethod
    def F1(y_hat, y_true, mode='multi'):
        epsilon = 1e-7
        #y_hat = tf.round(y_hat)#将经过sigmoid激活的张量四舍五入变为0，1输出
        y_hat = tf.cast(y_hat,tf.float64)
        y_true = tf.cast(y_true,tf.float64)
        
        tp = tf.reduce_sum(tf.cast(y_hat*y_true, 'float'), axis=0)
        #tn = tf.sum(tf.cast((1-y_hat)*(1-y_true), 'float'), axis=0)
        fp = tf.reduce_sum(tf.cast(y_hat*(1-y_true), 'float'), axis=0)
        fn = tf.reduce_sum(tf.cast((1-y_hat)*y_true, 'float'), axis=0)
        
        p = tp/(tp+fp+epsilon)#epsilon的意义在于防止分母为0，否则当分母为0时python会报错
        r = tp/(tp+fn+epsilon)
        
        f1 = 2*p*r/(p+r+epsilon)
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        if mode == 'single':
            return f1
        if mode == 'multi':
            return tf.reduce_mean(f1)
    @staticmethod
    def Recall(y_hat, y_true):
        epsilon = 1e-7
        #y_hat = tf.round(y_hat)#将经过sigmoid激活的张量四舍五入变为0，1输出
        y_hat = tf.cast(y_hat,tf.float64)
        y_true = tf.cast(y_true,tf.float64)
        
        tp = tf.reduce_sum(tf.cast(y_hat*y_true, 'float'), axis=0)
        #tn = tf.sum(tf.cast((1-y_hat)*(1-y_true), 'float'), axis=0)
        fp = tf.reduce_sum(tf.cast(y_hat*(1-y_true), 'float'), axis=0)
        fn = tf.reduce_sum(tf.cast((1-y_hat)*y_true, 'float'), axis=0)
        
        p = tp/(tp+fp+epsilon)#epsilon的意义在于防止分母为0，否则当分母为0时python会报错
        r = tp/(tp+fn+epsilon)
        
        return r
    @staticmethod    
    def metric_fn(labels, predictions):
        pr, pr_op = tf.metrics.precision(labels, predictions)
        re, re_op = tf.metrics.recall(labels, predictions)
        f1 = (2 * pr * re) / (pr + re)
        return f1