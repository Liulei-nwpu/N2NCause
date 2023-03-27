import tensorflow as tf

from layers.recurrent import rnn_layer
from layers.similarity import euclidean_distance, manhattan_similarity
from models.base_model import BaseSiameseNet


class LSTMBasedSiameseNet(BaseSiameseNet):
    
    def __init__(
            self,
            max_sequence_len,
            vocabulary_size,
            main_cfg,
            model_cfg,
    ):
        BaseSiameseNet.__init__(
            self,
            max_sequence_len,
            vocabulary_size,
            main_cfg,
            model_cfg,
        )
    
    def siamese_layer(
            self,
            sequence_len,
            model_cfg,
    ):
        hidden_size = model_cfg['PARAMS'].getint('hidden_size')
        cell_type = model_cfg['PARAMS'].get('cell_type')
          
        outputs_sen1 = rnn_layer(
            embedded_x=self.embedded_x1,
            hidden_size=hidden_size,
            bidirectional=True,
            cell_type=cell_type,
        )
        outputs_sen2 = rnn_layer(
            embedded_x=self.embedded_x2,
            hidden_size=hidden_size,
            bidirectional=True,
            cell_type=cell_type,
            reuse=True,
        )
        
        out1 = tf.reduce_mean(outputs_sen1, axis=1)
        out2 = tf.reduce_mean(outputs_sen2, axis=1)
        
        return manhattan_similarity(out1, out2),out1, out2
