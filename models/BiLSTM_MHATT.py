import tensorflow as tf

from layers.attention import stacked_multihead_attention
from layers.recurrent import rnn_layer
from layers.similarity import euclidean_distance, manhattan_similarity
from models.base_model import BaseSiameseNet


class LSTMATTBasedSiameseNet(BaseSiameseNet):
    
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
        #print(model_cfg)
        hidden_size = model_cfg['PARAMS'].getint('hidden_size')
        cell_type = model_cfg['PARAMS'].get('cell_type')
        num_blocks = model_cfg['PARAMS'].getint('num_blocks')
        num_heads = model_cfg['PARAMS'].getint('num_heads')
        use_residual = model_cfg['PARAMS'].getboolean('use_residual')
        dropout_rate = model_cfg['PARAMS'].getfloat('dropout_rate')

        BiLSTM_sen1 = rnn_layer(
            embedded_x=self.embedded_x1,
            hidden_size=hidden_size,
            bidirectional=True,
            cell_type=cell_type,
        )
        BiLSTM_sen2 = rnn_layer(
            embedded_x=self.embedded_x2,
            hidden_size=hidden_size,
            bidirectional=True,
            cell_type=cell_type,
            reuse=True,
        )
        
        ATT_out1, self.debug_vars['attentions_x1'] = stacked_multihead_attention(
            BiLSTM_sen1,
            num_blocks=num_blocks,
            num_heads=num_heads,
            use_residual=use_residual,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
        )
        
        ATT_out2, self.debug_vars['attentions_x2'] = stacked_multihead_attention(
            BiLSTM_sen2,
            num_blocks=num_blocks,
            num_heads=num_heads,
            use_residual=use_residual,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
            reuse=True,
        )

        out1 = tf.reduce_mean(ATT_out1, axis=1)
        out2 = tf.reduce_mean(ATT_out2, axis=1)
        
        return manhattan_similarity(out1, out2), out1, out2
