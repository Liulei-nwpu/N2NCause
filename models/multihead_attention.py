import tensorflow as tf

from layers.attention import stacked_multihead_attention
from layers.similarity import euclidean_distance, manhattan_similarity
from models.base_model import BaseSiameseNet


class MultiheadAttentionSiameseNet(BaseSiameseNet):
    
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
    
    def siamese_layer(self, sequence_len, model_cfg):
        num_blocks = model_cfg['PARAMS'].getint('num_blocks')
        num_heads = model_cfg['PARAMS'].getint('num_heads')
        use_residual = model_cfg['PARAMS'].getboolean('use_residual')
        dropout_rate = model_cfg['PARAMS'].getfloat('dropout_rate')
        
        out1, self.debug_vars['attentions_x1'] = stacked_multihead_attention(
            self.embedded_x1,
            num_blocks=num_blocks,
            num_heads=num_heads,
            use_residual=use_residual,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
        )
        
        out2, self.debug_vars['attentions_x2'] = stacked_multihead_attention(
            self.embedded_x2,
            num_blocks=num_blocks,
            num_heads=num_heads,
            use_residual=use_residual,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
            reuse=True,
        )
        
        out1 = tf.reduce_sum(out1, axis=1)
        out2 = tf.reduce_sum(out2, axis=1)
        
        return manhattan_similarity(out1, out2), out1, out2
