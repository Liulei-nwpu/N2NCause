from enum import Enum
from models.cnn import CnnSiameseNet
from models.rnn import LSTMBasedSiameseNet
from models.multihead_attention import MultiheadAttentionSiameseNet
from models.BiLSTM_MHATT import LSTMATTBasedSiameseNet


class ModelType(Enum):
    multihead = 0,
    rnn = 1,
    cnn = 2,
    bilstm_mhatt = 3



MODELS = {
    ModelType.cnn.name: CnnSiameseNet,
    ModelType.rnn.name: LSTMBasedSiameseNet,
    ModelType.multihead.name: MultiheadAttentionSiameseNet,
    ModelType.bilstm_mhatt.name:LSTMATTBasedSiameseNet
}

