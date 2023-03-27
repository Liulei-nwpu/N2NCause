import enum

from data import anli
from data import qqp
from data import snli
from data import Crest
from data import ECSIN
from data import ECMUL
from data import SCISIN
from data import SCIMUL


class DatasetType(enum.Enum):
    SNLI = 0,
    QQP = 1,
    ANLI = 2,
    Crest = 3,
    ECSIN = 4,
    ECMUL = 5,
    SCISIN = 6,
    SCIMUL = 7


DATASETS = {
    DatasetType.QQP.name: qqp.QQPDataset,
    DatasetType.SNLI.name: snli.SNLIDataset,
    DatasetType.ANLI.name: anli.ANLIDataset,
    DatasetType.Crest.name: Crest.CrestDataset,
    DatasetType.ECSIN.name: ECSIN.ECSINDataset,
    DatasetType.ECMUL.name: ECMUL.ECMULDataset,
    DatasetType.SCISIN.name: SCISIN.SCISINDataset,
    DatasetType.SCIMUL.name: SCIMUL.SCIMULDataset,
}

def get_dataset(dataset_name):
    return DATASETS[dataset_name]()