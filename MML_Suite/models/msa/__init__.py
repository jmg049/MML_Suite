__all__ = [
    "UttFusionModel",
    "msa_binarize",
    "Self_MM",
    "AuViSubNet",
    "LSTMEncoder",
    "TextCNN",
    "ResidualAE",
    "FcClassifier",
    "BertTextEncoder",
]

from .networks.autoencoder import ResidualAE
from .networks.bert_text_encoder import BertTextEncoder
from .networks.classifier import FcClassifier
from .networks.lstm import LSTMEncoder
from .networks.textcnn import TextCNN
from .self_mm import AuViSubNet, Self_MM
from .utt_fusion import UttFusionModel


def msa_binarize(preds, labels):
    test_preds = preds - 1
    test_truth = labels - 1
    non_zeros_mask = test_truth != 0

    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0

    return (
        binary_preds,
        binary_truth,
        non_zeros_mask,
    )
