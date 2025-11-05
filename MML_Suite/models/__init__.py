# from .avmnist import AVMNIST, MNISTAudio, MNISTImage

# # from .cmams import CMAM, DualCMAM, InputEncoders
# from .conv import ConvBlock, ConvBlockArgs
# from .gates import GatedBiModalNetwork
# from .maxout import MaxOut
# from .mmimdb import GMUModel, MLPGenreClassifier, MMIMDbModalityEncoder
# from .msa import (
#     AuViSubNet,
#     BertTextEncoder,
#     FcClassifier,
#     LSTMEncoder,
#     ResidualAE,
#     Self_MM,
#     TextCNN,
#     UttFusionModel,
#     msa_binarize,
# )
# from .protocols import MultimodalModelProtocol

# __all__ = [
#     "BasicCMAM",
#     "DualCMAM",
#     "MNISTAudio",
#     "MNISTImage",
#     "ConvBlockArgs",
#     "ConvBlock",
#     "AVMNIST",
#     "LSTMEncoder",
#     "TextCNN",
#     "resolve_encoder",
#     "MultimodalModelProtocol",
#     "Self_MM",
#     "UttFusionModel",
#     "msa_binarize",
#     "AuViSubNet",
#     "BertTextEncoder",
#     "InputEncoders",
#     # "CMAM",
#     # "DualCMAM",
#     "FcClassifier",
#     "ResidualAE",
#     "GMUModel",
#     "MLPGenreClassifier",
#     "MMIMDbModalityEncoder",
#     "GatedBiModalNetwork",
#     "MaxOut",
# ]

import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        # Normalize input tensor x along the embedding dimension (dim=1)
        return F.normalize(x, p=2, dim=1)


