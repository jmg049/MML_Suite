from __future__ import annotations
from collections.abc import Callable
from typing import Type

from models.protocols import MultimodalModelProtocol
from data import AVMNIST, IEMOCAP, MOSEI, MOSI, MSP_IMPROV, MMIMDb, KineticsSounds
from experiment_utils.printing import get_console
from experiment_utils.logging import get_logger
from experiment_utils.utils import kaiming_init
from torch import optim
from torch.optim import lr_scheduler
from data.base_dataset import MultimodalBaseDataset

logger = get_logger()
console = get_console()


def resolve_model_name(_type: str) -> Type[MultimodalModelProtocol]:
    console.print(f"Resolving {_type}")
    match _type.lower():
        case "avmnist":
            from models.avmnist import AVMNIST

            return AVMNIST
        case "avmnist_corenet":
            from models.corenet import AVMNISTCoreNet
            return AVMNISTCoreNet
        case "self-mm":
            from models.msa.self_mm import Self_MM

            return Self_MM
        case "utt-fusion":
            from models.msa.utt_fusion import UttFusionModel

            return UttFusionModel
        case "mmin":
            from models.msa.mmin import MMINModel as MMIN

            return MMIN
        case "redcore":
            from models.msa.redcore import RedCore

            return RedCore
        case "transformer":
            from models.msa.networks.transformer import Transformer

            return Transformer
        case "mmimdb":
            from models.mmimdb import GMUModel

            return GMUModel
        case "mmimdbmodalityencoder":
            from models.mmimdb import MMIMDbModalityEncoder

            return MMIMDbModalityEncoder
        case "mlp_genre":
            from models.mmimdb import MLPGenreClassifier

            return MLPGenreClassifier
        case "gated_bimodal":
            from models.mmimdb import GatedBiModalNetwork

            return GatedBiModalNetwork
        case "cmam":
            from models.cmams import CMAM

            return CMAM

        case "simple_cmam":
            from models.cmams import SimpleCMAM

            return SimpleCMAM
        
        case "contrastive_simple_cmam":
            from models.contrastive_cmam import ContrastiveSimpleCMAM

            return ContrastiveSimpleCMAM

        case "kineticssounds":
            from models.kinetics_sounds import KineticsSounds

            return KineticsSounds
        case "kinetics_sounds_audio_encoder":
            from models.kinetics_sounds import KineticsSoundsAudioEncoder

            return KineticsSoundsAudioEncoder
        case "kinetics_sounds_video_encoder":
            from models.kinetics_sounds import KineticsSoundsVideoEncoder

            return KineticsSoundsVideoEncoder
        case _:
            raise ValueError(f"Unknown model type: {_type}")


def resolve_init_fn(_type: str) -> Type[Callable]:
    ## TODO : Make their parameters configurable using partial functions and probably a config class
    match _type.lower():
        case "xavier":
            raise NotImplementedError("Xavier initialization not implemented yet")
        case "kaiming":
            return kaiming_init
        case "orthogonal":
            raise NotImplementedError("Orthogonal initialization not implemented yet")
        case _:
            raise ValueError(f"Unknown init function: {_type}")


def resolve_encoder(_type: str):
    from models.msa.networks import LSTMEncoder, TextCNN

    match _type.lower():
        case "lstmencoder":
            return LSTMEncoder
        case "textcnn":
            return TextCNN
        # case "mmimdbmodalityencoder":
        #     return MMIMDbModalityEncoder
        case _:
            raise ValueError(f"Unknown encoder type: {_type}")


def resolve_optimizer(optimizer_name: str) -> Type[optim.Optimizer]:
    """
    Resolve optimizer class from string name.

    Args:
        optimizer_name: Name of the optimizer (case-insensitive)

    Returns:
        Optimizer class
    """
    optimizer_map = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "lbfgs": optim.LBFGS,
        "sparse_adam": optim.SparseAdam,
    }

    optimizer_name = optimizer_name.lower()

    if optimizer_name not in optimizer_map:
        error_msg = f"Unknown optimizer: {optimizer_name}. Available optimizers: {list(optimizer_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved optimizer: {optimizer_name}")
    return optimizer_map[optimizer_name]


def resolve_scheduler(scheduler_name: str) -> Type[lr_scheduler._LRScheduler]:
    """
    Resolve scheduler class from string name.

    Args:
        scheduler_name: Name of the scheduler (case-insensitive)

    Returns:
        Scheduler class
    """
    scheduler_map = {
        "step": lr_scheduler.StepLR,
        "multistep": lr_scheduler.MultiStepLR,
        "exponential": lr_scheduler.ExponentialLR,
        "cosine": lr_scheduler.CosineAnnealingLR,
        "plateau": lr_scheduler.ReduceLROnPlateau,
        "cyclic": lr_scheduler.CyclicLR,
        "onecycle": lr_scheduler.OneCycleLR,
        "cosine_warmup": lr_scheduler.CosineAnnealingWarmRestarts,
        "lambda": lr_scheduler.LambdaLR,
    }

    scheduler_name = scheduler_name.lower()

    if scheduler_name not in scheduler_map:
        error_msg = f"Unknown scheduler: {scheduler_name}. Available schedulers: {list(scheduler_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved scheduler: {scheduler_name}")
    return scheduler_map[scheduler_name]


def resolve_dataset_name(dataset_name: str) -> Type[MultimodalBaseDataset]:
    """
    Resolve dataset class from string

    Args:
        dataset_name: Name of the dataset (case-insensitive)

    Returns:
        Dataset class
    """

    dataset_map = {
        "avmnist": AVMNIST,
        "mosi": MOSI,
        "mosei": MOSEI,
        "iemocap": IEMOCAP,
        "msp_improv": MSP_IMPROV,
        "mm_imdb": MMIMDb,
        "kinetics_sounds": KineticsSounds,
        "contrastive_avmnist": lambda: __import__('data.contrastive_dataset', fromlist=['ContrastiveAVMNIST']).ContrastiveAVMNIST,
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in dataset_map:
        error_msg = f"Unknown dataset: {dataset_name}. Available datasets: {list(dataset_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved dataset: {dataset_name}")
    return dataset_map[dataset_name]
