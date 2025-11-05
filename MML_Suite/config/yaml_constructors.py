import yaml
from config.federated_configs.federated_data_config import FedDataConfig
from fed import ClientSelectionStrategy, Aggregation, DistributionStrategy, DataSplitType, FedDataCondition, FedLearningType
from models.kinetics_sounds import KineticsSoundsAudioEncoder, KineticsSoundsVideoEncoder
from config.cmam_config import CMAMConfig
from config.data_config import DataConfig, DatasetConfig, ModalityConfig, MissingPatternConfig
from config.experiment_config import ExperimentConfig
from config.logging_config import LoggingConfig
from config.metric_config import MetricConfig
from config.model_config import ModelConfig
from config.multimodal_training_config import StandardMultimodalConfig, TrainingConfig
from config.optimizer_config import OptimizerConfig, ParameterGroupConfig
from experiment_utils.logging import get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.managers import CenterManager, FeatureManager, LabelManager
from modalities import add_modality
from models.avmnist import MNISTAudio, MNISTImage
from models.conv import ConvBlock, ConvBlockArgs
from models.maxout import MaxOut
from models.mmimdb import GMUModel, MLPGenreClassifier, MMIMDbModalityEncoder, GatedBiModalNetwork
from models.msa.networks.autoencoder import ResidualAE, ResidualXE
from models.msa.networks.bert_text_encoder import BertTextEncoder
from models.msa.networks.classifier import FcClassifier
from models.msa.networks.lstm import LSTMEncoder
from models.msa.networks.textcnn import TextCNN
from models.msa.self_mm import AuViSubNet, Self_MM
from models.msa.utt_fusion import UttFusionModel
from models.msa.networks.transformer import Transformer
from models.cmams import AssociationNetwork, InputEncoders
from config.federated_configs import FederatedExperimentConfig, FederatedConfig

logger = get_logger()


# YAML constructors
def register_constructor(tag, cls, from_dict=False, deep=False):
    def constructor(loader, node):
        data = loader.construct_mapping(node, deep=deep)
        return cls.from_dict(data) if from_dict else cls(**data)

    yaml.SafeLoader.add_constructor(tag, constructor)
    logger.debug(f"Added {cls.__name__} constructor to YAML Loader.")


# Specific YAML constructor functions
def register_scalar_constructor(tag, func):
    def constructor(loader, node):
        value = loader.construct_scalar(node) if func == add_modality else loader.construct_mapping(node)
        return func(value)

    yaml.SafeLoader.add_constructor(tag, constructor)
    logger.debug(f"Added {func.__name__} constructor to YAML Loader.")


# Registering YAML constructors for various configurations
register_constructor("!DatasetConfig", DatasetConfig, from_dict=True, deep=True)
register_constructor("!DataConfig", DataConfig, from_dict=True, deep=True)
register_constructor("!MetricConfig", MetricConfig, from_dict=True)
register_constructor("!LoggingConfig", LoggingConfig)
register_constructor("!ModelConfig", ModelConfig, from_dict=True)
register_constructor("!ExperimentConfig", ExperimentConfig, from_dict=True)
register_constructor("!StandardConfig", StandardMultimodalConfig)


register_constructor("!FederatedExperimentConfig", FederatedExperimentConfig)
register_constructor("!FedConfig", FederatedConfig, from_dict=True, deep=True)
register_constructor("!ClientSelection", ClientSelectionStrategy)
register_constructor("!Aggregation", Aggregation, from_dict=True, deep=True)
register_constructor("!FedDataConfig", FedDataConfig, from_dict=True, deep=True)


register_constructor("!ParameterGroupConfig", ParameterGroupConfig)
register_constructor("!Optimizer", OptimizerConfig, from_dict=True, deep=True)
register_constructor("!CMAMConfig", CMAMConfig, deep=True)
register_constructor("!AssociationNetwork", cls=AssociationNetwork, from_dict=True, deep=True)
register_constructor("!InputEncoders", cls=InputEncoders, from_dict=True, deep=True)
# Registering constructors for models and other components

register_scalar_constructor("!DataSplitType", DataSplitType)
register_scalar_constructor("!FedDataCondition", FedDataCondition)
register_scalar_constructor("!FedLearningType", FedLearningType)

register_scalar_constructor("!Modality", add_modality)
register_constructor(
    "!MNISTAudio",
    MNISTAudio,
    deep=True,
)
register_constructor(
    "!MNISTImage",
    MNISTImage,
    deep=True,
)
register_constructor(
    "!ConvBlockArgs",
    ConvBlockArgs,
    deep=True,
)
register_constructor("!ConvBlock", ConvBlock, deep=True)

register_constructor(
    "!Self_MM",
    Self_MM,
    deep=True,
)
register_constructor(
    "!AuViSubNet",
    AuViSubNet,
    deep=True,
)
register_constructor(
    "!LSTMEncoder",
    LSTMEncoder,
    deep=True,
)
register_constructor(
    "!TextCNN",
    TextCNN,
    deep=True,
)
register_constructor(
    "!FcClassifier",
    FcClassifier,
    deep=True,
)
register_constructor(
    "!ResidualAE",
    ResidualAE,
    deep=True,
)
register_constructor("!LossFunctionGroup", LossFunctionGroup, from_dict=True, deep=True)
register_constructor(
    "!UttFusionModel",
    UttFusionModel,
    deep=True,
)
register_constructor("!Transformer", Transformer, deep=True)
register_constructor("!ResidualXE", ResidualXE, deep=True)
register_constructor(
    "!MMIMDbModalityEncoder",
    MMIMDbModalityEncoder,
    deep=True,
)
register_constructor(
    "!MaxOut",
    MaxOut,
    deep=True,
)
register_constructor(
    "!GatedBiModalNetwork",
    GatedBiModalNetwork,
    deep=True,
)
register_constructor(
    "!GMUModel",
    GMUModel,
    deep=True,
)
register_constructor(
    "!MLPGenreClassifier",
    MLPGenreClassifier,
    deep=True,
)
register_constructor("!KineticsSoundsAudioEncoder", cls=KineticsSoundsAudioEncoder, deep=True)
register_constructor("!KineticsSoundsVideoEncoder", cls=KineticsSoundsVideoEncoder, deep=True)


register_constructor(
    "!FeatureManager",
    FeatureManager,
    deep=True,
)
register_constructor("!CenterManager", CenterManager, deep=True)
register_constructor("!LabelManager", LabelManager, deep=True)
register_constructor("!BertTextEncoder", BertTextEncoder)
register_constructor("!MissingPatternConfig", MissingPatternConfig)
register_constructor("!ModalityConfig", ModalityConfig)

# register_constructor("!AssociationNetworkConfig", AssociationNetworkConfig)
# register_constructor("!InputEncoders", InputEncoders)
# register_constructor("!CMAM", CMAM)
# register_constructor("!CMAMConfig", CMAMConfig)

logger.debug("All YAML constructors registered successfully.")
