from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.variants.optimizer.nnUnetTrainerAdam import nnUNetTrainerRAdam3en4
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.PanSegMamba import get_pansegmamba_from_plans


class nnUNetTrainerPanSegMamba(nnUNetTrainerRAdam3en4):
    """
    Residual Encoder + UMmaba Bottleneck + Residual Decoder + Skip Connections
    """
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_pansegmamba_from_plans(plans_manager, dataset_json, configuration_manager,
                                      num_input_channels, deep_supervision=enable_deep_supervision)
        
        print("PanSegMamba: {}".format(model))

        return model
    
