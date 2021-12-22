import ml_collections
from copy import deepcopy
'''
-------------------------------------------------
ResNet
-------------------------------------------------
'''

def get_resnet18_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.RG = False
    config.SFG = False
    config.multi_head = False
    return config

def get_resnet18_MultiHead_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.RG = False
    config.SFG = False
    config.multi_head = True
    return config

def get_resnet18_RKR_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.RG = True
    config.SFG = True
    config.multi_head = True
    config.K = 2
    config.rkr_scale = 1e-1
    return config

def get_resnet18_RKRwoRG_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.RG = False
    config.SFG = True
    config.multi_head = True
    config.K = 2
    config.rkr_scale = 1e-1
    return config

def get_resnet18_RKRwoSFG_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.RG = True
    config.SFG = False
    config.multi_head = True
    config.K = 2
    config.rkr_scale = 1e-1
    return config

def get_resnet18_PB_config():
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.threshold_fn = 'binarizer'
    config.mask_scale = 1e-2
    config.mask_init = '1s'
    return config

def get_resnet18_RKRPB_config():
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.RG = True
    config.SFG = True
    config.multi_head = True
    config.K = 2
    config.rkr_scale = 1e-1
    config.threshold_fn = 'binarizer'
    config.mask_scale = 1e-2
    config.mask_init = '1s'
    return config

def get_resnet18_PBG_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.lamb = 0.25
    return config

'''
-------------------------------------------------
ViT
-------------------------------------------------
'''

class ViTConfig(object):
    def __init__(self):
        config = ml_collections.ConfigDict()
        config.patches = ml_collections.ConfigDict({'size': (16, 16)})
        config.hidden_size = 768
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 3072
        config.transformer.num_heads = 12
        config.transformer.num_layers = 12
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.1
        config.classifier = 'token'
        config.representation_size = None

        self.config = config

    def get_b16_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = False
        output_config.SFG = False
        output_config.multi_head = False
        output_config.K = 2
        return output_config

    def get_b16_MultiHead_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = False
        output_config.SFG = False
        output_config.multi_head = True
        output_config.K = 2
        output_config.lamb = 1.
        output_config.rkr_scale = 1e-1
        return output_config

    def get_b16_RKR_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = True
        output_config.SFG = True
        output_config.multi_head = True
        output_config.K = 2
        output_config.rkr_scale = 1e-1
        return output_config

    def get_b16_RKRPBG_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = True
        output_config.SFG = True
        output_config.multi_head = True
        output_config.lamb = 0.5
        output_config.rkr_scale = 1e-1
        return output_config

    def get_b16_RKRTSN_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = True
        output_config.SFG = False # テストのためにSFGは一旦Falseにする
        output_config.multi_head = True
        output_config.K = 2
        output_config.lamb = 1.0 # distillの割合
        output_config.task_emb_d = 50
        return output_config

    def get_b16_PB_config(self):
        output_config = deepcopy(self.config)
        output_config.multi_head = True
        output_config.threshold_fn = 'binarizer'
        output_config.mask_scale = 1e-2
        output_config.mask_init = '1s'
        return output_config

    def get_b16_RKRPB_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = True
        output_config.SFG = True
        output_config.multi_head = True
        output_config.K = 2
        output_config.rkr_scale = 1e-1
        output_config.threshold_fn = 'binarizer'
        output_config.mask_scale = 1e-2
        output_config.mask_init = '1s'
        return output_config

    def get_b16_PBG_config(self):
        output_config = deepcopy(self.config)
        output_config.multi_head = True
        output_config.lamb = 0.25
        return output_config

    def get_b16_RKRwoRG_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = False
        output_config.SFG = True
        output_config.multi_head = True
        output_config.K = 2
        output_config.rkr_scale = 1e-1
        return output_config

    def get_b16_RKRwoSFG_config(self):
        output_config = deepcopy(self.config)
        output_config.RG = True
        output_config.SFG = False
        output_config.multi_head = True
        output_config.K = 2
        output_config.rkr_scale = 1e-1
        return output_config


'''
-------------------------------------------------
Swin Transformer
-------------------------------------------------
'''


class SwinConfig(object):
    def __init__(self, dataset):
        config = ml_collections.ConfigDict()
        config.use_checkpoint = False

        config.drop_rate = 0.0
        if dataset == 'cifar100':
            config.drop_path_rate = 0.2
        elif dataset == 'imagenet':
            config.drop_path_rate = 0.5
        config.label_smoothing = 0.1

        config.patch_size = 4
        config.in_chans = 3
        # img_sizeは，img_size // 2**len(depths)で偶数になるものじゃないとだめ
        if dataset == 'cifar100':
            # Tiny
            config.embed_dim = 96
            config.depths = [ 2, 2 ] # [ 2, 2, 6, 2 ]
            config.num_heads = [ 3, 6 ] # [ 3, 6, 12, 24 ]
        elif dataset == 'imagenet':
            # Base
            config.embed_dim = 128
            config.depths = [ 2, 2, 18, 2 ]
            config.num_heads = [ 4, 8, 16, 32 ]
        config.window_size = 7
        config.mlp_ratio = 4.
        config.qkv_bias = True
        config.qk_scale = None
        config.ape = False
        config.patch_norm = True

        self.config = config

    def get_swin_config(self, dataset=None):
        output_config = deepcopy(self.config)
        output_config.RG = False
        output_config.SFG = False
        output_config.multi_head = False
        output_config.K = 2
        output_config.rkr_scale = 1e-1
        return output_config

    def get_swin_MultiHead_config(self, dataset=None):
        output_config = deepcopy(self.config)
        output_config.RG = False
        output_config.SFG = False
        output_config.multi_head = True
        output_config.K = 2
        output_config.rkr_scale = 1e-1
        return output_config

    def get_swin_RKR_config(self, dataset=None):
        output_config = deepcopy(self.config)
        output_config.RG = True
        output_config.SFG = True
        output_config.multi_head = True
        output_config.K = 2
        output_config.rkr_scale = 1e-1
        return output_config

    def get_swin_PB_config(self, dataset=None):
        output_config = deepcopy(self.config)
        output_config.multi_head = True
        output_config.threshold_fn = 'binarizer'
        output_config.mask_scale = 1e-2
        output_config.mask_init = '1s'
        return output_config

    def get_swin_PBG_config(self, dataset=None):
        output_config = deepcopy(self.config)
        output_config.multi_head = False
        output_config.lamb = 0.25
        return output_config

def get_config(model_type, dataset=None):
    vit_config = ViTConfig()
    swin_config = SwinConfig(dataset)

    CONFIGS = {
        'ResNet18': get_resnet18_config(),
        'ResNet18_MultiHead': get_resnet18_MultiHead_config(),
        'ResNet18_RKR': get_resnet18_RKR_config(),
        'ResNet18_RKRwoRG': get_resnet18_RKRwoRG_config(),
        'ResNet18_RKRwoSFG': get_resnet18_RKRwoSFG_config(),
        'ResNet18_PB': get_resnet18_PB_config(),
        'ResNet18_RKRPB': get_resnet18_RKRPB_config(),
        'ResNet18_PBG': get_resnet18_PBG_config(),

        'ViT-B_16': vit_config.get_b16_config(),
        'ViT-B_16_MultiHead': vit_config.get_b16_MultiHead_config(),
        'ViT-B_16_RKR': vit_config.get_b16_RKR_config(),
        'ViT-B_16_RKRwoRG': vit_config.get_b16_RKRwoRG_config(),
        'ViT-B_16_RKRwoSFG': vit_config.get_b16_RKRwoSFG_config(),
        'ViT-B_16_RKRTSN': vit_config.get_b16_RKRTSN_config(),
        'ViT-B_16_PB': vit_config.get_b16_PB_config(),
        'ViT-B_16_RKRPB': vit_config.get_b16_RKRPB_config(),
        'ViT-B_16_RKRPBG': vit_config.get_b16_RKRPBG_config(),
        'ViT-B_16_PBG': vit_config.get_b16_PBG_config(),

        'Swin': swin_config.get_swin_config(),
        'Swin_MultiHead': swin_config.get_swin_MultiHead_config(),
        'Swin_RKR': swin_config.get_swin_RKR_config(),
        'Swin_PB': swin_config.get_swin_PB_config(),
        'Swin_PBG': swin_config.get_swin_PBG_config(),
    }

    return CONFIGS[model_type]