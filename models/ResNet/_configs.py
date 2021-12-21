import ml_collections

def get_resnet18_RKR_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.model = ml_collections.ConfigDict()
    config.RG = True
    config.SFG = True
    config.multi_head = True
    config.K = 2
    config.task_num = 10
    config.class_num = 10
    config.rkr_scale = 1e-1
    return config