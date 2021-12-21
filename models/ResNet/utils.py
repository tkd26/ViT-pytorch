import torch

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Don't remove this file and don't change the imports of load_state_dict_from_url
# from other files. We need this so that we can swap load_state_dict_from_url with
# a custom internal version in fbcode.
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def load_pre_model_state(from_model, to_model):
    from_state_keys = [k for k, v in from_model.state_dict().items()]
    model_dict = to_model.state_dict()
    for k, v in model_dict.items():
        if 'fc' in k:
            continue
        if k in from_state_keys:
            model_dict[k] = from_model.state_dict()[k]
    to_model.load_state_dict(model_dict)
    return to_model

def load_pre_rg_sfg_state(model, task):
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if 'LM_list' in k or 'RM_list' in k or 'M_list' in k:
            if k.split('.')[-1] == str(task):
                to_key = k
                from_key = k.split('.')
                from_key[-1] = str(task - 1)
                from_key = '.'.join(from_key)
                model_dict[to_key] = model_dict[from_key]
    model.load_state_dict(model_dict)
    return model

def load_pre_fc_state(model, task):
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if 'fc_list' in k:
            if k.split('.')[-2] == str(task):
                to_key = k
                from_key = k.split('.')
                from_key[-2] = str(task - 1)
                from_key = '.'.join(from_key)
                model_dict[to_key] = model_dict[from_key]
    model.load_state_dict(model_dict)
    return model