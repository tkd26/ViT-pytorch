"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

import copy

DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Binarizer, self).__init__()
        self.threshold = threshold

    @staticmethod
    def forward(self, inputs):
        threshold = DEFAULT_THRESHOLD

        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    @staticmethod
    def forward(self, inputs):
        threshold = DEFAULT_THRESHOLD

        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > threshold] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput

class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, config, bias=True, threshold=None):
        super(ElementWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = config.threshold_fn
        self.mask_scale = config.mask_scale
        self.mask_init = config.mask_init

        # 常にDEFAULT_THRESHOLDを使用するようになってる
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.threshold = threshold

        self.info = {
            'threshold_fn': config.threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
        self.weight = nn.Parameter(torch.Tensor(
            out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        if self.mask_init == '1s':
            self.mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            self.mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.
        # self.mask_real = Parameter(self.mask_real)

        self.mask_reals = nn.ParameterList()
        for _ in range(config.task_num):
            self.mask_reals.append(copy.deepcopy(Parameter(self.mask_real)))

        # Initialize the thresholder.
        if config.threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer(threshold=threshold)
        elif config.threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input, task):
        # Get binarized/ternarized mask from real-valued mask.
        mask_thresholded = self.threshold_fn.apply(self.mask_reals[task])
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        # Get output using modified weight.
        return F.linear(input, weight_thresholded, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)


class ElementWiseRG(nn.Module):
    def __init__(self, in_features, out_features, config, bias=True, threshold=None, task=None):
        super(ElementWiseRG, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = config.threshold_fn
        self.mask_scale = config.mask_scale
        self.mask_init = config.mask_init

        # 常にDEFAULT_THRESHOLDを使用するようになってる
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.threshold = threshold

        self.info = {
            'threshold_fn': config.threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(
            out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(nn.init.normal_(torch.Tensor(
                out_features)))
        else:
            self.register_parameter('bias', None)

        self.scale = config.rkr_scale
        K = config.K
        self.LM_base = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(in_features, K)) * self.scale)
        self.RM_base = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(K, out_features)) * self.scale)

        # Initialize real-valued mask weights.
        self.mask_real_LM = self.LM_base.data.new(self.LM_base.size())
        self.mask_real_RM = self.RM_base.data.new(self.RM_base.size())
        if self.mask_init == '1s':
            self.mask_real_LM.fill_(self.mask_scale)
            self.mask_real_RM.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            self.mask_real_LM.uniform_(-1 * self.mask_scale, self.mask_scale)
            self.mask_real_RM.uniform_(-1 * self.mask_scale, self.mask_scale)

        self.mask_reals_LM = nn.ParameterList()
        self.mask_reals_RM = nn.ParameterList()
        for _ in range(config.task_num - 1):
            self.mask_reals_LM.append(copy.deepcopy(Parameter(self.mask_real_LM)))
            self.mask_reals_RM.append(copy.deepcopy(Parameter(self.mask_real_RM)))

        # Initialize the thresholder.
        if config.threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer(threshold=threshold)
        elif config.threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input, task):

        if task == 0:
            R = torch.matmul(self.LM_base, self.RM_base)
            R = R.permute(1, 0)
        else:
            # Get binarized/ternarized mask from real-valued mask.
            mask_thresholded_LM = self.threshold_fn.apply(self.mask_reals_LM[task - 1])
            mask_thresholded_RM = self.threshold_fn.apply(self.mask_reals_RM[task - 1])

            # Mask weights with above mask.
            # weight_thresholded = mask_thresholded * self.weight
            LM =  mask_thresholded_LM * self.LM_base
            RM =  mask_thresholded_RM * self.RM_base

            R = torch.matmul(LM, RM)
            R = R.permute(1, 0)

        weight = R + self.weight

        return F.linear(input, weight, self.bias)



class ElementWiseSFG(nn.Module):
    def __init__(self, out_features, config, threshold=None, task=None):
        super(ElementWiseSFG, self).__init__()
        self.out_features = out_features
        self.threshold_fn = config.threshold_fn
        self.mask_scale = config.mask_scale
        self.mask_init = config.mask_init

        # 常にDEFAULT_THRESHOLDを使用するようになってる
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.threshold = threshold

        self.info = {
            'threshold_fn': config.threshold_fn,
            'threshold': threshold,
        }

        self.F_base = nn.Parameter(torch.ones(out_features))

        # Initialize real-valued mask weights.
        self.mask_real_F = self.F_base.data.new(self.F_base.size())
        if self.mask_init == '1s':
            self.mask_real_F.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            self.mask_real_F.uniform_(-1 * self.mask_scale, self.mask_scale)

        self.mask_reals_F = nn.ParameterList()
        for _ in range(config.task_num - 1):
            self.mask_reals_F.append(copy.deepcopy(Parameter(self.mask_real_F)))

        # Initialize the thresholder.
        if config.threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer(threshold=threshold)
        elif config.threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, x, task):

        if task == 0:
            F = self.F_base
        else:
            # Get binarized/ternarized mask from real-valued mask.
            mask_thresholded_F = self.threshold_fn.apply(self.mask_reals_F[task - 1])

            # Mask weights with above mask.
            # weight_thresholded = mask_thresholded * self.weight
            F = mask_thresholded_F * self.F_base

        F = F.unsqueeze(0).unsqueeze(0)
        F = F.repeat(x.shape[0], x.shape[1], 1)
        x = x * F

        return x
