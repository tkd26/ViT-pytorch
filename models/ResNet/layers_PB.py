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


class ElementWiseConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 config=None, threshold=None):
        super(ElementWiseConv2d, self).__init__()
        # kernel_size = _pair(kernel_size)
        # stride = _pair(stride)
        # padding = _pair(padding)
        # dilation = _pair(dilation)
        self.mask_scale = config.mask_scale
        self.mask_init = config.mask_init

        threshold_fn = config.threshold_fn

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        # weight and bias are no longer Parameters.
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        if self.mask_init == '1s':
            self.mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            self.mask_real.uniform_(-1 * mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.
        # self.mask_real = Parameter(self.mask_real)

        self.mask_reals = nn.ParameterList()
        for _ in range(config.task_num):
            self.mask_reals.append(copy.deepcopy(Parameter(self.mask_real)))

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            # print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            # print('Calling ternarizer with threshold:', threshold)
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input, task):
        # Get binarized/ternarized mask from real-valued mask.
        mask_thresholded = self.threshold_fn.apply(self.mask_reals[task])
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        # weight_thresholded = self.weight
        # Perform conv using modified weight.
        return F.conv2d(input, weight_thresholded, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ElementWiseRG(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 config=None, threshold=None):
        super(ElementWiseRG, self).__init__()
        self.mask_scale = config.mask_scale
        self.mask_init = config.mask_init

        threshold_fn = config.threshold_fn

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            # print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            # print('Calling ternarizer with threshold:', threshold)
            self.threshold_fn = Ternarizer(threshold=threshold)

        # weight and bias are no longer Parameters.
        self.weight = Parameter(nn.init.kaiming_uniform_(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size)))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels))
        else:
            self.register_parameter('bias', None)

        self.scale = config.rkr_scale
        self.RGnoPB = config.RGnoPB
        self.LMnoPB = config.LMnoPB
        self.RMnoPB = config.RMnoPB
        self.PBwR = config.PBwR

        if self.RGnoPB:
            self.LM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.kernel_size * self.in_channels, config.K)) * self.scale) for _ in range(config.task_num)])
            self.RM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(config.K, self.kernel_size * self.out_channels)) * self.scale) for _ in range(config.task_num)])

        elif self.LMnoPB:
            self.LM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.kernel_size * self.in_channels, config.K)) * self.scale) for _ in range(config.task_num)])

            self.RM_base = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(config.K, self.kernel_size * self.out_channels)) * self.scale)
            self.mask_real_RM = self.RM_base.data.new(self.RM_base.size())
            if self.mask_init == '1s':
                self.mask_real_RM.fill_(self.mask_scale)
            elif self.mask_init == 'uniform':
                self.mask_real_RM.uniform_(-1 * self.mask_scale, self.mask_scale)

            self.mask_reals_RM = nn.ParameterList()
            for _ in range(config.task_num - 1):
                self.mask_reals_RM.append(copy.deepcopy(Parameter(self.mask_real_RM)))

        elif self.RMnoPB:
            self.RM_list = nn.ParameterList(
                [nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(config.K, self.kernel_size * self.out_channels)) * self.scale) for _ in range(config.task_num)])

            self.LM_base = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.kernel_size * self.in_channels, config.K)) * self.scale)
            self.mask_real_LM = self.LM_base.data.new(self.LM_base.size())
            if self.mask_init == '1s':
                self.mask_real_LM.fill_(self.mask_scale)
            elif self.mask_init == 'uniform':
                self.mask_real_LM.uniform_(-1 * self.mask_scale, self.mask_scale)

            self.mask_reals_LM = nn.ParameterList()
            for _ in range(config.task_num - 1):
                self.mask_reals_LM.append(copy.deepcopy(Parameter(self.mask_real_LM)))

        else:
            self.LM_base = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.kernel_size * self.in_channels, config.K)) * self.scale)
            self.RM_base = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(config.K, self.kernel_size * self.out_channels)) * self.scale)

            if self.PBwR:
                self.mask_real = torch.Tensor(self.kernel_size * self.in_channels, self.kernel_size * self.out_channels)
                if self.mask_init == '1s':
                    self.mask_real.fill_(self.mask_scale)
                elif self.mask_init == 'uniform':
                    self.mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)

                self.mask_reals = nn.ParameterList()
                for _ in range(config.task_num - 1):
                    self.mask_reals.append(copy.deepcopy(Parameter(self.mask_real)))

            else:
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

    def forward(self, x, task):
        if self.RGnoPB:
            LM = self.LM_list[task]
            RM = self.RM_list[task]
            R = torch.matmul(LM, RM)

        elif self.LMnoPB:
            if task == 0:
                R = torch.matmul(self.LM_list[task], self.RM_base)
            else:
                mask_thresholded_RM = self.threshold_fn.apply(self.mask_reals_RM[task - 1])
                LM =  self.LM_list[task]
                RM =  mask_thresholded_RM * self.RM_base
                R = torch.matmul(LM, RM)

        elif self.RMnoPB:
            if task == 0:
                R = torch.matmul(self.LM_base, self.RM_list[task])
            else:
                mask_thresholded_LM = self.threshold_fn.apply(self.mask_reals_LM[task - 1])
                LM =  mask_thresholded_LM * self.LM_base
                RM = self.RM_list[task]
                R = torch.matmul(LM, RM)
                
        else:
            if task == 0:
                R = torch.matmul(self.LM_base, self.RM_base)
            else:
                if self.PBwR:
                    R = torch.matmul(self.LM_base, self.RM_base)
                    # Get binarized/ternarized mask from real-valued mask.
                    mask_thresholded = self.threshold_fn.apply(self.mask_reals[task - 1])
                    # Mask weights with above mask.
                    R =  mask_thresholded * R

                else:
                    # Get binarized/ternarized mask from real-valued mask.
                    mask_thresholded_LM = self.threshold_fn.apply(self.mask_reals_LM[task - 1])
                    mask_thresholded_RM = self.threshold_fn.apply(self.mask_reals_RM[task - 1])

                    # Mask weights with above mask.
                    # weight_thresholded = mask_thresholded * self.weight
                    LM =  mask_thresholded_LM * self.LM_base
                    RM =  mask_thresholded_RM * self.RM_base

                    R = torch.matmul(LM, RM)

        R = R.view(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        R = R.permute(3, 2, 0, 1)

        weight = R + self.weight

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ElementWiseSFG(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, out_channels, config=None, bias=False, threshold=None):
        super(ElementWiseSFG, self).__init__()
        self.mask_scale = config.mask_scale
        self.mask_init = config.mask_init

        threshold_fn = config.threshold_fn

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            # print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            # print('Calling ternarizer with threshold:', threshold)
            self.threshold_fn = Ternarizer(threshold=threshold)

        self.out_channels = out_channels

        self.SFGnoPB = config.SFGnoPB

        if self.SFGnoPB:
            self.F_list = nn.ParameterList([nn.Parameter(torch.ones(self.out_channels)) for _ in range(config.task_num)])

        else:
            self.F_base = nn.Parameter(torch.ones(self.out_channels))

            # Initialize real-valued mask weights.
            self.mask_real_F = self.F_base.data.new(self.F_base.size())
            if self.mask_init == '1s':
                self.mask_real_F.fill_(self.mask_scale)
            elif self.mask_init == 'uniform':
                self.mask_real_F.uniform_(-1 * self.mask_scale, self.mask_scale)

            self.mask_reals_F = nn.ParameterList()
            for _ in range(config.task_num - 1):
                self.mask_reals_F.append(copy.deepcopy(Parameter(self.mask_real_F)))

    def forward(self, x, task):
        if self.SFGnoPB:
            F = self.F_list[task]
        else:
            if task == 0:
                F = self.F_base
            else:
                # Get binarized/ternarized mask from real-valued mask.
                mask_thresholded_F = self.threshold_fn.apply(self.mask_reals_F[task - 1])

                # Mask weights with above mask.
                # weight_thresholded = mask_thresholded * self.weight
                F = mask_thresholded_F * self.F_base

        F = F.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        F = F.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * F
        
        return x