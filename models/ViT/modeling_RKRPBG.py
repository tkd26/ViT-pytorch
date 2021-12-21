# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import math

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from .modeling_resnet import ResNetV2

import sys

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

# K = 2
TASK_NUM = 10
# RG, SFG = True, True

class RG_FC(nn.Module):
    def __init__(self, RG, K, task_num, h_in, h_out, lambdas, scale, bias=True, task=0):
        super().__init__()

        self.RG = RG
        self.task = task
        self.h_in = h_in
        self.h_out = h_out

        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.h_out, self.h_in)))
        if bias:
            self.bias = nn.Parameter(nn.init.normal_(torch.Tensor(self.h_out)))
        else:
            self.register_parameter('bias', None)

        '''
        <Piggyback GANの手法>
        Piggyback GANのunc_filtとweights_matをそれぞれRMとLMから生成する

        出力フィルタ数 = h_out
        非制約フィルタ数 = self.lambdas * h_out
        piggybackフィルタ数 = 出力フィルタ数 - 非制約フィルタ数
        フィルタバンクのフィルタ数 = 出力フィルタ数 + タスク番号 * 非制約フィルタ数

        非制約フィルタ(非制約フィルタ数, h_in)
        重み行列(フィルタバンクのフィルタ数, piggybackフィルタ数)
        フィルタバンク(フィルタバンクのフィルタ数, h_in)

        フィルタバンク x 重み行列(K, piggybackフィルタ数)
        '''

        if self.RG:
            self.out_filt = self.h_out # 出力フィルタ数

            self.lambdas = lambdas # 非制約フィルタの割合
            self.lamb_num = math.ceil(lambdas * self.out_filt) # 非制約フィルタ数
            self.lamb_rem_num = self.out_filt - self.lamb_num # piggybackフィルタ数
            
            self.scale = scale

            if self.task == 0 or self.lambdas == 1.:
                # 非制約フィルタ(非制約フィルタ数, h_in)
                # 非制約フィルタのLM(非制約フィルタ数, K)
                self.unc_filt_LM = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.h_out, K)) * self.scale)
                # 非制約フィルタのRM(K, h_in)
                self.unc_filt_RM = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(K, self.h_in)) * self.scale)

                self.register_parameter('weights_mat_LM', None)
                self.register_parameter('weights_mat_RM', None)
            else:
                # 非制約フィルタ(非制約フィルタ数, h_in)
                # 非制約フィルタのLM(非制約フィルタ数, K)
                self.unc_filt_LM = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.lamb_num, K)) * self.scale)
                # 非制約フィルタのRM(K, h_in)
                self.unc_filt_RM = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(K, self.h_in)) * self.scale)

                # 重み行列((出力チャネル + タスク番号 * 非制約フィルタ数), piggybackフィルタ数)
                # 重み行列のLM((出力チャネル + タスク番号 * 非制約フィルタ数), K)
                self.weights_mat_LM = nn.Parameter(
                    nn.init.kaiming_uniform_(torch.Tensor((self.h_out + (self.task - 1) * self.lamb_num), K)))
                # 重み行列のRM(K, Piggybackフィルタ数)
                self.weights_mat_RM = nn.Parameter(
                    nn.init.kaiming_uniform_(torch.Tensor(K, self.lamb_rem_num)))

                # 過去タスクの非制約フィルタを保存するためのフィルタバンク
                self.register_buffer('concat_unc_filter', torch.Tensor(self.h_out + (task - 1) * self.lamb_num, self.h_in))
                # self.register_buffer('concat_unc_filter', torch.cat(unc_filt_list, dim=0))
    
    def forward(self, x, task: int):
        if self.RG:
            
            if self.task == 0 or self.lambdas == 1.:
                # RMとLMから非制約フィルタを生成（出力チャネル, 入力チャネル）
                self.unc_filt = torch.matmul(self.unc_filt_LM, self.unc_filt_RM).view(self.h_out, self.h_in)
                
                R = self.unc_filt

            else:
                # LMとRMから非制約フィルタを生成（非制約フィルタ数, 入力チャネル）
                self.unc_filt = torch.matmul(self.unc_filt_LM, self.unc_filt_RM).view(self.lamb_num, self.h_in)
                # LMとRMから重み行列を生成（フィルタバンクのフィルタ数, piggybackフィルタ数）
                self.weights_mat = torch.matmul(self.weights_mat_LM, self.weights_mat_RM)

                # フィルタバンクのリシェイプ（入力チャネル, フィルタバンクのフィルタ数）
                self.reshape_unc = torch.reshape(self.concat_unc_filter, (self.h_in, self.concat_unc_filter.shape[0]))
                # フィルタバンクと重み行列の行列積からpiggybackフィルタを作成（入力チャネル, piggybackフィルタ数）
                self.reshape_unc_mul_w = torch.matmul(self.reshape_unc, self.weights_mat)
                # piggybackフィルタのリシェイプ（piggybackフィルタ数, 入力チャネル）
                self.pb_filt = torch.reshape(self.reshape_unc_mul_w, (self.reshape_unc_mul_w.shape[1], self.h_in))
                # 非制約フィルタとpiggybackフィルタを結合（出力フィルタ数, 入力チャネル数）
                self.final_weight_mat = torch.cat((self.unc_filt, self.pb_filt), dim=0)
                
                R = self.final_weight_mat

            weight = R + self.weight
        else:
            weight = self.weight

        return nn.functional.linear(x, weight, bias=self.bias)

class SFG_FC(nn.Module):
    def __init__(self, c_out, task_num: int):
        super().__init__()
        self.SFG_F = nn.Parameter(torch.ones(c_out))
        # self.F_list = nn.ParameterList([nn.Parameter(torch.ones(c_out)) for _ in range(task_num)])
        # self.F_list = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.Tensor(c_out))) for _ in range(task_num)])
    
    def forward(self, x, task: int):
        F = self.SFG_F
        F = F.unsqueeze(0).unsqueeze(0)
        F = F.repeat(x.shape[0], x.shape[1], 1)
        x = x * F
        return x

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis, task):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.query = Linear(config.hidden_size, self.all_head_size)
        # self.key = Linear(config.hidden_size, self.all_head_size)
        # self.value = Linear(config.hidden_size, self.all_head_size)

        # self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

        self.RG, self.SFG = config.RG, config.SFG

        self.query = RG_FC(self.RG, config.K, config.task_num, config.hidden_size, self.all_head_size, config.lamb, config.rkr_scale, task=task)
        self.key = RG_FC(self.RG, config.K, config.task_num, config.hidden_size, self.all_head_size, config.lamb, config.rkr_scale, task=task)
        self.value = RG_FC(self.RG, config.K, config.task_num, config.hidden_size, self.all_head_size, config.lamb, config.rkr_scale, task=task)
        self.out = RG_FC(self.RG, config.K, config.task_num, config.hidden_size, config.hidden_size, config.lamb, config.rkr_scale, task=task)

        if self.SFG:
            self.SFG_query = SFG_FC(self.all_head_size, config.task_num)
            self.SFG_key = SFG_FC(self.all_head_size, config.task_num)
            self.SFG_value = SFG_FC(self.all_head_size, config.task_num)
            self.SFG_out = SFG_FC(self.all_head_size, config.task_num)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, task):

        mixed_query_layer = self.query(hidden_states, task)
        mixed_key_layer = self.key(hidden_states, task)
        mixed_value_layer = self.value(hidden_states, task)

        if self.SFG:
            mixed_query_layer = self.SFG_query(mixed_query_layer, task)
            mixed_key_layer = self.SFG_key(mixed_key_layer, task)
            mixed_value_layer = self.SFG_value(mixed_value_layer, task)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer, task)
        if self.SFG:
            attention_output = self.SFG_out(attention_output, task)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config, task):
        super(Mlp, self).__init__()
        # self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        # self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self.RG, self.SFG = config.RG, config.SFG

        self.fc1 = RG_FC(self.RG, config.K, config.task_num, config.hidden_size, config.transformer["mlp_dim"], config.lamb, config.rkr_scale, task=task)
        self.fc2 = RG_FC(self.RG, config.K, config.task_num, config.transformer["mlp_dim"], config.hidden_size, config.lamb, config.rkr_scale, task=task)

        self._init_weights()

        if self.SFG:
            self.SFG1 = SFG_FC(config.transformer["mlp_dim"], config.task_num)
            self.SFG2 = SFG_FC(config.hidden_size, config.task_num)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x, task):
        x = self.fc1(x, task)
        if self.SFG:
            x = self.SFG1(x, task)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x, task)
        if self.SFG:
            x = self.SFG2(x, task)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis, task):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config, task)
        self.attn = Attention(config, vis, task)

    def forward(self, x, task):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, task)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x, task)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, task):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, task)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, task):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, task)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, task):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, task)

    def forward(self, input_ids, task):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output, task)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, task=0):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.num_tasks = len(num_classes)
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.multi_head = config.multi_head

        self.transformer = Transformer(config, img_size, vis, task)
        self.head = Linear(config.hidden_size, num_classes[task])

    def forward(self, x, labels=None, task=None):
        x, attn_weights = self.transformer(x, task)
        logits = self.head(x[:, 0])

        if type(self.num_classes) == list:
            num_class = self.num_classes[task]
        else:
            num_class = self.num_classes

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_class), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
                # if self.multi_head:
                #     for i in range(self.num_tasks):
                #         nn.init.zeros_(self.head[i].weight)
                #         nn.init.zeros_(self.head[i].bias)
                # else:
                #     nn.init.zeros_(self.head.weight)
                #     nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)