#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.models.effidehead import Detect, build_effidehead_layer


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        #self.mode = config.training_mode
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, anchors, num_layers)

        # Init Detect head
        begin_indices = config.model.head.begin_indices
        out_indices_head = config.model.head.out_indices
        self.stride = self.detect.stride
        self.detect.i = begin_indices
        self.detect.f = out_indices_head
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export()
        x = self.backbone(x)
        x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode is True else [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, anchors, num_layers):
    '''add mobilenet'''
    if 'MobileNetV1' in config.model.backbone.type:
        planes_list = config.model.backbone.planes_list
    '''add mobilenetv2'''
    if 'MobileNetV2' in config.model.backbone.type:
        cfgs = config.model.backbone.cfgs
    '''add mobilevit_xxs'''
    if 'MobileViT' in config.model.backbone.type and 'xxs' in config.model.backbone.size:
        dims = config.model.backbone.dims
        channels = config.model.backbone.channels
        img_size = config.model.backbone.img_size
        expansion = config.model.backbone.expansion
    '''add mobileViT_xs'''
    if 'MobileViT' in config.model.backbone.type and 'xxx' in config.model.backbone.size:
        dims = config.model.backbone.dims
        channels = config.model.backbone.channels
        img_size = config.model.backbone.img_size
    '''add mobileViT_s'''
    if 'MobileViT' in config.model.backbone.type and 'sss' in config.model.backbone.size:
        dims = config.model.backbone.dims
        channels = config.model.backbone.channels
        img_size = config.model.backbone.img_size
    '''add mobileViTV2'''
    if 'MobileViTV2' in config.model.backbone.type and 'v2-xs' in config.model.backbone.size:
        dims = config.model.backbone.dims
        channels = config.model.backbone.channels
        img_size = config.model.backbone.img_size
    if 'MobileViTV2' in config.model.backbone.type and 'v2-s' in config.model.backbone.size:
        dims = config.model.backbone.dims
        channels = config.model.backbone.channels
        img_size = config.model.backbone.img_size

    '''neck && backbone parameters'''
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    '''head parameters'''
    num_anchors = config.model.head.anchors
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    
    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e
        )
    elif 'MobileNetV2' in config.model.backbone.type:
        backbone = BACKBONE(
            cfgs=cfgs
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    elif 'MobileNetV1' in config.model.backbone.type:
        backbone = BACKBONE(
            planes_list=planes_list
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    elif 'MobileViT' in config.model.backbone.type and 'xxs' in config.model.backbone.size:     #vit-xxs
        backbone = BACKBONE(
            image_size=img_size,
            dims=dims,
            channels=channels,
            expansion=expansion
        )

        neck =  NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    elif 'MobileViT' in config.model.backbone.type and 'xxx' in config.model.backbone.size:      #vit-xs
        backbone = BACKBONE(
            image_size=img_size,
            dims=dims,
            channels=channels
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    elif 'MobileViT' in config.model.backbone.type and 'sss' in config.model.backbone.size:        #vit-s
        backbone = BACKBONE(
            image_size=img_size,
            dims=dims,
            channels=channels
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    elif 'MobileViTV2' in config.model.backbone.type and 'v2-xs' in config.model.backbone.size:           #vitv2-xs
        backbone = BACKBONE(
            image_size=img_size,
            dims=dims,
            channels=channels
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    elif 'MobileViTV2' in config.model.backbone.type and 'v2-s' in config.model.backbone.size:           #vitv2-s
        backbone = BACKBONE(
            image_size=img_size,
            dims=dims,
            channels=channels
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    head_layers = build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max)

    head = Detect(num_classes, anchors, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def build_model(cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes, anchors=cfg.model.head.anchors).to(device)
    return model
