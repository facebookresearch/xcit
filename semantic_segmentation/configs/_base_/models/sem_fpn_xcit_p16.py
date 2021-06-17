# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='XCiT',
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
    ),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=384,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        feature_strides=[16, 16, 16, 16],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
