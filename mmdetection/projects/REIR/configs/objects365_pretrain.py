_base_ = [
    './objects365.py', './default_runtime.py'
]

hidden_dim=1280

custom_imports = dict(
    imports=['projects.REIR.reir'], allow_failed_imports=False)
model = dict(
    type='REIR',
    num_queries=900,
    num_feature_levels=4,
    with_box_refine=True,
    as_two_stage=True,
    # init_cfg=dict(type='Pretrained', checkpoint='/public/haoxiangzhao/REIR_upload/output/coco_pretrain/epoch_3.pth',
    #             #   override=dict(type='Pretrained', name='backbone')
    #               ),
    dataset_name='coco',
    freeze_text_encoder=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=64,
        ),
    backbone=dict(
        type="Radio",
        with_MHCA=False,
        focus_layers_num=2,
        pre_norm=False,
        hidden_dim=hidden_dim,
        nheads=8,
        dropout=0.1,
        text_focus_prompt=[
            'something',
            'object',
            'scene',
            'relationship',
            'interaction',
            'characteristic',
        ],
        init_cfg=dict(
            type="Pretrained",
        ),
        path='/public/haoxiangzhao/weights/radio/NVlabs_RADIO_main',
        name='radio_model',
        weights='/public/haoxiangzhao/weights/radio/radio_v2.5-h.pth.tar',
        version='radio_v2.5-h',
        progress=True,
        skip_validation=True,
        source='local',
        adaptor_names='clip',
        freeze_backbone=True,
    ),
    neck=[
        dict(
            type='VitFPN',
            in_channels=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
            out_channels=256,
            use_residual=True,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=3
        )
    ],
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='ReirHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        prior_prob= 0.01,
        log_scale= 1.0,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # retrieval_branch=dict(
    #     type='TextRetrievalBranch', 
    #     freeze_encoder=True,
    #     dataset_name='coco',
    #     with_text_prompt = False,
    #     with_cata_retrieval = False,
    #     with_ref_retrieval = True,
    #     loss_retrieval=dict(type='FocalLoss', loss_weight=7.0, reduction='sum'),

    #     assigner=dict(
    #         type='HungarianAssigner',
    #         match_costs=[
    #             # dict(type='FocalLossCost', weight=2.0),
    #             dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
    #             dict(type='IoUCost', iou_mode='giou', weight=2.0)
    #         ]),
    #     kmeans_cluster_centers='/home/haoxiangzhao/REIR/kmeans_cluster_centers.pth'
    #     # assigner=dict(
    #     #     type='MaxIoUAssigner',
    #     #     pos_iou_thr=0.8,
    #     #     neg_iou_thr=0.2,
    #     #     min_pos_iou=0.4,
    #     #     match_low_quality=False,
    #     #     ignore_iof_thr=-1),
    #     # init_cfg=dict(type='Kaiming', layer='text_feature_projection')

    # ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # 'backbone': dict(lr_mult=0.0),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
