# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_clare_config(cfg):
    """
    Add config for CLARE.
    """
    # Unification of detection & grounding
    cfg.UNI = True # Unified detection & grounding
    cfg.MODEL.DECOUPLE_TGT = False # detection and grounding use different tgt (nn.Embedding vs Language)
    cfg.MODEL.STILL_TGT_FOR_BOTH = False # both detection and grounding use still (learnable) tgt
    cfg.MODEL.CLS_POOL_TYPE = "average" # average, max
    cfg.MODEL.USE_IOU_BRANCH = False # add an IoU branch parallel to cls head
    cfg.MODEL.PARALLEL_DET = False # parallel formulation for object detection
    cfg.MODEL.OTA = False
    cfg.MODEL.FREEZE_TEXT_ENCODER = False # freeze the text encoder
    cfg.MODEL.RETRIEVAL = False
    cfg.MODEL.RETRIEVAL_GROUNDING = False
    cfg.MODEL.USE_BIAS = True
    cfg.MODEL.USE_BATCH_LOSS = False
    cfg.MODEL.USE_SIGLIP_LOSS = False
    cfg.MODEL.USE_CONTRASTIVE_FOCAL_LOSS = False
    cfg.MODEL.USE_CLIP_POOL_HEAD = False
    cfg.MODEL.DO_LN = False
    cfg.MODEL.VISUALIZE = False
    cfg.MODEL.USE_MULTITHREADING = False
    cfg.INPUT.OPENCLIP_INPUT = False
    cfg.MODEL.OBJECT_QUERY_TOPK = 20
    cfg.MODEL.OBJECT_QUERY_THRES = 0.0

    # ReID head
    cfg.DETACH_REID = False # whether to detach reid
    cfg.USE_DEFORMABLE_REID_HEAD = False
    cfg.N_LAYER_DEFORMABLE_REID = 2

    cfg.DATASETS.TRAIN = [] # replace tuple with List


    # Unified dataloader for multiple tasks
    # cfg.DATALOADER.SAMPLER_TRAIN = "MultiDatasetSampler"
    cfg.DATALOADER.DATASET_RATIO = [1, 1]
    cfg.DATALOADER.USE_DIFF_BS_SIZE = True
    cfg.DATALOADER.DATASET_BS = [2, 2]
    cfg.DATALOADER.USE_RFS = [False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = True
    cfg.DATALOADER.DATASET_ANN = ['box', 'image']

    # Allow different datasets to use different input resolutions
    cfg.INPUT.MIN_SIZE_TRAIN_MULTI = [(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), (320, 352, 392, 416, 448, 480, 512, 544, 576, 608, 640)]
    cfg.INPUT.MAX_SIZE_TRAIN_MULTI = [1333, 768]

    # BoxInst
    cfg.MODEL.BOXINST = CN()
    cfg.MODEL.BOXINST.ENABLED = False # Whether to enable BoxInst
    cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10
    cfg.MODEL.BOXINST.PAIRWISE = CN()
    cfg.MODEL.BOXINST.PAIRWISE.SIZE = 3
    cfg.MODEL.BOXINST.PAIRWISE.DILATION = 2
    cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
    cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3
    cfg.MODEL.BOXINST.TOPK = 64 # max number of proposals for computing mask loss

    cfg.MODEL.LANG_GUIDE_DET = True # Language-guided detection (similar to GLIP)
    cfg.MODEL.VL_FUSION_USE_CHECKPOINT = True # Use gradient checkpoint for VL Early Fusion
    cfg.MODEL.USE_EARLY_FUSION = True # Use Early Fusion (Bidirectional Cross-Modal Attention)
    cfg.MODEL.USE_ADDITIONAL_BERT = False # Use additional BERT Layers in early fusion
    cfg.MODEL.LANG_AS_CLASSIFIER = True # Use language embedding as classifier 
    cfg.MODEL.STILL_CLS_FOR_ENCODER = False # Use still classifier for encoder

    cfg.MODEL.LANGUAGE_BACKBONE = CN()
    cfg.MODEL.LANGUAGE_BACKBONE.NAME = "bert"
    cfg.MODEL.LANGUAGE_BACKBONE.PRETRAINED = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT = False
    cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE = "projects/CLARE/bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 768
    cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN = 256 # max length of the tokenized captions. 
    cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS = 1
    cfg.MODEL.LANGUAGE_BACKBONE.UNUSED_TOKEN = 106
    cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False
    cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX = True

    cfg.MODEL.DYHEAD = CN()
    cfg.MODEL.DYHEAD.PRIOR_PROB = 0.01
    cfg.MODEL.DYHEAD.LOG_SCALE = 0.0
    cfg.MODEL.DYHEAD.FUSE_CONFIG = CN()
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = True
    cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D = False
    cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = True

    # MoE
    cfg.MODEL.MOE = CN()
    cfg.MODEL.MOE.WEIGHT = 0.2
    cfg.MODEL.MOE.USE_MOE = False
    cfg.MODEL.MOE.NUM_EXPERTS = 4
    cfg.MODEL.MOE.TOP_K = 2
    cfg.MODEL.MOE.HIDDEN_DIM = 1024
    cfg.MODEL.MOE.DROP_OUT = 0.1

    # DataLoader
    cfg.INPUT.DATASET_MAPPER_NAME = "detr" # use "coco_instance_lsj" for LSJ aug
    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.IMAGE_SIZE_LARGE = 1024 # Larger input size (1536)
    # mixup
    cfg.INPUT.USE_MIXUP = False
    cfg.INPUT.MIXUP_PROB = 1.0
    
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    cfg.MODEL.DDETRS = CN()
    cfg.MODEL.DDETRS.NUM_CLASSES = None
    cfg.MODEL.DDETRS.USE_CHECKPOINT = False # whether to use gradient-checkpoint for the transformer
    # LOSS
    cfg.MODEL.DDETRS.MASK_WEIGHT = 2.0
    cfg.MODEL.DDETRS.DICE_WEIGHT = 5.0
    cfg.MODEL.DDETRS.GIOU_WEIGHT = 2.0
    cfg.MODEL.DDETRS.L1_WEIGHT = 5.0
    cfg.MODEL.DDETRS.CLASS_WEIGHT = 2.0
    cfg.MODEL.DDETRS.BATCH_LOSS_WEIGHT = 4.0
    cfg.MODEL.DDETRS.TEXT_IMAGE_LOSS_WEIGHT = 4.0
    cfg.MODEL.DDETRS.SIGLIP_LOSS_WEIGHT = 4.0
    cfg.MODEL.DDETRS.CONTRASTIVE_FOCAL_LOSS_WEIGHT = 4.0
    cfg.MODEL.DDETRS.REID_WEIGHT = 2.0
    cfg.MODEL.DDETRS.DEEP_SUPERVISION = True
    cfg.MODEL.DDETRS.MASK_STRIDE = 4
    cfg.MODEL.DDETRS.MATCH_STRIDE = 4
    cfg.MODEL.DDETRS.FOCAL_ALPHA = 0.25
    # COST
    cfg.MODEL.DDETRS.SET_COST_CLASS = 2
    cfg.MODEL.DDETRS.SET_COST_BOX = 5
    cfg.MODEL.DDETRS.SET_COST_GIOU = 2
    # TRANSFORMER
    cfg.MODEL.DDETRS.NHEADS = 8
    cfg.MODEL.DDETRS.DROPOUT = 0.1
    cfg.MODEL.DDETRS.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DDETRS.ENC_LAYERS = 6
    cfg.MODEL.DDETRS.DEC_LAYERS = 6
    cfg.MODEL.DDETRS.NUM_VL_LAYERS = 1 # one layer for early fusion is enough
    cfg.MODEL.DDETRS.VL_HIDDEN_DIM = 2048 # embed_dim of BiAttentionBlock
    cfg.MODEL.DDETRS.TWO_STAGE = False
    cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS = 300
    cfg.MODEL.DDETRS.MIXED_SELECTION = False
    cfg.MODEL.DDETRS.LOOK_FORWARD_TWICE = False
    cfg.MODEL.DDETRS.CTRL_LAYERS = 3
    cfg.MODEL.DDETRS.USE_DINO = False
    cfg.MODEL.DDETRS.DYNAMIC_LABEL_ENC = False
    cfg.MODEL.DDETRS.HIDDEN_DIM = 256
    cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.DDETRS.DEC_N_POINTS = 4
    cfg.MODEL.DDETRS.ENC_N_POINTS = 4
    cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS = 4
    # Mask Postprocessing & Upsampling
    cfg.MODEL.DDETRS.MASK_THRES = 0.5
    cfg.MODEL.DDETRS.NEW_MASK_HEAD = False
    cfg.MODEL.DDETRS.USE_RAFT = False
    cfg.MODEL.DDETRS.USE_REL_COORD = True
    # Denoising
    cfg.MODEL.DDETRS.DN_NUMBER = 100
    cfg.MODEL.DDETRS.LABEL_NOISE_RATIO = 0.5
    cfg.MODEL.DDETRS.BOX_NOISE_SCALE = 1.0

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.LINEAR_PROJ_MULTIPLIER = 0.1
    cfg.SOLVER.LANG_LR = 0.00001 # 1e-5
    cfg.SOLVER.VL_LR = 0.00001 # 1e-5
    cfg.SOLVER.LOSS_WEIGHT_DET = 1.0
    cfg.SOLVER.LOSS_WEIGHT_GRD = 1.0
    cfg.SOLVER.LOSS_WEIGHT_SOT = 1.0
    
    # R50 backbone
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
    # supprt ConvNeXt backbone
    cfg.MODEL.CONVNEXT = CN()
    cfg.MODEL.CONVNEXT.NAME = "tiny"
    cfg.MODEL.CONVNEXT.DROP_PATH_RATE = 0.7
    cfg.MODEL.CONVNEXT.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.CONVNEXT.USE_CHECKPOINT = False
    # supprt ViT backbone
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.NAME = "ViT-Base"
    cfg.MODEL.VIT.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.VIT.USE_CHECKPOINT = False
    cfg.MODEL.VIT.MODEL_TYPE = "ViT-Base"

    
    cfg.FIND_UNUSED_PARAMETERS = True # find_unused_parameters
    cfg.TEST.EVAL_AFTER_TRAIN = True # eval after train
