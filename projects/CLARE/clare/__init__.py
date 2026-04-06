from .config import add_clare_config
from .clare_model import CLARE
from .data import build_detection_train_loader, build_detection_test_loader
from .data.objects365 import categories
from .data.objects365_v2 import categories
from .backbone.convnext import D2ConvNeXt
from .backbone.vit import D2ViT
from .backbone.siglip import D2SigLIP
