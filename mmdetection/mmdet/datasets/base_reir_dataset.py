import os.path as osp
from typing import List, Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import is_abs

from ..registry import DATASETS
import collections
import os.path as osp
import random
from typing import Dict, List

import mmengine
from mmengine.dataset import BaseDataset

from mmdet.registry import DATASETS
@DATASETS.register_module()
class BaseReirDataset(BaseDataset):
    """Base dataset for reir.

    Args:
        proposal_file (str, optional): Proposals file path. Defaults to None.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        return_classes (bool): Whether to return class information
            for open vocabulary-based algorithms. Defaults to False.
        caption_prompt (dict, optional): Prompt for captioning.
            Defaults to None.
    """
    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: Dict,
                 split: str = 'train',
                 text_mode: str = 'random',
                 **kwargs):
        self.split = split


        assert text_mode in ['original', 'random', 'concat', 'select_first']
        self.text_mode = text_mode
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs,
        )

    def _join_prefix(self):
        if not mmengine.is_abs(self.ann_file) and self.ann_file:
            self.ann_file = osp.join(self.data_root, self.ann_file)

        return super()._join_prefix()

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        self.annos = mmengine.load(self.ann_file, file_format='pkl')
        self.cat_name2cat_label = {cat_name: i for i, cat_name in enumerate(self.metainfo['classes'])}

        split_annos = [anno for anno in self.annos if anno['split'] == self.split]

        img_prefix = self.data_prefix['img_path']
        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path

        data_list = []
        
        for anno in split_annos:
            data_info = {}
            img_id = anno['image_id']
            ref_anno = anno['ref_anno']
            instances = []
            # cata_dict = {}
            for det_ins in anno['det_anno']:
                instance = {}
                if det_ins.get('ignore', False):
                    continue
                x1, y1, w, h = det_ins['bbox']
                bbox = [x1, y1, x1 + w, y1 + h]
                if det_ins.get('iscrowd', False):
                    instance['ignore_flag'] = 1
                else:
                    instance['ignore_flag'] = 0
                instance['bbox'] = bbox
                instance['bbox_label'] = self.cat_name2cat_label[det_ins['category_name']]
                # cata_dict.setdefault(det_ins['category_name'], []).append(bbox)
                instances.append(instance)

            img_path = join_path(img_prefix, anno['file_name'])
            texts = [x['raw'].lower() for x in anno['sentences']]
            # random select one text
            if self.text_mode == 'random':
                idx = random.randint(0, len(texts) - 1)
                text = [texts[idx]]
            # concat all texts
            elif self.text_mode == 'concat':
                text = [''.join(texts)]
            # select the first text
            elif self.text_mode == 'select_first':
                text = [texts[0]]
            # use all texts
            elif self.text_mode == 'original':
                text = texts
            else:
                raise ValueError(f'Invalid text mode "{self.text_mode}".')
            
            x1, y1, w, h = ref_anno['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            ref_instances = [{
                'mask': ref_anno['segmentation'],
                'bbox': bbox,
                'bbox_label': self.cat_name2cat_label[ref_anno['category_name']],
                'ignore_flag': 0
            }] * len(text)
            
            # cata_instances = []
            # for key, value in cata_dict.items():
            #     cata_instances.append({
            #         'bbox': value,
            #         'cata_name': key,
            #     })

            data_info = {
                'img_path': img_path,
                'img_id': img_id,
                'height': anno['height'],
                'width': anno['width'],
                'ref_instances': ref_instances,
                'instances': instances,
                # 'cata_instances': cata_instances,
                'text': text
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list
