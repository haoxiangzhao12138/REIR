import torch
import os
from pycocotools.coco import COCO
import mmengine
from collections import defaultdict

refcoco_anno_split = '/home/haoxiangzhao/REIR/dataset/coco/refcoco/refs(unc).p'
refcoco_anno_instance = '/home/haoxiangzhao/REIR/dataset/coco/refcoco/instances.json'
coco_anno = '/public/haoxiangzhao/datasets/coco/annotations/instances_train2014.json'

coco = COCO(coco_anno)

splits = mmengine.load(refcoco_anno_split, file_format='pkl')
instances = mmengine.load(refcoco_anno_instance, file_format='json')

ref_annos = {}
for ref_anno in instances['annotations']:
    anno_id = ref_anno['id']
    ref_annos[anno_id] = ref_anno



idtocat = {}
for cat in coco.dataset['categories']:
    idtocat[cat['id']] = cat['name']

idtoimage = {}
for image in instances['images']:
    idtoimage[image['id']] = image

group_by_image_id = defaultdict(list)
for anno in instances['annotations']:
    cat_id = anno['category_id']
    category = idtocat[cat_id]
    anno['category_name'] = category
    group_by_image_id[anno['image_id']].append(anno)

group_by_image_id = dict(group_by_image_id)

for anno in splits:
    ann_id = anno['ann_id']
    anno['det_anno'] = group_by_image_id[anno['image_id']]
    anno['ref_anno'] = ref_annos[ann_id]

    image_id = anno['image_id']
    image_info = idtoimage[image_id]
    anno['file_name'] = image_info['file_name']
    anno['height'] = image_info['height']
    anno['width'] = image_info['width']



mmengine.dump(splits, '/public/haoxiangzhao/datasets/coco/reircoco/anno.p', file_format='pkl')
pass