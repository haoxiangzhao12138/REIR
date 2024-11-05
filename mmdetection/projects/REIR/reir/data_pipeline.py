from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

from mmcv.transforms import Resize as MMCV_Resize
from mmcv.image.geometric import _scale_size
from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes

@TRANSFORMS.register_module()
class LoadReirAnnotations(MMCV_LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in instance/panoptic segmentation. The segmentation mask
                # of the instance or the information of segments.
                # 1. If list[list[float]], it represents a list of polygons,
                # one for each connected component of the object. Each
                # list[float] is one simple polygon in the format of
                # [x1, y1, ..., xn, yn] (n >= 3). The Xs and Ys are absolute
                # coordinates in unit of pixels.
                # 2. If dict, it represents the per-pixel segmentation mask in
                # COCO's compressed RLE format. The dict should have keys
                # “size” and “counts”.  Can be loaded by pycocotools
                'mask': list[list[float]] or dict,

                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height
    - width
    - instances

      - bbox (optional)
      - bbox_label
      - mask (optional)
      - ignore_flag

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        poly2mask (bool): Whether to convert mask to bitmap. Default: True.
        box_type (str): The box type used to wrap the bboxes. If ``box_type``
            is None, gt_bboxes will keep being np.ndarray. Defaults to 'hbox'.
        reduce_zero_label (bool): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to False.
        ignore_index (int): The label index to be ignored.
            Valid only if reduce_zero_label is true. Defaults is 255.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
            self,
            with_mask: bool = False,
            with_det_instances: bool = True,
            poly2mask: bool = True,
            box_type: str = 'hbox',
            # use for semseg
            reduce_zero_label: bool = False,
            ignore_index: int = 255,
            **kwargs) -> None:
        super(LoadReirAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.with_det_instances = with_det_instances
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int,
                   img_w: int) -> np.ndarray:
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon) for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and \
                    not (gt_mask.get('counts') is not None and
                         gt_mask.get('size') is not None and
                         isinstance(gt_mask['counts'], (list, str))):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            # fake polygon masks will be ignored in `PackDetInputs`
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if results.get('seg_map_path', None) is None:
            return

        img_bytes = get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = self.ignore_index
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == self.ignore_index -
                            1] = self.ignore_index

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['ignore_index'] = self.ignore_index

    def _load_ref_instances(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        ref_gt_bboxes = []
        ref_gt_bboxes_labels = []
        ref_gt_ignore_flags = []
        for ref_instance in results.get('ref_instances', []):
            ref_gt_bboxes.append(ref_instance['bbox'])
            ref_gt_bboxes_labels.append(ref_instance['bbox_label'])
            ref_gt_ignore_flags.append(ref_instance['ignore_flag'])
        if self.box_type is None:
            results['ref_gt_bboxes'] = np.array(ref_gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['ref_gt_bboxes'] = box_type_cls(ref_gt_bboxes, dtype=torch.float32)
        results['ref_gt_ignore_flags'] = np.array(ref_gt_ignore_flags, dtype=bool)
        results['ref_gt_bboxes_labels'] = np.array(ref_gt_bboxes_labels, dtype=np.int64)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_det_instances:
            self._load_ref_instances(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
    


@TRANSFORMS.register_module()
class ReirResize(MMCV_Resize):
    """Resize images & bbox & seg.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                results['gt_masks'] = results['gt_masks'].rescale(
                    results['scale'])
            else:
                results['gt_masks'] = results['gt_masks'].resize(
                    results['img_shape'])

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['gt_bboxes'].clip_(results['img_shape'])
    
    def _resize_ref_instances(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('ref_gt_bboxes', None) is not None:
            results['ref_gt_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['ref_gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_ref_instances(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str



@TRANSFORMS.register_module()
class PackReirInputs(BaseTransform):
    """Pack the inputs data for the REIR

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        ref_instance_data = InstanceData()
        ref_instance_data['bboxes'] = results['ref_gt_bboxes']
        ref_instance_data['labels'] = to_tensor(results['ref_gt_bboxes_labels'])

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_bboxes':
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]

            if key == 'gt_bboxes_labels':
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])
                     
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        img_meta['ref_gt_instances'] = ref_instance_data
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
