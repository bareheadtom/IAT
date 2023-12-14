import torch
import unittest
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.models.utils.transformer import Transformer
from mmdet.models.utils.transformer import DetrTransformerEncoder
from mmdet.models.utils.transformer import DetrTransformerDecoder
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmdet.core.bbox.assigners.hungarian_assigner import HungarianAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS

import torch.nn as nn
class TestAssigner(unittest.TestCase):
    def test_assign(self):
        cfg = dict(
            type = 'HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1),
            reg_cost=dict(type='BBoxL1Cost',weight=5.0,box_format='xywh'),
            iou_cost=dict(type='IoUCost',iou_mode='giou',weight=2.0)
        )
        assigner = build_from_cfg(cfg, BBOX_ASSIGNERS)
        