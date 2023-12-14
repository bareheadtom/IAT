
import torch
import unittest
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.models.utils.transformer import Transformer
from mmdet.models.utils.transformer import DetrTransformerEncoder
from mmdet.models.utils.transformer import DetrTransformerDecoder
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE, MY_LAYER)
import torch.nn as nn

@MY_LAYER.register_module()
class PrintModule(nn.Module):
    def __init__(self, echostr, num) -> None:
        super().__init__()
        self.echostr = echostr
        self.num = num

    def echo(self, appstr):
        print([self.echostr]*self.num,appstr)


class TestRegitsMylayer(unittest.TestCase):
    def test_PrintModule(self):
        cfg = dict(
            type = 'PrintModule',
            echostr= 'hello',
            num = 5
        )
        printModule = build_from_cfg(cfg, MY_LAYER)
        printModule.echo("end")

if __name__ == '__main__':
    unittest.main()