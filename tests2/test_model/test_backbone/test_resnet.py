import torch
import unittest
from mmdet.models.backbones.resnet import ResNet

class TestResNetMethods(unittest.TestCase):
    def test_resnet50(self):
        backbone = ResNet(depth=50, 
                          num_stages= 4, 
                          out_indices=(3,),
                          norm_cfg=dict(type='BN', requires_grad=False),
                          init_cfg=dict(type='Pretrained',checkpoint='torchvision://resnet50')
                          )
        backbone.eval()
        inputs = torch.randn(1, 3, 224, 224)
        out = backbone(inputs)
        print(len(out))
        for layer in out:
            print("lay:",layer.shape)
        #lay: torch.Size([1, 2048, 7, 7])

    # 添加更多测试用例...

if __name__ == '__main__':
    unittest.main()
