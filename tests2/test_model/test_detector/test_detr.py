import torch
import unittest
from mmdet.models.detectors.detr import DETR

# img_metas: [{'filename': '/root/autodl-tmp/Exdark/JPEGImages/IMGS/2015_06500.jpg', 'ori_filename': '2015_06500.jpg', 'ori_shape': (300, 452, 3), 'img_shape': (672, 970, 3), 'pad_shape': (672, 970, 3), 'scale_factor': array([1.7137809, 1.7142857, 1.7137809, 1.7142857], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([255., 255., 255.], dtype=float32), 'to_rgb': True}, 'batch_input_shape': (672, 970)}]
class TestDETRMethods(unittest.TestCase):
    def setUp(self):
        # 初始化测试所需的参数等
        self.backbone = None  # 设置backbone
        self.bbox_head = None  # 设置bbox_head

    def test_forward_dummy(self):
        # 测试 forward_dummy 函数是否正常运行
        detr = DETR(self.backbone, self.bbox_head)
        img = torch.randn(1, 3, 224, 224)  # 设置一个示例的输入图像
        outputs = detr.forward_dummy(img)
        # 在这里添加对输出的断言，确保函数的输出符合预期

    def test_onnx_export(self):
        # 测试 onnx_export 函数是否正常运行
        detr = DETR(self.backbone, self.bbox_head)
        img = torch.randn(1, 3, 224, 224)  # 设置一个示例的输入图像
        img_metas = [{'img_shape': (224, 224, 3)}]  # 设置一个示例的图像元数据
        det_bboxes, det_labels = detr.onnx_export(img, img_metas)
        # 在这里添加对输出的断言，确保函数的输出符合预期

if __name__ == '__main__':
    unittest.main()
