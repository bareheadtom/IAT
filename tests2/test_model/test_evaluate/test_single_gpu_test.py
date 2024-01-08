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
from mmcv import Config, DictAction
import torch.nn as nn
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.core import encode_mask_results

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        cfg = Config.fromfile('/root/autodl-fs/projects/Illumination-Adaptive-Transformer/IAT_high/IAT_mmdetection/configs/detr/detr_ours_LOL.py')
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        self.model.init_weights()
        val_dataset = build_dataset(cfg.data.val,dict(test_mode=True))
        self.dataloader = build_dataloader(
                            val_dataset,
                            samples_per_gpu=val_samples_per_gpu,
                            workers_per_gpu=cfg.data.workers_per_gpu,
                            dist=False,
                            shuffle=False)
        
    # def test_single_gpu_test(self):
    #     print("\n**********************test_single_gpu_test")
    #     from mmdet.apis import single_gpu_test
    #     #print("runner.model",self.model,"self.dataloader",self.dataloader)
    #     results = single_gpu_test(self.model, self.dataloader, show=False)
    #     print("results",results)
    
    def test_simple_test(self):
        dataset = self.dataloader.dataset
        results = []
        self.model.eval()
        for i, data in enumerate(self.dataloader):
            #print("data",data)
            imgs = data['img']
            img_metas = data['img_metas']
            #print("imgs",imgs,"img_metas",img_metas[0].data)
            img_metas = img_metas[0].data
            #bbox_results = self.model.forward_test(imgs, img_metas)
            #print("bbox_results",bbox_results)
            num_augs = len(imgs)
            for img, img_meta in zip(imgs, img_metas):
                batch_size = len(img_meta)
                #print("\nimg_meta",img_meta,"batch_size",batch_size)
                for img_id in range(batch_size):
                    img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
            with torch.no_grad():
                result = self.model.simple_test(imgs[0], img_metas[0])
            #print("bbox_results",result)
            # if isinstance(result[0], tuple):
            #     print("true")
            #     result = [(bbox_results, encode_mask_results(mask_results))
            #             for bbox_results, mask_results in result]
            results.extend(result)
        eval_res = self.dataloader.dataset.evaluate(
            results)
        print("eval_res",eval_res) 
            
    
    



if __name__ == '__main__':
    unittest.main()
