import torch
import unittest
import sys
#sys.path.append('/root/autodl-tmp/projects/Illumination-Adaptive-Transformer/IAT_high/IAT_mmdetection/')
from mmdet.models.detectors.detr import DETR
from mmdet.datasets.exdark import ExdarkDataset
from mmdet.datasets.pipelines.loading import (LoadImageFromFile, LoadAnnotations)
from mmdet.datasets.pipelines import Compose
from tests.test_data.test_pipelines.test_transform.test_translate import _construct_ann_info
import numpy as np
from mmdet.core import eval_map, eval_recalls

# img_metas: [{'filename': '/root/autodl-tmp/Exdark/JPEGImages/IMGS/2015_06500.jpg', 'ori_filename': '2015_06500.jpg', 'ori_shape': (300, 452, 3), 'img_shape': (672, 970, 3), 'pad_shape': (672, 970, 3), 'scale_factor': array([1.7137809, 1.7142857, 1.7137809, 1.7142857], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([255., 255., 255.], dtype=float32), 'to_rgb': True}, 'batch_input_shape': (672, 970)}]
def list_from_file(filename, prefix='', offset=0, max_num=0, encoding='utf-8'):
    cnt = 0
    item_list = []
    with open(filename, 'r', encoding=encoding) as file:
        for _ in range(offset):
            file.readline()
        for line in file:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list
def bbox2result(bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor | np.ndarray): shape (n, 5)
            labels (torch.Tensor | np.ndarray): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]

class TestExdarkDataset(unittest.TestCase):
    def setUp(self):
        self.pipeline = [{
                        'type': 'LoadImageFromFile'
                    }, {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }, {
                        'type': 'RandomFlip',
                        'flip_ratio': 0.5
                    }, {
                        'type': 'AutoAugment',
                        'policies': [
                            [{
                                'type': 'Resize',
                                'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            }],
                            [{
                                'type': 'Resize',
                                'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
                                'multiscale_mode': 'value',
                                'keep_ratio': True
                            }, {
                                'type': 'RandomCrop',
                                'crop_type': 'absolute_range',
                                'crop_size': (384, 600),
                                'allow_negative_crop': True
                            }, {
                                'type': 'Resize',
                                'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
                                'multiscale_mode': 'value',
                                'override': True,
                                'keep_ratio': True
                            }]
                        ]
                    }, {
                        'type': 'Normalize',
                        'mean': [0, 0, 0],
                        'std': [255.0, 255.0, 255.0],
                        'to_rgb': True
                    }, {
                        'type': 'Pad',
                        'size_divisor': 1
                    }, {
                        'type': 'DefaultFormatBundle'
                    }, {
                        'type': 'Collect',
                        'keys': ['img', 'gt_bboxes', 'gt_labels']
                    }]
        self.results = {
                            'img_info': {
                                'id': '2015_02346.jpg',
                                'filename': '2015_02346.jpg',
                                'width': 900,
                                'height': 675
                            },
                            'ann_info': {
                                'bboxes': np.array([
                                    [20., 427., 366., 637.],
                                    [638., 374., 669., 398.],
                                    [645., 365., 664., 387.],
                                    [578., 403., 603., 444.],
                                    [588., 429., 608., 487.],
                                    [491., 337., 554., 372.],
                                    [589., 412., 667., 459.],
                                    [776., 594., 869., 670.],
                                    [825., 535., 897., 620.],
                                    [851., 319., 892., 360.]
                                ], dtype = np.float32),
                                'labels': np.array([3, 9, 10, 10, 10, 4, 4, 4, 4, 4]),
                                'bboxes_ignore': np.array([[20., 427., 366., 637.]], dtype = np.float32),
                                'labels_ignore': np.array([], dtype = np.int64)
                            },
                            'img_prefix': '/root/autodl-tmp/Exdark/JPEGImages/IMGS',
                            'seg_prefix': None,
                            'proposal_file': None,
                            'bbox_fields': [],
                            'mask_fields': [],
                            'seg_fields': []
                        }
        self.exdarkset = ExdarkDataset(ann_file='/root/autodl-tmp/Exdark/main/train.txt',
                                pipeline = self.pipeline,
                                img_prefix='/root/autodl-tmp/Exdark/JPEGImages/IMGS'
                                )
        #print("self.exdarkset.data_infos",self.exdarkset.data_infos)
    
    # def test_evaluation(self):
    #     # batch head
    #     resultshead = [(torch.randn(3,5),torch.randint(size=3,low=0,high=12)),(torch.randn(1,5),torch.randint(1,low=0,high=12)),(torch.randn(4,5),torch.randint(4,low=0,high=12))]
    #     # detector
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, 12)
    #         for det_bboxes, det_labels in results
    #     ]
    #     results = [bbox_results,bbox_results]
    #     print(self.exdarkset.evaluate(results))
    

#     image0,result [(0, 5), (3, 5), (1, 5), (1, 5), (0, 5), (0, 5), (0, 5), (4, 5), (3, 5), (57, 5), (5, 5), (26, 5)]
# image0,annotation {'bboxes': array([[ 84., 186., 243., 322.],
#        [494., 239., 638., 316.],
#        [373., 201., 419., 229.],
#        [404., 210., 451., 237.],
#        [433., 220., 491., 245.],
#        [454., 236., 524., 267.],
#        [549., 209., 618., 246.]], dtype=float32), 'labels': array([1, 1, 1, 1, 1, 1, 4]), 'bboxes_ignore': array([], shape=(0, 4), dtype=float32), 'labels_ignore': array([], dtype=int64)}
    def test_eval_map(self):
        # batch head
        
        resultshead = [(torch.randn(3,5),torch.tensor([1,4,5])),(torch.randn(1,5),torch.tensor([5])),(torch.randn(4,5),torch.tensor([1,4,7,5]))]
        # detector
        bbox_results = [
            bbox2result(det_bboxes, det_labels, 12)
            for det_bboxes, det_labels in resultshead
        ]
        #print("bbox_results",bbox_results)
        results = bbox_results
        annotations = [{
            'bboxes': np.array([[ 84., 186., 243., 322.],
                            [494., 239., 638., 316.],
                            [373., 201., 419., 229.],
                            [404., 210., 451., 237.],
                            [433., 220., 491., 245.],
                            [454., 236., 524., 267.],
                            [549., 209., 618., 246.]], dtype=np.float32), 
            'labels': np.array([1, 1, 1, 1, 1, 1, 4]), 
            # 'bboxes_ignore': np.array([], shape=(0, 4), dtype=np.float32), 
            # 'labels_ignore': np.array([], dtype=np.int64)
        },{
            'bboxes': np.array([[ 84., 186., 243., 322.],
                            [494., 239., 638., 316.],
                            [373., 201., 419., 229.],
                            [404., 210., 451., 237.],
                            [433., 220., 491., 245.],
                            [454., 236., 524., 267.],
                            [549., 209., 618., 246.]], dtype=np.float32), 
            'labels': np.array([1, 1, 1, 1, 1, 1, 4]), 
            # 'bboxes_ignore': np.array([], shape=(0, 4), dtype=np.float32), 
            # 'labels_ignore': np.array([], dtype=np.int64)
        },{
            'bboxes': np.array([[ 84., 186., 243., 322.],
                            [494., 239., 638., 316.],
                            [373., 201., 419., 229.],
                            [404., 210., 451., 237.],
                            [433., 220., 491., 245.],
                            [454., 236., 524., 267.],
                            [549., 209., 618., 246.]], dtype=np.float32), 
            'labels': np.array([1, 1, 1, 1, 1, 1, 4]), 
            # 'bboxes_ignore': np.array([], shape=(0, 4), dtype=np.float32), 
            # 'labels_ignore': np.array([], dtype=np.int64)
        }
        ]
        mean_ap, _ = eval_map(results,annotations)
    def t1est_exDarkset(self):
        print("self.exdarkset.data_infos",len(self.exdarkset.data_infos))
    def t1est_load_annotations(self):
        print("\n*************test_load_annotations")
        self.exdarkset.load_annotations('/root/autodl-tmp/Exdark/main/train.txt')
        #print(self.exdarkset.load_annotations('/root/autodl-tmp/Exdark/main/train.txt'))
        #{'id': '2015_04023.jpg', 'filename': '2015_04023.jpg', 'width': 470, 'height': 640}, {'id': '2015_04097.jpg', 'filename': '2015_04097.jpg', 'width': 500, 'height': 375}, {'id': '2015_04014.jpg', 'filename': '2015_04014.jpg', 'width': 338, 'height': 507}, {'id': '2015_04094.jpg', 'filename': '2015_04094.jpg', 'width': 640, 'height': 480}, {'id': '2015_03930.jpg', 'filename': '2015_03930.jpg', 'width': 640, 'height': 454}]
    def t1est_list_from_file(self):
        print("\n*************test_list_from_file")
        print(list_from_file('/root/autodl-tmp/Exdark/main/train.txt'))
    
    def t1est_get_ann_info(self):
        print("\n*************test_get_ann_info")
        print(self.exdarkset.get_ann_info(1))
    #     {'bboxes': array([[383., 129., 410., 189.],
    #    [153., 196., 290., 298.],
    #    [163., 225., 328., 374.],
    #    [  1., 295.,  22., 349.],
    #    [  3., 194.,  79., 370.],
    #    [172., 110., 324., 370.]], dtype=float32), 'labels': array([ 2,  8,  6,  2, 10, 10]), 'bboxes_ignore': array([], shape=(0, 4), dtype=float32), 'labels_ignore': array([], dtype=int64)}
    def t1est_getitem(self):
        print("\n*************test_getitem")
        print("\nexdarkset[1]",self.exdarkset[1])
    #     exdarkset[1] {'img_info': {'id': '2015_01794.JPEG', 'filename': '2015_01794.JPEG', 'width': 500, 'height': 377}, 'ann_info': {'bboxes': array([[383., 129., 410., 189.],
    #    [153., 196., 290., 298.],
    #    [163., 225., 328., 374.],
    #    [  1., 295.,  22., 349.],
    #    [  3., 194.,  79., 370.],
    #    [172., 110., 324., 370.]], dtype=float32), 'labels': array([ 2,  8,  6,  2, 10, 10]), 'bboxes_ignore': array([], shape=(0, 4), dtype=float32), 'labels_ignore': array([], dtype=int64)}, 'img_prefix': '/root/autodl-tmp/Exdark/JPEGImages/IMGS', 'seg_prefix': None, 'proposal_file': None, 'bbox_fields': [], 'mask_fields': [], 'seg_fields': []}
    #after pipeline 
        # exdarkset[1] {'img_metas': DataContainer({'filename': '/root/autodl-tmp/Exdark/JPEGImages/IMGS/2015_01794.JPEG', 'ori_filename': '2015_01794.JPEG', 'ori_shape': (377, 500, 3), 'img_shape': (800, 835, 3), 'pad_shape': (800, 835, 3), 'scale_factor': array([1.6800805, 1.6806723, 1.6800805, 1.6806723], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([255., 255., 255.], dtype=float32), 'to_rgb': True}}), 'img': DataContainer(tensor([[[0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039],
        #  [0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039],
        #  [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #  ...,
        #  [0.6745, 0.6745, 0.6784,  ..., 0.2431, 0.2275, 0.2196],
        #  [0.6745, 0.6784, 0.6863,  ..., 0.2196, 0.2196, 0.2157],
        #  [0.6745, 0.6784, 0.6902,  ..., 0.2078, 0.2118, 0.2157]],

        # [[0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039],
        #  [0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039],
        #  [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #  ...,
        #  [0.6431, 0.6471, 0.6510,  ..., 0.1765, 0.1765, 0.1765],
        #  [0.6431, 0.6431, 0.6471,  ..., 0.1843, 0.1765, 0.1725],
        #  [0.6431, 0.6431, 0.6431,  ..., 0.1922, 0.1804, 0.1725]],

        # [[0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039],
        #  [0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039],
        #  [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #  ...,
        #  [0.6667, 0.6667, 0.6706,  ..., 0.1608, 0.1529, 0.1490],
        #  [0.6627, 0.6627, 0.6667,  ..., 0.1412, 0.1490, 0.1490],
        #  [0.6588, 0.6627, 0.6667,  ..., 0.1294, 0.1412, 0.1529]]])), 'gt_bboxes': DataContainer(tensor([[332.4510, 413.3562, 637.6577, 640.7151],
        # [354.7289, 477.9975, 722.3136, 800.0000],
        # [  0.0000, 634.0281,  40.6109, 754.3945],
        # [  0.0000, 408.8982, 167.5948, 800.0000],
        # [374.7789, 221.6615, 713.4025, 800.0000]])), 'gt_labels': DataContainer(tensor([ 8,  6,  2, 10, 10]))}


    # # # def test_Compose(self):
    #     pipeline = [{
    #                     'type': 'LoadImageFromFile'
    #                 }, {
    #                     'type': 'LoadAnnotations',
    #                     'with_bbox': True
    #                 }, {
    #                     'type': 'RandomFlip',
    #                     'flip_ratio': 0.5
    #                 }, {
    #                     'type': 'AutoAugment',
    #                     'policies': [
    #                         [{
    #                             'type': 'Resize',
    #                             'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    #                             'multiscale_mode': 'value',
    #                             'keep_ratio': True
    #                         }],
    #                         [{
    #                             'type': 'Resize',
    #                             'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
    #                             'multiscale_mode': 'value',
    #                             'keep_ratio': True
    #                         }, {
    #                             'type': 'RandomCrop',
    #                             'crop_type': 'absolute_range',
    #                             'crop_size': (384, 600),
    #                             'allow_negative_crop': True
    #                         }, {
    #                             'type': 'Resize',
    #                             'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    #                             'multiscale_mode': 'value',
    #                             'override': True,
    #                             'keep_ratio': True
    #                         }]
    #                     ]
    #                 }, {
    #                     'type': 'Normalize',
    #                     'mean': [0, 0, 0],
    #                     'std': [255.0, 255.0, 255.0],
    #                     'to_rgb': True
    #                 }, {
    #                     'type': 'Pad',
    #                     'size_divisor': 1
    #                 }, {
    #                     'type': 'DefaultFormatBundle'
    #                 }, {
    #                     'type': 'Collect',
    #                     'keys': ['img', 'gt_bboxes', 'gt_labels']
    #                 }]
    #     pipeline = Compose(pipeline)
    #     results = pipeline(self.results)
        #print("\nalter pipeline:",results)
        # alter pipeline: {'img_metas': DataContainer({'filename': '/root/autodl-tmp/Exdark/JPEGImages/IMGS/2015_02346.jpg', 'ori_filename': '2015_02346.jpg', 'ori_shape': (675, 900, 3), 'img_shape': (768, 942, 3), 'pad_shape': (768, 942, 3), 'scale_factor': array([1.7034358, 1.7028825, 1.7034358, 1.7028825], dtype=float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([255., 255., 255.], dtype=float32), 'to_rgb': True}}), 'img': DataContainer(tensor([[[0.0235, 0.0235, 0.0235,  ..., 0.0275, 0.0275, 0.0275],
        #  [0.0235, 0.0235, 0.0235,  ..., 0.0275, 0.0275, 0.0275],
        #  [0.0235, 0.0235, 0.0235,  ..., 0.0275, 0.0235, 0.0235],
        #  ...,
        #  [0.3373, 0.3373, 0.3412,  ..., 0.2000, 0.1882, 0.1843],
        #  [0.3373, 0.3373, 0.3412,  ..., 0.1961, 0.1882, 0.1804],
        #  [0.3373, 0.3373, 0.3412,  ..., 0.1961, 0.1843, 0.1804]],

        # [[0.0235, 0.0235, 0.0235,  ..., 0.0275, 0.0275, 0.0275],
        #  [0.0235, 0.0235, 0.0235,  ..., 0.0275, 0.0275, 0.0275],
        #  [0.0235, 0.0235, 0.0235,  ..., 0.0275, 0.0235, 0.0235],
        #  ...,
        #  [0.1647, 0.1647, 0.1686,  ..., 0.2118, 0.2000, 0.1961],
        #  [0.1647, 0.1647, 0.1686,  ..., 0.2039, 0.1922, 0.1843],
        #  [0.1647, 0.1647, 0.1686,  ..., 0.2000, 0.1882, 0.1804]],

        # [[0.0157, 0.0157, 0.0157,  ..., 0.0275, 0.0275, 0.0275],
        #  [0.0157, 0.0157, 0.0157,  ..., 0.0275, 0.0275, 0.0275],
        #  [0.0157, 0.0157, 0.0157,  ..., 0.0275, 0.0235, 0.0235],
        #  ...,
        #  [0.0196, 0.0196, 0.0235,  ..., 0.1294, 0.1216, 0.1216],
        #  [0.0196, 0.0196, 0.0235,  ..., 0.1176, 0.1098, 0.1059],
        #  [0.0196, 0.0196, 0.0235,  ..., 0.1098, 0.1020, 0.0980]]])), 'gt_bboxes': DataContainer(tensor([[  0.0000, 485.8261, 269.5630, 750.7189],
        # [612.9454, 418.9721, 652.0809, 449.2456],
        # [621.7825, 407.6196, 645.7687, 435.3703],
        # [537.1993, 455.5526, 568.7601, 507.2698],
        # [549.8236, 488.3488, 575.0723, 561.5097],
        # [427.3674, 372.3006, 506.9008, 416.4494],
        # [551.0861, 466.9052, 649.5560, 526.1907],
        # [787.1615, 696.4789, 904.5679, 768.0000],
        # [849.0208, 622.0567, 939.9161, 729.2752],
        # [881.8441, 349.5955, 933.6039, 401.3126]])), 'gt_labels': DataContainer(tensor([ 3,  9, 10, 10, 10,  4,  4,  4,  4,  4]))}


    # def test_LoadImageFromFile(self):
    #     print("\n********************test_LoadImageFromFile")
    #     results = self.results
    #     print("results",results)
    #     transform = LoadImageFromFile()
    #     results = transform(results)
    #     print("results",results)
    #     # results {'img_prefix': '/root/autodl-tmp/Exdark/JPEGImages/IMGS', 'img_info': {'filename': '2015_00001.png'}, 'filename': '/root/autodl-tmp/Exdark/JPEGImages/IMGS/2015_00001.png', 'ori_filename': '2015_00001.png', 'img': array([[[ 61,  73,  83],
    #     # [ 93, 112, 115],
    #     # [ 82, 113, 106],
    #     # [  7,   7,  21]]], dtype=uint8), 'img_shape': (375, 500, 3), 'ori_shape': (375, 500, 3), 'img_fields': ['img']}
    # def test_LoadAnnotations(self):
    #     print("\n********************test_LoadAnnotations")
    #     img_info = dict(height=375, width = 500)
    #     ann_info = _construct_ann_info(h = 375, w = 500)
    #     results = dict(img_info = img_info, 
    #                    ann_info = ann_info,
    #                    bbox_fields = [])
    #     #print("results",results)
    #     transform = LoadAnnotations(with_bbox=True)
    #     results = transform(results)
    #     print("results",results)


if __name__ == '__main__':
    unittest.main()
# 		'train': {
    # 			'type': 'ExdarkDataset',
    # 			'ann_file': '/root/autodl-tmp/Exdark/main/train.txt',
    # 			'img_prefix': '/root/autodl-tmp/Exdark/JPEGImages/IMGS',
    # 			'pipeline': [{
    # 				'type': 'LoadImageFromFile'
    # 			}, {
    # 				'type': 'LoadAnnotations',
    # 				'with_bbox': True
    # 			}, {
    # 				'type': 'RandomFlip',
    # 				'flip_ratio': 0.5
    # 			}, {
    # 				'type': 'AutoAugment',
    # 				'policies': [
    # 					[{
    # 						'type': 'Resize',
    # 						'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    # 						'multiscale_mode': 'value',
    # 						'keep_ratio': True
    # 					}],
    # 					[{
    # 						'type': 'Resize',
    # 						'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
    # 						'multiscale_mode': 'value',
    # 						'keep_ratio': True
    # 					}, {
    # 						'type': 'RandomCrop',
    # 						'crop_type': 'absolute_range',
    # 						'crop_size': (384, 600),
    # 						'allow_negative_crop': True
    # 					}, {
    # 						'type': 'Resize',
    # 						'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    # 						'multiscale_mode': 'value',
    # 						'override': True,
    # 						'keep_ratio': True
    # 					}]
    # 				]
    # 			}, {
    # 				'type': 'Normalize',
    # 				'mean': [0, 0, 0],
    # 				'std': [255.0, 255.0, 255.0],
    # 				'to_rgb': True
    # 			}, {
    # 				'type': 'Pad',
    # 				'size_divisor': 1
    # 			}, {
    # 				'type': 'DefaultFormatBundle'
    # 			}, {
    # 				'type': 'Collect',
    # 				'keys': ['img', 'gt_bboxes', 'gt_labels']
    # 			}]
    # 		}