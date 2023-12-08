import torch
import unittest
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.models.utils.transformer import Transformer
from mmdet.models.utils.transformer import DetrTransformerEncoder

class TestResNetMethods(unittest.TestCase):
    def test_DetrTransformerEncoder(self):
        print("\n**************test_DetrTransformerEncoder")
        detrTransformerEncoder = DetrTransformerEncoder(num_layers=6,
                                                        transformerlayers=dict(
                                                            type='BaseTransformerLayer',
                                                            attn_cfgs=[
                                                                dict(
                                                                    type='MultiheadAttention',
                                                                    embed_dims=256,
                                                                    num_heads=8,
                                                                    dropout=0.1
                                                                )
                                                            ],
                                                            feedforward_channels=2048,
                                                            ffn_dropout=0.1,
                                                            operation_order=('self_attn','norm','ffn','norm')
                                                        ))
        x = torch.randn(1, 256, 34, 34)
        mask = torch.zeros(1, 34, 34).bool()
        pos_embed = torch.randn(1, 256, 34, 34)
        bs,c,h,w = x.shape
        x = x.view(bs, c ,-1).permute(2,0,1)
        pos_embed = pos_embed.view(bs,c,-1).permute(2,0,1)
        mask = mask.view(bs, -1)
        memmory = detrTransformerEncoder(query = x, 
                                         key=None, 
                                         value = None,
                                         query_pos = pos_embed,
                                         query_key_padding_mask = mask
                                         )
        print("memmory", memmory.shape)
        
    def test_transformer(self):
        print("\n**************test_transformer")
        ttransformer = Transformer(encoder=dict(
                                     type='DetrTransformerEncoder',
                                     num_layers=6,
                                     transformerlayers=dict(
                                         type='BaseTransformerLayer',
                                         attn_cfgs=[
                                             dict(
                                                 type='MultiheadAttention',
                                                 embed_dims=256,
                                                 num_heads=8,
                                                 dropout=0.1
                                             )
                                         ],
                                         feedforward_channels=2048,
                                         ffn_dropout=0.1,
                                         operation_order=('self_attn','norm','ffn','norm')
                                     )
                                 ),
                                 decoder=dict(
                                     type='DetrTransformerDecoder',
                                     return_intermediate=True,
                                     num_layers=6,
                                     transformerlayers=dict(
                                         type='DetrTransformerDecoderLayer',
                                         attn_cfgs=dict(
                                             type='MultiheadAttention',
                                             embed_dims=256,
                                             num_heads=8,
                                             dropout=0.1
                                         ),
                                         feedforward_channels=2048,
                                         ffn_dropout=0.1,
                                         operation_order=('self_attn','norm','cross_attn','norm','ffn','norm')
                                     ),

                                 ))
        x = torch.randn(1, 256, 34, 34)
        mask = torch.zeros(1, 34, 34).bool()
        query_embed = torch.randn(100, 256)
        pos_embed = torch.randn(1, 256, 34, 34)
        out_dec, memmory = ttransformer(x, mask, query_embed, pos_embed)
        print("out_dec",out_dec.shape)
        print("memmory",memmory.shape)

    def test_detrhead(self):
        print("\n**************test_detrhead")
        # operation_order = ('self_attn','norm','ffn','nom')
        # print("assert",set(operation_order) & set(
        #     ['self_attn', 'norm', 'ffn', 'cross_attn']),
        # set(operation_order) & set(
        #     ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
        #     set(operation_order)
        # )
        # [{'filename': '/root/autodl-tmp/Exdark/JPEGImages/IMGS/2015_06500.jpg', 
        #   'ori_filename': '2015_06500.jpg', 'ori_shape': (300, 452, 3), 'img_shape': (672, 970, 3), 
        #   'pad_shape': (672, 970, 3), 'scale_factor': array([1.7137809, 1.7142857, 1.7137809, 1.7142857], dtype=float32), 
        #   'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 
        #                                                           'std': array([255., 255., 255.], dtype=float32), 
        #                                                           'to_rgb': True}, 'batch_input_shape': (672, 970)}]
        
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
            'pad_shape': (s, s, 3),
            'batch_input_shape': (s, s)
        }]
        detr_head = DETRHead(num_classes=12, 
                             in_channels=2048,
                             transformer=dict(
                                 type='Transformer',
                                 encoder=dict(
                                     type='DetrTransformerEncoder',
                                     num_layers=6,
                                     transformerlayers=dict(
                                         type='BaseTransformerLayer',
                                         attn_cfgs=[
                                             dict(
                                                 type='MultiheadAttention',
                                                 embed_dims=256,
                                                 num_heads=8,
                                                 dropout=0.1
                                             )
                                         ],
                                         feedforward_channels=2048,
                                         ffn_dropout=0.1,
                                         operation_order=('self_attn','norm','ffn','norm')
                                     )
                                 ),
                                 decoder=dict(
                                     type='DetrTransformerDecoder',
                                     return_intermediate=True,
                                     num_layers=6,
                                     transformerlayers=dict(
                                         type='DetrTransformerDecoderLayer',
                                         attn_cfgs=dict(
                                             type='MultiheadAttention',
                                             embed_dims=256,
                                             num_heads=8,
                                             dropout=0.1
                                         ),
                                         feedforward_channels=2048,
                                         ffn_dropout=0.1,
                                         operation_order=('self_attn','norm','cross_attn','norm','ffn','norm')
                                     ),

                                 )
                            ),
                            positional_encoding=dict(
                                type='SinePositionalEncoding', num_feats=128, normalize=True
                            ),
                            loss_cls=dict(type='CrossEntropyLoss',
                                          bg_cls_weight=0.1,
                                          use_sigmoid=False,
                                          loss_weight=1.0,
                                          class_weight=1.0
                                          ),
                            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                            loss_iou=dict(type='GIoULoss', loss_weight=2.0)
        )
        detr_head.init_weights()
        feat = [torch.rand(1, 2048, 7, 7)]
        cls_scores, bbox_preds = detr_head.forward(feat, img_metas)
        print(len(cls_scores), len(bbox_preds))
        print("cls_score",cls_scores[0].shape, "bbox_pred",bbox_preds[0].shape)
        #cls_score torch.Size([6, 1, 100, 13]) bbox_pred torch.Size([6, 1, 100, 4])


    # 添加更多测试用例...

if __name__ == '__main__':
    unittest.main()
# bbox_head=dict(
#         type='DETRHead',
#         num_classes=12,
#         in_channels=2048,
#         transformer=dict(
#             type='Transformer',
#             encoder=dict(
#                 type='DetrTransformerEncoder',
#                 num_layers=6,
#                 transformerlayers=dict(
#                     type='BaseTransformerLayer',
#                     attn_cfgs=[
#                         dict(
#                             type='MultiheadAttention',
#                             embed_dims=256,
#                             num_heads=8,
#                             dropout=0.1)
#                     ],
#                     feedforward_channels=2048,
#                     ffn_dropout=0.1,
#                     operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
#             decoder=dict(
#                 type='DetrTransformerDecoder',
#                 return_intermediate=True,
#                 num_layers=6,
#                 transformerlayers=dict(
#                     type='DetrTransformerDecoderLayer',
#                     attn_cfgs=dict(
#                         type='MultiheadAttention',
#                         embed_dims=256,
#                         num_heads=8,
#                         dropout=0.1),
#                     feedforward_channels=2048,
#                     ffn_dropout=0.1,
#                     operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
#                                      'ffn', 'norm')),
#             )),
#         positional_encoding=dict(
#             type='SinePositionalEncoding', num_feats=128, normalize=True),
#         loss_cls=dict(
#             type='CrossEntropyLoss',
#             bg_cls_weight=0.1,
#             use_sigmoid=False,
#             loss_weight=1.0,
#             class_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=5.0),
#         loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
#     # training and testing settings
#     train_cfg=dict(
#         assigner=dict(
#             type='HungarianAssigner',
#             cls_cost=dict(type='ClassificationCost', weight=1.),
#             reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
#             iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
#     test_cfg=dict(max_per_img=100))