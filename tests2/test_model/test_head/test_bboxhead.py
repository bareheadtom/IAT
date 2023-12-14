import torch
import unittest
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.models.utils.transformer import Transformer
from mmdet.models.utils.transformer import DetrTransformerEncoder
from mmdet.models.utils.transformer import DetrTransformerDecoder
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.builder import TRANSFORMER
import torch.nn as nn
class TestResNetMethods(unittest.TestCase):

    def test_nnMultiheadAttention(self):
        print("\n**************test_nnMultiheadAttention")
        x = torch.randn(1,128, 38, 38)
        bs,c,h,w = x.shape
        x = x.view(bs,c,-1).permute(2,0,1)
        #(38*38 1 128)
        mask = torch.zeros(1, 38,38)
        mask = mask.view(bs,-1)
        mem = torch.randn_like(x)
        pos_embed = torch.randn_like(x)
        query_embed = torch.randn(100, 128)
        query_embed = query_embed.unsqueeze(1).repeat(1,bs,1)
        tgt = torch.zeros_like(query_embed)
        #print("tgt",tgt.shape)
        attmask = torch.zeros(100, h*w)
        multiattn = nn.MultiheadAttention(128, 8)
        out,outweight = multiattn(query=tgt+query_embed, 
                        key=mem+pos_embed, 
                        value=mem,
                        attn_mask = attmask,
                        key_padding_mask= mask)
        print("out", out.shape)

    def test_MultiheadAttention(self):
        print("\n**************test_MultiheadAttention")
        multiheadcfg = dict(
            type='MultiheadAttention',
            embed_dims = 128,
            num_heads = 8
        )
        multiheadmodel = build_from_cfg(multiheadcfg, ATTENTION)
        x = torch.randn(1,128, 38, 38)
        bs,c,h,w = x.shape
        x = x.view(bs,c,-1).permute(2,0,1)
        mask = torch.zeros(1, 38,38)
        mask = mask.view(bs,-1)
        mem = torch.randn_like(x)
        pos_embed = torch.randn_like(x)
        query_embed = torch.randn(100, 128)
        query_embed = query_embed.unsqueeze(1).repeat(1,bs,1)
        tgt = torch.zeros_like(query_embed)
        #print("tgt",tgt.shape)
        attmask = torch.zeros(100, h*w)
        attendedout = multiheadmodel(query = tgt,
                                     key = mem, 
                                     value = mem,
                                     identity = tgt,
                                     query_pos = query_embed,
                                     key_pos = pos_embed,
                                     attn_mask = attmask,
                                     key_padding_mask = mask
                                     )
        print("attendedout",attendedout.shape)


    def test_BaseTransformerLayer(self):
        print("\n**************test_BaseTransformerLayer")
        laycfg = dict(
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
        baseTransformerLayer=build_from_cfg(laycfg, TRANSFORMER_LAYER)
        x = torch.randn(1, 256, 34, 34)
        mask = torch.zeros(1, 34, 34).bool()
        pos_embed = torch.randn(1, 256, 34, 34)
        bs,c,h,w = x.shape
        x = x.view(bs, c ,-1).permute(2,0,1)
        pos_embed = pos_embed.view(bs,c,-1).permute(2,0,1)
        mask = mask.view(bs, -1)
        nonmemmory = baseTransformerLayer(query=x,key=None,value=None,query_pos = pos_embed,query_key_padding_mask=mask)
        print("nonmemmory",nonmemmory.shape)
    
    def test_DetrTransformerDecoderLayer(self):
        print("\n**************test_DetrTransformerDecoderLayer")
        delayercfg=dict(
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
                    )
        x = torch.randn(2, 256, 88,88)
        mask = torch.randn(2, 88, 88)
        pos_embed = torch.randn(2,256,88,88)
        bs, c ,h,w = x.shape
        x = x.view(bs, c, -1).permute(2,0,1)
        pos_embed = pos_embed.view(bs,c,-1).permute(2,0,1)
        mask = mask.view(bs, -1)
        detrTransformerDecoderLayer = build_from_cfg(delayercfg, TRANSFORMER_LAYER)
        meme = torch.randn_like(x)
        query_embed = torch.randn(100, 256)
        query_embed = query_embed.unsqueeze(1).repeat(1,bs,1)
        target = torch.zeros_like(query_embed)
        decoderout = detrTransformerDecoderLayer(query = target, 
                                               key = meme,
                                               value = meme,
                                               query_pos = query_embed,
                                               key_pos = pos_embed,
                                               key_pdding_mask = mask
                                               )
        print("decoderout",decoderout.shape)

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
    def test_DetrTransformerDecoder(self):
        print("\n**************test_DetrTransformerDecoder")
        detrTransformerDecoder = DetrTransformerDecoder(return_intermediate=True,
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
                                                        )
                                                        )
        x = torch.randn(1, 256, 34, 34)
        mask = torch.zeros(1, 34, 34).bool()
        pos_embed = torch.randn(1, 256, 34, 34)
        query_embed = torch.randn(100, 256)
        bs,c,h,w = x.shape
        x = x.view(bs, c ,-1).permute(2,0,1)
        pos_embed = pos_embed.view(bs,c,-1).permute(2,0,1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)

        target = torch.zeros_like(query_embed)
        mem = torch.randn_like(x)
        deout = detrTransformerDecoder(query=target,
                                       key=mem,
                                       value=mem,
                                       key_pos= pos_embed,
                                       query_pos = query_embed,
                                       key_padding_maskk = mask
                                       )
        print(len(deout))
        for layer in deout:
            print(layer.shape)
        
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

    def test_transformerRaw(self):
        print("\n**************test_transformerRaw TransformerRaw")
        cfg = dict(type='TransformerRaw',
                   d_model=256,
                   nhead=8,
                   num_encoder_layers=6,
                   num_decoder_layers=6,
                   dim_feedforward=2048,
                   dropout=0.1
                   )
        ttransformer = build_from_cfg(cfg, TRANSFORMER)
        x = torch.randn(1, 256, 34, 34)
        mask = torch.zeros(1, 34, 34).bool()
        query_embed = torch.randn(100, 256)
        pos_embed = torch.randn(1, 256, 34, 34)
        out_dec, memmory = ttransformer(x, mask, query_embed, pos_embed)
        print("out_dec",out_dec.shape)
        print("memmory",memmory.shape)

    def test_deformableTransformerRaw(self):
        print("\n**************test_deformableTransformerRaw")
        cfg = dict(type='DeformableTransformerRaw',
                   d_model=256,
                   nhead=8,
                   num_encoder_layers=6,
                   num_decoder_layers=6,
                   dim_feedforward=2048,
                   dropout=0.1
                   )
        deformabletransformer = build_from_cfg(cfg, TRANSFORMER)
        x = torch.randn(4,1, 256, 34, 34)
        mask = torch.zeros(4,1, 34, 34).bool()
        query_embed = torch.randn(100, 256)
        pos_embed = torch.randn(4,1, 256, 34, 34)
        out_dec, memmory = deformabletransformer(srcs=x, masks=mask, query_embed = query_embed, pos_embeds=pos_embed)
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