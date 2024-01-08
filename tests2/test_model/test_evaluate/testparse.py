from mmcv.parallel import DataContainer 
import numpy as np
import torch

# data = {'aa':77,'bb':"adad",'img_metas': [DataContainer([[{'filename': '/root/autodl-tmp/Exdark/JPEGImages/IMGS/2015_00975.jpg', 'ori_filename': '2015_00975.jpg', 'ori_shape': (338, 507, 3), 'img_shape': (800, 1200, 3), 'pad_shape': (800, 1200, 3), 'scale_factor': np.array([2.366864, 2.366864, 2.366864, 2.366864], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([0., 0., 0.], dtype=np.float32), 'std': np.array([255., 255., 255.], dtype=np.float32), 'to_rgb': True}}]])]}
# #aa, bb,img_metas = (**data)
# #print(aa,bb,img_metas[0].data)

# def pp(aa, bb,img_metas):
#     print(aa,bb,img_metas[0].data)

# pp(**data)

# t = torch.randn(2,3,5,5)
# t1={'aa',t}
# print(t,t1)
np.ndarray
arr = np.array([1,2,3])
d = {'d':arr}
print(arr,arr.shape,d)