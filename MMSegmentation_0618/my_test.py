import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2

# 载入 config 配置文件
from mmengine import Config
cfg = Config.fromfile('/root/autodl-tmp/mmsegmentation/work_dirs/pspnet_r50-d32_4xb2-80k_cityscapes-512x1024/pspnet_r50-d32_4xb2-80k_cityscapes-512x1024.py')

from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner

register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)

checkpoint_path = '/root/autodl-tmp/mmsegmentation/work_dirs/pspnet_r50-d32_4xb2-80k_cityscapes-512x1024/iter_80000.pth'
model = init_model(cfg, checkpoint_path, 'cuda:0')

img = mmcv.imread('/root/autodl-tmp/mmsegmentation/data/origin.JPG')


result = inference_model(model, img)
result.keys()

pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

np.unique(pred_mask)

# plt.imshow(pred_mask)
# plt.savefig('/root/autodl-tmp/mmsegmentation/data/pred.jpg',dpi=300)
# plt.show()



# 可视化预测结果
visualization = show_result_pyplot(model, img, result, opacity=0.7, out_file='/root/autodl-tmp/mmsegmentation/data/pred.jpg')
plt.imshow(mmcv.bgr2rgb(visualization))
plt.savefig('/root/autodl-tmp/mmsegmentation/data/pred_all.jpg',dpi=300)
plt.show()
