import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

img="demo/origin.jpg"
control_img = mmcv.imread(img)
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)

# prompt = 'Room with blue walls and a yellow ceiling.'
#prompt = 'Room with Baroque style doors and windows.'
prompt = 'Room with Deconstructivist doors and windows.'
output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'Deconstructivist_{idx}.png')
# controls = output_dict['controls']
# for idx, control in enumerate(controls):
#     control.save(f'control_{idx}.png')