# import torch
# from claymodel.finetune.segment.kelp_model import KelpSegmentor

# model = KelpSegmentor(ckpt_path='checkpoints/clay-v1.5.ckpt')
# print('Model patch embedding wave_dim:', model.model.encoder.patch_embedding.wave_dim)


import torch
from claymodel.finetune.segment.factory import Segmentor
model = Segmentor(num_classes=1, ckpt_path='checkpoints/clay-v1.5.ckpt')
print('Encoder patch_embedding wave_dim:', model.encoder.patch_embedding.wave_dim)
if hasattr(model.encoder.patch_embedding, 'weight_generator'):
    print('Weight generator wave_dim:', model.encoder.patch_embedding.weight_generator.weight_tokens.shape)