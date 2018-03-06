import torch
from torch.autograd import Variable
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from model.I3D_Pytorch import I3D
import os

_CHECKPOINT_PATHS = {
    'rgb': 'data/pytorch_checkpoints/rgb_scratch.pkl',
    'flow': 'data/pytorch_checkpoints/flow_scratch.pkl',
    'rgb_imagenet': 'data/pytorch_checkpoints/rgb_imagenet.pkl',
    'flow_imagenet': 'data/pytorch_checkpoints/flow_imagenet.pkl',
}
from dataloader import videoDataset, transform
import torch
import torch.utils.data as data

dataset = videoDataset(root="/home/xuchengming/BingZhang/figure_skating/frames",
                   label="./remain.txt", transform=transform)
videoLoader = torch.utils.data.DataLoader(dataset,
                                   batch_size=1, shuffle=False, num_workers=2)

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for idx, name in enumerate(own_state.keys()):
        param = state_dict[state_dict.keys()[idx]]
        own_state[name].copy_(param)
rgb_i3d = I3D(input_channel=3)
state_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
load_my_state_dict(rgb_i3d, state_dict)
rgb_i3d.eval()
for i, (videos, ids) in enumerate(videoLoader):
    print(ids)
    if os.path.isfile("/home/xuchengming/BingZhang/features/" + ids[0] + '.binary'):
        continue
    features = []
    if torch.cuda.is_available():
        videos = Variable(videos).cuda(0)
    for j in range(videos.shape[2] // 40):
        part_videos = videos[:, :, j*40:min((j+1)*40, videos.shape[2]), :, :]
        part_features = rgb_i3d(part_videos).data.cpu().numpy()
        features.append(part_features)
    fea = np.mean(np.concatenate(features, axis=2), axis=2)
    fea.tofile("/home/xuchengming/BingZhang/features/" + ids[0] + '.binary')
