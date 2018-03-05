import os
import torch
import torch.utils.data as data
import numpy as np
from torchvision.transforms import Normalize, CenterCrop
import PIL.Image as Image

def default_loader(path, crop):
    files = os.listdir(path)
    files = sorted(files, key=lambda x:int(x.split('.')[0]))
    images = []
    for idx, filename in enumerate(files):
        img = Image.open(path + '/' + filename).convert("RGB")
	img = np.asarray(crop(img))
        images.append((img - 128) / 128.0)
        #if idx > 2000:
        #    break
    return np.array(images)
    # return Image.open(path).convert('RGB')

def transform(fea):
    return torch.Tensor(fea).permute(3, 0, 1, 2)

class videoDataset(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None,
                 loader=default_loader):
        fh = open(label)
        videos = []
        for line in fh.readlines():
            video_id = line.strip()
            videos.append(video_id)
        self.root = root
        self.videos = videos
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.crop = CenterCrop(224)

    def __getitem__(self, index):
        fn = self.videos[index]
        fea = self.loader(os.path.join(self.root, fn), self.crop)
        if self.transform is not None:
            fea = self.transform(fea)
        return fea, fn
        # return fea

    def __len__(self):
        return len(self.videos)

if __name__ == "__main__":
# torch.utils.data.DataLoader
    dataset = videoDataset(root="/home/xuchengming/figure_skating/frames",
                   label="./videos.txt", transform=transform)
    videoLoader = torch.utils.data.DataLoader(dataset,
                                      batch_size=1, shuffle=True, num_workers=2)

    for i, features in enumerate(videoLoader):
        import pdb;pdb.set_trace()
        print((i, len(labels)))
