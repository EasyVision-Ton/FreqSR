import torch
import torch.nn as nn
import cv2
import os


if __name__ == '__main__':
    img_dir = '/data/SR_data/benchmark/Set14/LR_bicubic/X4/barbarax4.png'
    name = os.path.basename(img_dir)
    save_dir = os.path.join('../results', name)

    up = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
    img = cv2.imread(img_dir)
    print(img.shape)
    x = torch.from_numpy(img).unsqueeze(0).float().permute(0, 3, 1, 2)
    print(x.shape)
    out = up(x).permute(0, 2, 3, 1).squeeze(0).numpy()
    print(out.shape)
    cv2.imwrite(save_dir, out)
