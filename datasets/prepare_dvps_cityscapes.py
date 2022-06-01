import json
from glob import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

root = '/path/to/cityscapes-dvps/video_sequence/'

for split in ['val','train']:
    imgs      = glob(root+split+'/*_leftImg8bit.png')
    seg_gts   = glob(root+split+'/*_gtFine_instanceTrainIds.png')
    depth_gts = glob(root+split+'/*_depth.png')
    assert len(imgs) == len(seg_gts) == len(depth_gts) > 0, (len(imgs), len(seg_gts), len(depth_gts))
    print('{} samples found'.format(len(imgs)))

    imgs      = [os.path.basename(_) for _ in sorted(imgs)]
    seg_gts   = [os.path.basename(_) for _ in sorted(seg_gts)]
    depth_gts = [os.path.basename(_) for _ in sorted(depth_gts)]

    prefix1 = imgs[0][:-len('_leftImg8bit.png')]
    prefix2 = seg_gts[0][:-len('_gtFine_instanceTrainIds.png')]
    prefix3 = depth_gts[0][:-len('_depth.png')]
    assert prefix1 == prefix2 == prefix3, (prefix1, prefix2, prefix3)

    data_json = []
    for path_image, path_depth, path_seg in tqdm(zip(imgs, depth_gts, seg_gts)):
        w, h = Image.open(os.path.join(root,split,path_image)).size
        dj= {
            "height": h,
            "width": w,
            "image": path_image,
            "depth": path_depth,
            "seg": path_seg,
            }
        data_json.append(dj)

    save_dir = os.path.join(root, 'dvps_cityscapes_{}.json'.format(split))
    print('saving to '+save_dir)
    with open(save_dir, 'w') as f:
        json.dump(data_json, f, indent=4)
