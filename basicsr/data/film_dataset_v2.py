import random
import torch
from torch.utils import data as data
import os
import numpy as np
import cv2
import operator
from PIL import Image
from skimage.color import rgb2lab

import torchvision.transforms as transforms
from basicsr.utils.util import get_root_logger
from basicsr.utils.data_util import img2tensor, paired_random_crop, augment, paired_random_crop_240
from basicsr.data.Data_Degradation.util import degradation_video_list_4, degradation_video_colorization, transfer_1, \
    transfer_2, degradation_video_colorization_test
from basicsr.utils.LAB_util import to_mytensor, Normalize_LAB


# import util as util
# from basicsr.data.util import rgb2lab,rgb2xyz,lab2rgb,lab2xyz,xyz2lab,read_img


class Film_color_train_dataset(data.Dataset):  ##  for DAVIS colorization dataset

    def __init__(self, data_config):
        super(Film_color_train_dataset, self).__init__()

        self.data_config = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)

        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2

        if self.is_train:
            self.lq_frames = getfilelist_with_length(self.lq_root)
            self.gt_frames = getfilelist_with_length(self.gt_root)

        else:
            ## Now: Append the first frame name, then load all frames based on the clip length
            self.lq_frames = getfolderlist(self.lq_root)
            self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        # gt_size_w = gt_size[0]
        # gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        # current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = int(frame_name[:-4])

        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            # ensure not exceeding the borders
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # each clip has 100 frames starting from 0 to 99. TODO: if the training clip is not 100 frames [√]
            # Training start frames should be 0
            while (start_frame_idx < 0) or (end_frame_idx > current_len - 1):
                center_frame_idx = random.randint(self.num_half_frames * interval,
                                                  current_len - self.num_half_frames * interval)
                start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # Sample number should equal to the numer we set
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for tmp_id, frame in enumerate(frame_list):
            # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.png')
            img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            current_frame = Image.open(img_gt_path).convert("RGB")
            img_gt = rgb2lab(current_frame)
            img_gts.append(np.array(img_gt))

        ###########################################crop and degradation##########################################################
        if self.is_train:
            # img_lqs = img_gts
            img_gts = paired_random_crop_240(img_gts, gt_size, self.scale, clip_name)  # crop
            # print(img_lqs[0].shape)
            # img_lqs, img_gts = degradation_video_colorization_v1(img_gts)  # a,b channels to 0 :degradation
            img_lqs, img_gts = degradation_video_colorization(img_gts)
        ########################################augmentation - flip, rotate#############################################################
        # augmentation - flip, rotate
        img_lqs.extend(img_gts)

        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        if self.data_config['normalizing']:
            # transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = Normalize_LAB(img_results[i])


        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        # print(img_lqs[0, 0:1, :, :])  # a channel
        # print(img_gts[0, 0:1, :, :])

        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list}

    def __len__(self):
        return len(self.lq_frames)


class Film_color_test_dataset(data.Dataset):  ##  for REDS dataset

    def __init__(self, data_config):
        super(Film_color_test_dataset, self).__init__()

        self.data_config = data_config
        self.scale = data_config['scale']
        self.gt_root = data_config['dataroot_gt']
        self.lq_root = data_config['dataroot_lq']
        self.is_train = False

        ## TODO: dynamic frame num for different video clips
        # self.num_frame = data_config['num_frame']
        # self.num_half_frames = data_config['num_frame'] // 2
        ## Now: Append the first frame name, then load all frames based on the clip length

        # self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        interval_length = self.data_config.get('interval_length', 1)
        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name

        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name

        center_frame_idx = int(frame_name[:-4])

        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))

        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []

        for tmp_id, frame in enumerate(frame_list):
            img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            img_gt = rgb2lab(Image.open(img_gt_path).convert("RGB"))
            img_gts.append(np.array(img_gt))

        for i in range(len(img_gts)):  ##TODO: inference stage: set the non-reference AB channel to 0 [√]
            img_gts[i] = cv2.resize(img_gts[i], (gt_size[1], gt_size[0]), interpolation=cv2.INTER_AREA)

        # img_lqs, img_gts = degradation_video_colorization_v1(img_gts)
        img_lqs, img_gts = degradation_video_colorization_test(img_gts, interval_length)
        img_lqs.extend(img_gts)

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        if self.data_config['normalizing']:
            # transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = Normalize_LAB(img_results[i])

        img_lqs = torch.stack(img_results[:current_len], dim=0)
        img_gts = torch.stack(img_results[current_len:], dim=0)

        # print(img_lqs[0, 1:2, :, :])  #a channel
        # print(img_gts[0, 1:2, :, :])

        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list}

    def __len__(self):
        return len(self.gt_frames)

def resize_240_short_side(img):
    frame_pil = transfer_1(img)
    # width, height = frame_pil.size
    # print(frame_pil.size())
    n_h = 240
    n_w = 432
    # if width < height:
    #     new_height = int(368 * height / width)
    #     new_height = new_height // 16 * 16
    #     new_width = 368
    # else:
    #     new_width = int(368 * width / height)
    #     new_width = new_width // 16 * 16
    #     new_height = 368

    # frame_pil = frame_pil.resize((new_width, new_height), resample=Image.BILINEAR)
    frame_pil = frame_pil.resize((n_w, n_h), resample=Image.BILINEAR)

    # return transfer_2(frame_pil.convert("RGB"))
    return np.array(frame_pil).astype(np.float32) / 255.


def getfilelist(file_path):
    all_file = []
    for dir, folder, file in os.walk(file_path):
        for i in file:
            t = "%s/%s" % (dir, i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file


def getfilelist_with_length(file_path):
    all_file = []
    for dir, folder, file in os.walk(file_path):
        for i in file:
            t = "%s/%s" % (dir, i)
            all_file.append((t, len(os.listdir(dir))))

    all_file.sort(key=operator.itemgetter(0))
    return all_file


def getfolderlist(file_path):
    all_folder = []
    for dir, folder, file in os.walk(file_path):
        if len(file) == 0:
            continue
        rerank = sorted(file)
        t = "%s/%s" % (dir, rerank[0])
        if t.endswith('.avi'):
            continue
        all_folder.append((t, len(file)))

    all_folder.sort(key=operator.itemgetter(0))
    # all_folder = sorted(all_folder)
    return all_folder


def resize_256_short_side(img):
    width, height = img.size

    if width < height:
        new_height = int(256 * height / width)
        new_width = 256
    else:
        new_width = int(256 * width / height)
        new_height = 256

    return img.resize((new_width, new_height), resample=Image.BILINEAR)


def resize_368_short_side(img):
    frame_pil = transfer_1(img)

    width, height = frame_pil.size

    if width < height:
        new_height = int(368 * height / width)
        new_height = new_height // 16 * 16
        new_width = 368
    else:
        new_width = int(368 * width / height)
        new_width = new_width // 16 * 16
        new_height = 368

    frame_pil = frame_pil.resize((new_width, new_height), resample=Image.BILINEAR)
    return transfer_2(frame_pil.convert("RGB"))


def augment_color(img, hflip=True, vflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip
    vflip = rot
    rot90 = rot

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return _augment(img)


# def getfolderlist_with_length(file_path):
#     all_folder = []
#     for dir,folder,file in os.walk(file_path):
#         for i in folder:
#             t = "%s/%s"%(dir,i)
#             all_folder.append((t,len(os.listdir(t))))
#     all_folder.sort(key = operator.itemgetter(0))
#     return all_folder


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __name__ == '__main__':

    opt = {
        "interval_list": [1],
        "random_reverse": False,
        "name": "Color",
        "scale": 1,
        "use_flip": False,
        "use_rot": False,
        "dataroot_gt": "/home/jq/Trans/VSR-Transformer-main/data/DAVIS_test/GT",
        "dataroot_lq": "/home/jq/Trans/VSR-Transformer-main/data/DAVIS_test/GT",
        "gt_size": [240, 432],
        "is_train": True,
        "num_frame": 15,
        "normalizing": True
    }
    train_dataset = Film_color_train_dataset(opt)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=1,
        rank=0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=(train_sampler is None),
        num_workers=1,
        sampler=train_sampler)

    for data in train_loader:
        lq_l = data['lq']
        gt_lab = data['gt']
        #
        # print(lq_l.shape)
        # print(gt_lab.shape)

    # test_dataset =  Film_color_test_dataset(opt)
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle= False,
    #     num_workers=1,
    #    )
    #
    # for data in test_loader:
    #
    #     lq_l = data['lq']
    #     gt_lab = data['gt']
    #     lq = lq_l.unsqueeze(2)
    #     a = torch.zeros_like(lq)
    #     b = torch.zeros_like(lq)
    #     lq_lab = torch.cat([lq, a, b],dim=2)
    #     # print(lq_lab[:,:,1,:,:])
    #
    #
    #     # print(lq_l.shape)
    #     # print(gt_lab.shape)
    #
    #     break
