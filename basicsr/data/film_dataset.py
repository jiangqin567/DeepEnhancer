import random
import torch
from torch.utils import data as data
import os
import numpy as np
import cv2
import operator
from PIL import Image
# from skimage.color import rgb2lab

import torchvision.transforms as transforms
from basicsr.utils.util import get_root_logger
from basicsr.utils.data_util import img2tensor, paired_random_crop, augment,paired_random_crop_240
from basicsr.data.Data_Degradation.util import degradation_video_list, degradation_video_list_2, degradation_video_list_3, degradation_video_list_4, degradation_video_list_simple_debug, degradation_video_colorization, transfer_1, transfer_2, degradation_video_colorization_v2, degradation_video_colorization_v3, degradation_video_colorization_v4
from basicsr.utils.LAB_util import to_mytensor, Normalize_LAB
# import util as util
from basicsr.data.util import rgb2lab,rgb2xyz,lab2rgb,lab2xyz,xyz2lab,read_img

class Film_train_dataset(data.Dataset): ##  for REDS dataset

    def __init__(self, data_config):
        super(Film_train_dataset, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        # self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']

        self.gt_root1, self.lq_root1 = data_config['dataroot_gt1'], data_config['dataroot_lq1']
        self.gt_root2, self.lq_root2 = data_config['dataroot_gt2'], data_config['dataroot_lq2']

        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2

        if self.is_train:
            self.lq_frames = getfilelist_with_length_2(self.lq_root1, self.lq_root2)
            self.gt_frames = getfilelist_with_length_2(self.gt_root1, self.gt_root2)
            # self.lq_frames = getfilelist_with_length_1(self.lq_root)
            # self.gt_frames = getfilelist_with_length_1(self.gt_root)


        else:
            ## Now: Append the first frame name, then load all frames based on the clip length
            self.lq_frames = getfolderlist(self.lq_root)
            self.gt_frames = getfolderlist(self.gt_root)
            # self.lq_frames = []
            # self.gt_frames = []
            # for i in range(len(self.lq_folders))
            #     val_frame_list_this = sorted(os.listdir(self.lq_folders[i]))
            #     first_frame_name = val_frame_list_this[0]
            #     clip_length = len(val_frame_list_this)
            #     self.lq_frames.append((os.path.join(self.lq_folders[i],f'{first_frame_name:08d}.png'),clip_length))
            #     self.gt_frames.append((os.path.join(self.gt_folders[i],f'{first_frame_name:08d}.png'),clip_length))

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
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

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
            while (start_frame_idx < 0) or (end_frame_idx > current_len-1):
                center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval)
                start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # Sample number should equal to the numer we set
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()
            # print(frame_list)

        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []

        # for tmp_id, frame in enumerate(frame_list):
        for tmp_id in frame_list:

            img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

###########################################crop and degradation##########################################################
        if self.is_train:

            img_gts = paired_random_crop_240(img_gts, gt_size, self.scale, clip_name)      #crop
            # print(img_gts[0].shape)
            img_lqs, img_gts = degradation_video_list_4(img_gts, texture_url=self.data_config['texture_template'])    #degradation
########################################augmentation - flip, rotate#############################################################
        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])
        img_results = img2tensor(img_lqs) ## List of tensor

        if self.data_config['normalizing']:
            transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i]=transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        # print(img_lqs.shape)
        # print(img_gts.shape)

        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, \
                'video_name': os.path.basename(current_lq_root), 'name_list': new_clip_sequence}

    def __len__(self):
        return len(self.lq_frames)


class Film_test_dataset(data.Dataset):  ##  for REDS dataset

    def __init__(self, data_config):
        super(Film_test_dataset, self).__init__()

        self.data_config = data_config

        self.scale = data_config['scale']
        self.gt_root = data_config['dataroot_gt']
        self.lq_root =  data_config['dataroot_lq']
        self.is_train = False


        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2

            ## Now: Append the first frame name, then load all frames based on the clip length
        self.lq_frames = getfolderlist(self.lq_root)
        if self.gt_root is not None:
            self.gt_frames = getfolderlist(self.gt_root)

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)

        # key = self.gt_frames[index][0]
        # current_len = self.gt_frames[index][1]
        key = self.lq_frames[index][0]
        current_len = self.lq_frames[index][1]

        ## Fetch the parent directory of clip name

        if self.gt_root is not None:
            current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = int(frame_name[:-4])

        # new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
        new_clip_sequence = sorted(os.listdir(os.path.join(current_lq_root, clip_name)))
        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')
        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for tmp_id, frame in enumerate(frame_list):

            # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.png')
            if self.gt_root is not None:
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
                img_gt = cv2.imread(img_gt_path)
                img_gt = img_gt.astype(np.float32) / 255.
                img_gts.append(img_gt)

            img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])
            img_lq = cv2.imread(img_lq_path)
            img_lq = img_lq.astype(np.float32) / 255.
            img_lqs.append(img_lq)

###########################################crop and degradation##########################################################

        if gt_size is not None:  # validation

            for i in range(len(img_lqs)):
                if self.gt_root is not None:
                    img_gts[i] = resize_short_side(img_gts[i], gt_size)
                img_lqs[i] = resize_short_side(img_lqs[i], gt_size)

########################################augmentation - flip, rotate#############################################################
        img_lqs.extend(img_gts)
        img_results = img2tensor(img_lqs)  ## List of tensor

        if self.data_config['normalizing']:
            transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = transform_normalize(img_results[i])

        img_lqs = torch.stack(img_results[:current_len], dim=0)
        if self.gt_root is not None:
            img_gts = torch.stack(img_results[current_len:], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        # print(img_lqs.shape)
        # print(img_gts.shape)

        if self.gt_root is not None:
            return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, \
                'video_name': os.path.basename(current_lq_root), 'name_list': new_clip_sequence}
        else:
            return {'lq': img_lqs, 'key': key, 'frame_list': frame_list, \
                'video_name': os.path.basename(current_lq_root), 'name_list': new_clip_sequence}

    def __len__(self):
        return len(self.lq_frames)



def resize_240_short_side(img):
    frame_pil = transfer_1(img)
    width, height = frame_pil.size
    # print(frame_pil.size())
    n_h = 240
    n_w = 432
    #
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

    return transfer_2(frame_pil.convert("RGB"))
    # return np.array(frame_pil).astype(np.float32) / 255.


def resize_256_short_side(img):
    frame_pil = transfer_1(img)
    # width, height = frame_pil.size
    # print(frame_pil.size())
    # n_h = 240
    # n_w = 432
    n_h = 256
    n_w = 256
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

    return transfer_2(frame_pil.convert("RGB"))
    # return np.array(frame_pil).astype(np.float32) / 255.


def resize_short_side(img, size):
    frame_pil = transfer_1(img)
    # width, height = frame_pil.size
    # print(frame_pil.size())

    n_h = size[0]
    n_w = size[1]

    # frame_pil = frame_pil.resize((new_width, new_height), resample=Image.BILINEAR)
    frame_pil = frame_pil.resize((n_w, n_h), resample=Image.BILINEAR)

    return transfer_2(frame_pil.convert("RGB"))
    # return np.array(frame_pil).astype(np.float32) / 255.


def getfilelist(file_path):
    all_file = []
    for dir, folder, file in os.walk(file_path):
        for i in file:
            t = "%s/%s" % (dir, i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file

#
def getfilelist_with_length(file_path):
    all_file = []
    for dir, folder, file in os.walk(file_path):

        for i in file:
            t = "%s/%s" % (dir, i)
            all_file.append((t, len(os.listdir(dir))))

    all_file.sort(key=operator.itemgetter(0))
    return all_file


def getfilelist_with_length_1(file_path):
    all_file = []
    for dir, folder, file in os.walk(file_path):

        if folder not in ['000','011','015','020']:
            for i in file:
                t = "%s/%s" % (dir, i)
                all_file.append((t, len(os.listdir(dir))))

    all_file.sort(key=operator.itemgetter(0))
    return all_file


def getfilelist_with_length_2(file_path1, file_path2):
    all_file = []
    for dir, folder, file in os.walk(file_path1):
        for i in file:
            t = "%s/%s" % (dir, i)
            # print(t)
            all_file.append((t, len(os.listdir(dir))))
            all_file.sort(key=operator.itemgetter(0))

    for dir, folder, file in os.walk(file_path2):
        if folder not in ['000', '011', '015', '020']:
            for i in file:
                t = "%s/%s" % (dir, i)
                all_file.append((t, len(os.listdir(dir))))
                all_file.sort(key=operator.itemgetter(0))

    # all_file.sort(key=operator.itemgetter(0))

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

#
# def resize_256_short_side_init(img):
#
#     width, height = img.size
#
#
#     if width < height:
#         new_height = int(256 * height / width)
#         new_width = 256
#     else:
#         new_width = int(256 * width / height)
#         new_height = 256
#
#     return img.resize((new_width, new_height), resample=Image.BILINEAR)
#

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

    opt ={
          "interval_list": [1],
          "random_reverse": False,
        "name": "Color",
        "scale": 1,
        "use_flip": False,
        "use_rot": False,
        # "dataroot_gt1": "/home/jq/Trans/VSR-Transformer-main/data/DAVIS_train/GT",
        # "dataroot_lq1": "/home/jq/Trans/VSR-Transformer-main/data/DAVIS_train/GT",
        "dataroot_gt": "/home/jq/Trans/VSR-Transformer-main/data/REDS/train_sharp",
        "dataroot_lq": "/home/jq/Trans/VSR-Transformer-main/data/REDS/train_sharp",
        "gt_size": [240, 432],
        "is_train": True,
        "num_frame": 15,
        "normalizing": True,
        "texture_template":  "/home/jq/Color/Old_film_restoration/texture_template/noise_data"
    }
    train_dataset = Film_train_dataset(opt)

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
        # print(lq_l.size())
        # print(gt_lab.size())

    #
    # test_dataset = Film_test_dataset(opt)
    #
    #
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
