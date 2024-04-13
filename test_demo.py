import sys
import os
import cv2
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
import argparse
import yaml
import torchvision.transforms as transforms
from basicsr.data.film_dataset import resize_240_short_side

sys.path.append(os.path.dirname(sys.path[0]))
import torch
from basicsr.data.dataset import Film_dataset
from basicsr.data.film_dataset import Film_test_dataset
from basicsr.utils.util import worker_set_seed, get_root_logger, set_device, frame_to_video,get_time_str
from basicsr.utils.data_util import tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
import logging
from torch.utils.data import DataLoader

def Load_model(opts, config_dict):
    net = importlib.import_module('basicsr.models.' + opts.model_name)
    netG = net.DeepEnhancer()

    model_path = os.path.join('./pretrained_models', opts.model_name,
                              '{}.pth'.format(str(opts.which_iter)))
    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()
    print("Finish loading model ...")
    return netG


def Load_dataset(opts, config_dict):
    val_dataset = Film_test_dataset(config_dict['datasets']['val'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, sampler=None)
    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')
    return val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='real_old_film', help='The name of this experiment')
    parser.add_argument('--model_name', type=str, default='DeepEnhancer', help='The name of adopted model')
    parser.add_argument('--which_iter', type=str, default='restore', help='Load which iteraiton')
    parser.add_argument('--task', type=str, default='restore',help='task configuration')
    parser.add_argument('--temporal_length', type=int, default=15,
                        help='How many frames should be processed in one forward')
    parser.add_argument('--temporal_stride', type=int, default=10, help='Stride value while sliding window')
    parser.add_argument('--save_image', type=str, default='True', help='save')
    parser.add_argument('--save_place', type=str, default='visual_restore_results', help='save place')

    opts = parser.parse_args()
    lq_root = opts.name + '/' + 'degradation_lq_full'

    with open(os.path.join('./configs', opts.task + '.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    val_frame_num = opts.temporal_length
    temporal_stride = opts.temporal_stride
    opts.save_dir = opts.save_place + '/' + opts.model_name
    os.makedirs(opts.save_dir,exist_ok=True)
    log_file = os.path.join(opts.save_dir,
                            f"test_{opts.name}_{get_time_str()}.log")

    opts.logger = get_root_logger(
        logger_name='Video_Process', log_level=logging.INFO, log_file=log_file)

    model = Load_model(opts, config_dict)
    video_list = sorted(os.listdir(lq_root))

    log_str = f"{opts}\n"
    for video in video_list:
        print(video)
        log_str += f"Validation on {video}\t"
        lq_video_path = os.path.join(lq_root, video)

        frame_list = sorted(os.listdir(lq_video_path))
        all_len = len(frame_list)
        img_lqs = []


        for tmp_id, frame in enumerate(frame_list):

            img_lq_path = os.path.join(lq_root, video, frame)
            img_lq = cv2.imread(img_lq_path)
            img_lq = img_lq.astype(np.float32) / 255.
            img_lqs.append(img_lq)

        from basicsr.utils.data_util import img2tensor
        img_results = img2tensor(img_lqs)

        transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        for i in range(len(img_results)):
            img_results[i] = transform_normalize(img_results[i])

        img_lqs = torch.stack(img_results[:all_len], dim=0)

        all_output = []
        model.eval()

        for i in range(0, all_len, opts.temporal_stride):
            current_part = {}
            current_part['lq'] = img_lqs[i:min(i + val_frame_num, all_len), :, :, :]
            current_part['key'] = frame
            current_part['frame_list'] = frame_list[i:min(i + val_frame_num, all_len)]
            part_lq = current_part['lq'].cuda()

            part_lq = part_lq.unsqueeze(0)

            h, w = part_lq.shape[3], part_lq.shape[4]

            with torch.no_grad():

                mod_size_h = config_dict['datasets']['val']['crop_size'][0]
                mod_size_w = config_dict['datasets']['val']['crop_size'][1]

                ####################################################################
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w

                part_lq = torch.cat(
                    [part_lq, torch.flip(part_lq, [3])],
                    3)[:, :, :, :h + h_pad, :]
                part_lq = torch.cat(
                    [part_lq, torch.flip(part_lq, [4])],
                    4)[:, :, :, :, :w + w_pad]

                part_output, _ = model(part_lq)
                part_output = part_output[:, :, :, :h, :w]


            if i == 0:
                all_output.append(part_output.detach().cpu().squeeze(0))
            else:
                restored_temporal_length = min(i + val_frame_num, all_len) - i - (val_frame_num - opts.temporal_stride)
                all_output.append(part_output[:, 0 - restored_temporal_length:, :, :, :].detach().cpu().squeeze(0))

            del part_lq
            if (i + val_frame_num) >= all_len:
                break

        val_output = torch.cat(all_output, dim=0).squeeze(0)
        val_output = (val_output + 1) / 2

        sr_imgs = []
        for j in range(len(val_output)):
            sr_imgs.append(tensor2img(val_output[j]))

        ## Save the image
        for id, sr_img in zip(frame_list, sr_imgs):
            save_place = os.path.join(opts.save_dir, opts.name, video, id)  # e.g. test_results_20_7000
            dir_name = os.path.abspath(os.path.dirname(save_place))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(save_place, sr_img)
