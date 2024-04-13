import sys
import os
import cv2
import time
import importlib
import datetime
from collections import OrderedDict

sys.path.append(os.path.dirname(sys.path[0]))
import torch

# from basicsr.data.film_dataset import Film_train_dataset, Film_color_train_dataset, Film_color_test_dataset
from basicsr.data.film_dataset_v3 import Film_color_train_dataset, Film_color_test_dataset
from basicsr.utils.data_util import tensor2img
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.models.loss import AdversarialLoss, VGGLoss_torch
from basicsr.utils.util import worker_set_seed, get_root_logger, set_device, seed_worker
from basicsr.models import lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.nn import DataParallel
import torchvision.utils as vutils
from basicsr.models.modules.flow_comp import FlowCompletionLoss
from basicsr.data.util import rgb2lab,rgb2xyz,lab2rgb,lab2xyz,xyz2lab
# from basicsr.data.util import tensor2img
import numpy as np
from torchvision.utils import make_grid
import math

def tensor2img_v1(tensor, out_type=np.float32):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, LAB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, LAB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, LAB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np).round()
    #         Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps).mean()


def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='mean')


def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class Trainer():
    def __init__(self, config, opts, this_logger, debug=False):

        self.config = config
        self.opts = opts
        self.epoch = self.opts.epoch
        self.iteration = 0
        self.is_train = opts.is_train
        self.logger = this_logger
        self.old_lr = config['trainer']['lr']
        self.old_gan_lr = config['trainer']['gan_lr']
        self.schedulers = []
        self.optimizers = []
        # resume_state = config['path']['resume_states']

        #### Define dataset and dataloader first
        # if config['datasets']['train']['name'] == 'REDS':
        #     self.train_dataset = Film_train_dataset(config['datasets']['train'])

        if config['datasets']['train']['name'] == 'colorization':
            self.train_dataset = Film_color_train_dataset(config['datasets']['train'])

        self.val_dataset = Film_color_test_dataset(config['datasets']['val'])

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=opts.world_size, rank=opts.global_rank)

        ## Total # of batch and worker will be multiplied by GPU number
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['datasets']['train']['batch_size_per_gpu'],
                                       shuffle=(self.train_sampler is None),
                                       num_workers=config['datasets']['train']['num_worker_per_gpu'],
                                       pin_memory=True, sampler=self.train_sampler, worker_init_fn=seed_worker)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
                                     sampler=None)

        num_iter_per_epoch = len(self.train_dataset) / (
                config['datasets']['train']['batch_size_per_gpu'] * self.opts.world_size)
        self.total_iters = self.opts.epoch * num_iter_per_epoch
        self.logger.info(
            'Training statistics:'
            f'\n\tNumber of train images: {len(self.train_dataset)}'
            f"\n\tBatch size per gpu: {config['datasets']['train']['batch_size_per_gpu']}"
            f'\n\tWorld size (gpu number): {self.opts.world_size}'
            f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
            f'\n\tTotal epochs: {self.opts.epoch}; iters: {self.total_iters}.')
        self.logger.info(
            '\nValidation statistics:'
            f"\n\tNumber of val images/folders in {config['datasets']['val']['name']}: "f'{len(self.val_dataset)}')
        ####

        #### TODO: Set loss functions: GAN Loss [√], Perception Loss [√]
        self.l1_loss = torch.nn.L1Loss().to(self.opts.device)
        self.MSE_loss = torch.nn.MSELoss().to(self.opts.device)
        self.adversarial_loss = AdversarialLoss(type=self.opts.which_gan).to(self.opts.device)
        self.perceptual_loss = VGGLoss_torch().to(self.opts.device)
        self.flow_comp_loss = FlowCompletionLoss().to(self.opts.device)
        ####

        ####
        net = importlib.import_module('basicsr.models.' + self.opts.model_name)
        self.netG = net.DeepEnhancer().to(self.opts.device)

        load_path = config['path'].get('pretrain_network_g', None)
        strict = config['path'].get('strict_load_g', True)
        if load_path is not None:
            print("loading pretrained model")
            checkpoint = torch.load(load_path)
            # self.netG.load_state_dict(checkpoint, strict=strict)
            self.netG.load_state_dict(checkpoint['netG'], strict=strict)
        # self.print_network(self.netG, self.opts.model_name)

        #### TODO: use Discriminator [√]
        d_net = importlib.import_module('basicsr.models.' + opts.discriminator_name)
        self.netD = d_net.Discriminator(in_channels=3, use_sigmoid=self.opts.which_gan != 'hinge').to(self.opts.device)

        # self.print_network(self.netD, "Discriminator")

        self.setup_optimizers()
        self.setup_schedulers()

        if config['distributed']:
            ### TODO: modify to local rank [√]

            self.netG = DDP(self.netG, device_ids=[opts.local_rank], find_unused_parameters=True)
            self.netD = DDP(self.netD, device_ids=[opts.local_rank])

    def print_network(self, net, model_name):

        if isinstance(net, (DataParallel, DDP)):
            net = net.module

        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        self.logger.info(
            f'Network: {model_name}, with parameters: {net_params:,d}')
        self.logger.info(net_str)

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.config['trainer']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))

        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def optimize_parameters(self, lq, gt, current_iter):
        self.fix_flow_iter = self.config['trainer'].get('fix_flow')
        if self.fix_flow_iter > 0:
            if current_iter == 1:
                for k, v in self.netG.named_parameters():
                    if 'spynet' in k:
                        v.requires_grad = False
            elif current_iter == self.fix_flow_iter + 1:
                for v in self.netG.parameters():
                    v.requires_grad = True

        Discriminator_Loss = 0
        Generator_Loss = 0
        loss_dict = OrderedDict()

        ## Feed Forward
        predicted, pred_flows = self.netG(lq)
        predicted = torch.cat([lq[:, :, :1, :, :], predicted[:,:,1:3,:,:]], dim=2)
        # compute flow completion loss
        flow_loss = self.flow_comp_loss(pred_flows, gt)

        ## Calculate the D Loss and update D
        real_vid_feat = self.netD(gt)
        fake_vid_feat = self.netD(predicted.detach())
        dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
        dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
        Discriminator_Loss += (dis_real_loss + dis_fake_loss) / 2 * self.config['trainer']['D_adv_loss_weight']
        loss_dict['loss_adv_D'] = Discriminator_Loss
        self.optimizer_D.zero_grad()
        Discriminator_Loss.backward()
        self.optimizer_D.step()

        ## Calculate the adversarial loss of Generator
        gen_vid_feat = self.netD(predicted)  # torch.Size([2, 7, 128, 8, 8])
        gan_loss = self.adversarial_loss(gen_vid_feat, True, False) * self.config['trainer']['G_adv_loss_weight']
        Generator_Loss += gan_loss
        loss_dict['loss_adv_G'] = gan_loss

        ## Calculate the l1 loss
        loss_pix = self.l1_loss(predicted, gt) * self.config['trainer']['pix_loss_weight']
        loss_dict['loss_pix'] = loss_pix
        Generator_Loss += loss_pix

        ### Calculate the flow loss
        flow_loss = flow_loss * self.config['trainer']['flow_loss_weight']
        Generator_Loss += flow_loss
        loss_dict['loss_flow'] = flow_loss

        ## Calculate perceptual loss
        _, _, c, h, w = predicted.size()  # 3,256,256

        loss_perceptual = self.perceptual_loss(predicted.contiguous().view(-1, c, h, w),
                                               gt.contiguous().view(-1, c, h, w)) * self.config['trainer'][
                              'perceptual_loss_weight']
        loss_dict['loss_perceptual'] = loss_perceptual
        Generator_Loss += loss_perceptual

        self.optimizer_G.zero_grad()
        Generator_Loss.backward()

        self.optimizer_G.step()

        if self.opts.global_rank == 0 and current_iter % self.config[
            'train_visualization_iter'] == 0:  # Save the training samples

            gt_v = lab2rgb(gt[0])
            lq_v = lab2rgb(lq[0])
            predicted_v = lab2rgb(predicted[0])
            saved_results = torch.cat((gt_v, lq_v, predicted_v), 0)
            # saved_results = torch.cat((gt[0], lq[0], predicted[0]), 0)
            vutils.save_image(saved_results.data.cpu(), os.path.join(self.config['path']['experiments_root'],
                                                                     'training_show_%s.png' % (current_iter)),
                              nrow=self.config['datasets']['train']['num_frame'], padding=0, normalize=False)

        self.log_dict = self.reduce_loss_dict(loss_dict)  ## Gather the loss from other GPUs

    def setup_optimizers(self):

        if self.config['trainer']['flow_lr_mul'] == 0.0:  ## Don't use flow
            optim_params = self.netG.parameters()
        else:
            normal_params = []
            spynet_params = []
            for name, param in self.netG.named_parameters():
                if 'spynet' in name:
                    spynet_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': self.config['trainer']['lr']
                },
                {
                    'params': spynet_params,
                    'lr': self.config['trainer']['lr'] * self.config['trainer']['flow_lr_mul']
                },
            ]

        self.optimizer_G = torch.optim.Adam(optim_params, lr=self.config['trainer']['lr'],
                                            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.config['trainer']['gan_lr'],
                                            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append(
                    [v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_netG_learning_rate(self):
        return [
            param_group['lr'] for param_group in self.optimizers[0].param_groups
        ]

    def get_current_netD_learning_rate(self):
        return [
            param_group['lr'] for param_group in self.optimizers[1].param_groups
        ]

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.
        In distributed training, it averages the losses among different GPUs .
        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.config['distributed']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opts.global_rank == 0:
                    losses /= self.opts.world_size
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def print_iter_message(self, log_vars):

        message = (f"[{self.opts.name[:5]}..][epoch:{log_vars['epoch']:5d}, "f"iter:{log_vars['iter']:8,d}, lr:(")
        for v in log_vars['lr_g']:
            message += f'{v:.3e},'
        message += '), '
        for v in log_vars['lr_d']:
            message += f'{v:.3e},'
        message += ')] '

        ### Timer
        total_time = time.time() - log_vars['start_time']
        time_sec_avg = total_time / log_vars[
            'iter']  ## TODO: if resume training --> total_time / (log_vars['iter'] - start_iter +1)
        eta_sec = time_sec_avg * (self.total_iters - log_vars['iter'] - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        message += f'[eta: {eta_str}, '
        message += f'time (data): {log_vars["iter_time"]:.3f} ({log_vars["data_time"]:.3f})] '
        ### Loss
        for k, v in log_vars.items():
            if k.startswith('loss_'):
                message += f'{k}: {v:.4e} '

        self.logger.info(message)

    def save_model(self, epoch, it):

        net_G_path = os.path.join(self.config['path']['models'], 'net_G_{}.pth'.format(str(it).zfill(5)))
        net_D_path = os.path.join(self.config['path']['models'], 'net_D_{}.pth'.format(str(it).zfill(5)))

        optimizer_path = os.path.join(self.config['path']['models'], 'optimizer_{}.pth'.format(str(it).zfill(5)))

        if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
            netG = self.netG.module
            netD = self.netD.module
        else:
            netG = self.netG
            netD = self.netD

        torch.save({'netG': netG.state_dict()}, net_G_path)
        torch.save({'netD': netD.state_dict()}, net_D_path)
        # torch.save({'netD': netD.state_dict()}, dis_path)
        torch.save({'epoch': epoch,
                    'iteration': it,
                    'optimG': self.optimizer_G.state_dict(),
                    'optimD': self.optimizer_D.state_dict()}, optimizer_path)  ## TODO: Save the schedulers
        torch.save({'epoch': epoch,
                    'iteration': it,
                    'optimG': self.optimizer_G.state_dict(),
                    }, optimizer_path)  ## TODO: Save the schedulers


    def train(self):
        self.logger.info(f'Start training from epoch: {0}, iter: {0}')
        data_time, iter_time = time.time(), time.time()
        start_time = time.time()
        # self.print_network(self.netG, self.config['name'])

        for epoch in range(self.epoch):
            ## Reset the status
            self.train_sampler.set_epoch(epoch)
            for data in self.train_loader:
                data_time = time.time() - data_time
                lq, gt = data['lq'].to(self.opts.device), data['gt'].to(self.opts.device)
                self.iteration += 1

                self.update_learning_rate(
                    self.iteration, warmup_iter=self.config['trainer'].get('warmup_iter', -1))

                self.optimize_parameters(lq, gt, self.iteration)
                iter_time = time.time() - iter_time

                if self.iteration % self.config['logger']['print_freq'] == 0 and self.opts.global_rank == 0:
                    self.old_lr = self.get_current_netG_learning_rate()
                    self.old_gan_lr = self.get_current_netD_learning_rate()

                    log_vars = {'epoch': epoch, 'iter': self.iteration}
                    log_vars.update({'lr_g': self.old_lr,
                                     'lr_d': self.old_gan_lr
                                     })  ## List learning rate: include the lrs of flow and generator
                    log_vars.update({'iter_time': iter_time, 'data_time': data_time, 'start_time': start_time})
                    log_vars.update(self.log_dict)  ## Record the loss values
                    self.print_iter_message(log_vars)

                if self.iteration % self.config['logger']['save_checkpoint_freq'] == 0 and self.opts.global_rank == 0:
                    self.logger.info('Saving models and training states.')
                    self.save_model(epoch, self.iteration)


                if self.iteration % self.config['val']['val_freq'] == 0 and self.opts.global_rank == 0:
                    # self.validation_any_resolution()
                    self.validation()

                data_time = time.time()
                iter_time = time.time()

            # if epoch > self.config['trainer']['nepoch_steady']:
            #     self.update_learning_rate()

    def validation(self):

        interval_length = self.config['datasets']['val']['interval_length']

        if "metrics" in self.config['val']:
            calculate_metric = True
            self.PSNR = 0.0
            self.SSIM = 0.0
        else:
            calculate_metric = False

        for val_data in self.val_loader:  ### Once load all frames

            # val_frame_num = self.config['val']['val_frame_num']
            all_len = val_data['lq'].shape[1]
            all_output = torch.zeros_like(val_data['gt'])
            keyframe_idx = list(range(0, all_len, interval_length + 1))

            if keyframe_idx[-1] == (all_len - 1):
                keyframe_idx = keyframe_idx[:-1]
            clip_name, frame_name = val_data['key'][0].split('/')
            # print(clip_name)
            for k in keyframe_idx:
                current_part = {}
                end_id = min(k + interval_length + 2, all_len)
                current_part['lq'] = val_data['lq'][:, k:end_id, :, :]
                current_part['gt'] = val_data['gt'][:, k:end_id, :, :, :]
                current_part['key'] = val_data['key']
                current_part['frame_list'] = val_data['frame_list'][k:end_id]
                self.part_lq = current_part['lq'].to(self.opts.device)
                self.part_gt = current_part['gt'].to(self.opts.device)

                self.netG.eval()
                with torch.no_grad():
                    self.part_output, _ = self.netG(self.part_lq)
                    self.part_output = torch.cat([self.part_lq[:, :, :1, :, :], self.part_output[:,:,1:3,:,:]], dim=2)
                if self.opts.fix_iter == float("inf"):  ## Eval mode for the flow estimation, even though for training
                    self.netG.module.spynet.eval()

                # all_output.append(self.part_output.detach().cpu().squeeze(0))
                all_output[:, k:end_id, :, :, :] = self.part_output.detach().cpu()

                del self.part_lq
                del self.part_gt
                del self.part_output
            #############
            self.netG.train()
            self.val_output = all_output.squeeze(0)
            self.gt = val_data['gt'].squeeze(0)

            val_output = lab2rgb(self.val_output)  # torch.Size([71, 3, 240, 432])
            gt_rgb_img = lab2rgb(self.gt)

            sr_rgb_img = [tensor2img_v1(np.clip(val_output[i, ...] * 255., 0, 255), np.uint8) for i in
                          range(val_output.size(0))]
            gt_rgb_img = [tensor2img_v1(np.clip(gt_rgb_img[i, ...] * 255., 0, 255), np.uint8) for i in
                          range(gt_rgb_img.size(0))]

            if self.config['val']['save_img']:

                for id, sr_img in zip(val_data['frame_list'], sr_rgb_img):
                    save_place = os.path.join(self.opts.save_dir,
                                              self.config['datasets']['val']['name'],
                                              clip_name, str(id.item()).zfill(5) + '.png')
                    dir_name = os.path.abspath(os.path.dirname(save_place))
                    os.makedirs(dir_name, exist_ok=True)
                    # cv2.imwrite(save_place, sr_img)
                    from PIL import Image
                    sr = Image.fromarray(sr_img)
                    sr.save(save_place)

                    #########################################################################################################################
            if calculate_metric:
                self._initialize_best_metric_results(self.config['datasets']['val']['name'])
                PSNR_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(sr_rgb_img, gt_rgb_img)]
                SSIM_this_video = [calculate_ssim(sr, gt) for sr, gt in zip(sr_rgb_img, gt_rgb_img)]
                self.PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
                self.SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

        self.metric_results = {"psnr": 0, "ssim": 0}
        if calculate_metric:
            self.PSNR /= len(self.val_loader)
            self.SSIM /= len(self.val_loader)

            self.metric_results["psnr"] = self.PSNR
            self.metric_results["ssim"] = self.SSIM

            log_str = f"Validation on {self.config['datasets']['val']['name']}\n"
            log_str += f'\t # PSNR: {self.PSNR:.4f}\n'
            log_str += f'\t # SSIM: {self.SSIM:.4f}\n'

            self._update_best_metric_result(self.config['datasets']['val']['name'], 'psnr', self.PSNR,
                                            self.iteration)
            self._update_best_metric_result(self.config['datasets']['val']['name'], 'ssim', self.SSIM,
                                            self.iteration)

            # self.logger.info(log_str)

            self._log_validation_metric_values(self.iteration, self.config['datasets']['val']['name'])


    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        self.metrics = self.config['val']['metrics']
        for metric, content in self.config['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def _log_validation_metric_values(self, current_iter, dataset_name):

        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

