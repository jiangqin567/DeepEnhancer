''' Towards An End-to-End Framework for Video Inpainting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.modules.flow_comp import SPyNet
from basicsr.models.modules.feat_prop import SecondOrderDeformableAlignment
# from basicsr.models.modules.tfocal_transformer_hq import TemporalFocalTransformerBlock, SoftSplit, SoftComp
from basicsr.models.modules.tfocal_transformer_v1 import TemporalFocalTransformerBlock, SoftSplit, SoftComp
from basicsr.models.modules.spectral_norm import spectral_norm as _spectral_norm
from mmcv.cnn import constant_init

from basicsr.models.modules.flow_comp import flow_warp

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        hidden = 512
        self.stride = 4
        # encoder
        self.encoder = Encoder()

        # decoder
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # feature propagation module

        # self.feat_prop_module = BidirectionalPropagation(channel // 2)

        ############################Bidirectional   Propagation#########################################
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel // 2

        self.LTAM = LTAM(stride=self.stride)

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * self.channel, self.channel, 3, padding=1, deform_groups=16)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * self.channel, self.channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(self.channel, self.channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * self.channel, self.channel, 1, 1, 0)
        ###################################################################################

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = SoftSplit(channel // 2,
                            hidden,
                            kernel_size,
                            stride,
                            padding,
                            t2t_param=t2t_params)
        self.sc = SoftComp(channel // 2, hidden, kernel_size, stride, padding)

        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] -
                           (d - 1) - 1) / stride[i] + 1)

        blocks = []
        depths = 8
        num_heads = [4] * depths
        window_size = [(5, 9)] * depths
        focal_windows = [(5, 9)] * depths
        focal_levels = [2] * depths
        pool_method = "fc"

        for i in range(depths):
            blocks.append(
                TemporalFocalTransformerBlock(dim=hidden,
                                              num_heads=num_heads[i],
                                              window_size=window_size[i],
                                              focal_level=focal_levels[i],
                                              focal_window=focal_windows[i],
                                              n_vecs=n_vecs,
                                              t2t_params=t2t_params,
                                              pool_method=pool_method))
        self.transformer = nn.Sequential(*blocks)

        if init_weights:
            self.init_weights()
            # Need to initial the weights of MSDeformAttn specifically
            for m in self.modules():
                if isinstance(m, SecondOrderDeformableAlignment):
                    m.init_offset()

        # flow completion network
        self.update_spynet = SPyNet()

    def forward_bidirect_flow(self, masked_frames):
        b, t, c, h, w = masked_frames.size()

        # compute forward and backward flows of masked frames
        masked_frames = F.interpolate(masked_frames.view(
            -1, c, h, w),
                                            scale_factor=1 / 4,
                                            mode='bilinear',
                                            align_corners=True,
                                            recompute_scale_factor=True)

        masked_frames = masked_frames.view(b, t, c, h // 4, w // 4)


        mlf_1 = masked_frames[:, :-1, :, :, :].reshape(
            -1, c, h // 4, w // 4)
        mlf_2 = masked_frames[:, 1:, :, :, :].reshape(
            -1, c, h // 4, w // 4)

        pred_flows_forward = self.update_spynet(mlf_1, mlf_2)
        pred_flows_backward = self.update_spynet(mlf_2, mlf_1)

        pred_flows_forward = pred_flows_forward.view(b, t - 1, 2, h // 4,
                                                     w // 4)
        pred_flows_backward = pred_flows_backward.view(b, t - 1, 2, h // 4,
                                                       w // 4)

        return pred_flows_forward, pred_flows_backward


    def forward(self, masked_frames, keyframe_stride=3):

        b, t, ori_c, ori_h, ori_w = masked_frames.size()


        # self.fea_key.eval()
        #
        # # first key frame
        # x_p = masked_frames[:, 0, :, :, :]
        # first_key_HR, fea_forward = self.fea_key(x_p)
        #
        # # last key frame
        # x_n = masked_frames[:, -1, :, :, :]
        # last_key_HR, fea_backward = self.fea_key(x_n)
        # print(first_key_HR.shape)

        # normalization before feeding into the flow completion module
        masked_frames = (masked_frames + 1.) / 2
        flows_backward, flows_forward = self.forward_bidirect_flow(masked_frames)


        # extracting features and performing the feature propagation on local features
        enc_feat = self.encoder(masked_frames.view(b * t, ori_c, ori_h, ori_w))  #torch.Size([7, 128, 60, 108])
        _, c, h, w = enc_feat.size()

        enc_feat = enc_feat.view(b, t, c, h, w)
        output_size = (h, w)

        # Bidirectional Propagation
        b, t, c, h, w = enc_feat.shape
        feats = {}
        feats['spatial'] = [enc_feat[:, i, :, :, :] for i in range(0, t)]

        for module_name in ['backward_', 'forward_']:

            feats[module_name] = []
            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]
                flows = flows_backward
                keyframe_idx = list(range(t - 1, 0, 0 - keyframe_stride))
            else:
                flows = flows_forward
                keyframe_idx = list(range(0, t, keyframe_stride))

            feat_prop = enc_feat.new_zeros(b, self.channel, h, w)  # 128

            sparse_feat_buffers_s1 = []
            sparse_feat_buffers_s2 = []
            sparse_feat_buffers_s3 = []
            index_feat_buffers_s1 = []

            grid_y, grid_x = torch.meshgrid(torch.arange(0, h // self.stride), torch.arange(0, w // self.stride))
            location_update = torch.stack([grid_x, grid_y], dim=0).type_as(masked_frames).expand(b, -1, -1, -1)

            for i, idx in enumerate(frame_idx):

                feat_current = feats['spatial'][mapping_idx[idx]]

                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]
                    cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                    flow = F.adaptive_avg_pool2d(flow_n1, (h // self.stride, w // self.stride)) / self.stride
                    location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1), padding_mode='border',
                                                interpolation="nearest")  # b , 2t , h , w

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)

                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                        flow_n2 = flow_n1 + flow_warp(
                            flow_n2, flow_n1.permute(0, 2, 3, 1))  # torch.Size([b, 2, 60, 108])
                        cond_n2 = flow_warp(feat_n2,
                                            flow_n2.permute(0, 2, 3, 1))

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                               flow_n1,
                                                               flow_n2)  # b, c, h, w     128

                    sparse_feat_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                    sparse_feat_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                    sparse_feat_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)
                    index_feat_s1 = torch.stack(index_feat_buffers_s1, dim=1)

                    feat_prop = self.LTAM(feat_current, index_feat_s1, feat_prop, sparse_feat_s1,
                                          sparse_feat_s2, sparse_feat_s3, location_update)  # 128

                    if idx in keyframe_idx:
                        location_update = torch.cat(
                            [location_update,
                             torch.stack([grid_x, grid_y], dim=0).type_as(feat_current).expand(b, -1, -1, -1)],
                            dim=1)  # n , 2t , h , w

                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]
                                         ] + [feat_prop]
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)  # b, c, h, w     128
                feats[module_name].append(feat_prop)

                if idx in keyframe_idx:
                    # feature tokenization *4
                    # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                    sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride, self.stride), padding=0,
                                                   stride=self.stride)
                    # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                    sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(h // self.stride, w // self.stride),
                                                 kernel_size=(1, 1), padding=0, stride=1)
                    sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                    # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                    index_feat_prop_s1 = F.unfold(feat_current, kernel_size=(self.stride, self.stride), padding=0,
                                                  stride=self.stride)
                    # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                    index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(h // self.stride, w // self.stride),
                                                kernel_size=(1, 1), padding=0, stride=1)
                    index_feat_buffers_s1.append(index_feat_prop_s1)

                    # feature tokenization *6
                    # bs * c * h * w --> # bs * (c*6*6) * (h//4*w//4)
                    sparse_feat_prop_s2 = F.unfold(feat_prop,
                                                   kernel_size=(int(1.5 * self.stride), int(1.5 * self.stride)),
                                                   padding=int(0.25 * self.stride), stride=self.stride)
                    # bs * (c*6*6) * (h//4*w//4) -->  bs * c * (h*1.5) * (w*1.5)
                    sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5 * h), int(1.5 * w)),
                                                 kernel_size=(int(1.5 * self.stride), int(1.5 * self.stride)),
                                                 padding=0, stride=int(1.5 * self.stride))
                    # bs * c * (h*1.5) * (w*1.5) -->  bs * c * h * w
                    sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2, (h, w))
                    # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                    sparse_feat_prop_s2 = F.unfold(sparse_feat_prop_s2, kernel_size=(self.stride, self.stride),
                                                   padding=0, stride=self.stride)
                    # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                    sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(h // self.stride, w // self.stride),
                                                 kernel_size=(1, 1), padding=0, stride=1)
                    sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                    # feature tokenization * 8
                    # bs * c * h * w --> # bs * (c*8*8) * (h//4*w//4)
                    sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2 * self.stride), int(2 * self.stride)),
                                                   padding=int(0.5 * self.stride), stride=self.stride)
                    # bs * (c*8*8) * (h//4*w//4) -->  bs * c * (h*2) * (w*2)
                    sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2 * h), int(2 * w)),
                                                 kernel_size=(int(2 * self.stride), int(2 * self.stride)), padding=0,
                                                 stride=int(2 * self.stride))
                    # bs * c * (h*2) * (w*2) -->  bs * c * h * w
                    sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3, (h, w))
                    # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                    sparse_feat_prop_s3 = F.unfold(sparse_feat_prop_s3, kernel_size=(self.stride, self.stride),
                                                   padding=0, stride=self.stride)
                    # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
                    sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(h // self.stride, w // self.stride),
                                                 kernel_size=(1, 1), padding=0, stride=1)
                    sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        enc_feat =  torch.stack(outputs, dim=1) + enc_feat

        # enc_feat = self.feat_prop_module(enc_feat, masked_frames, pred_flows[0], pred_flows[1], keyframe_stride)
        # content hallucination through stacking multiple temporal focal transformer blocks
        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, output_size)   #torch.Size([B, T, 20, 36, 512])    (60, 108)
        # print(trans_feat.shape)
        trans_feat = self.transformer([trans_feat, output_size])    # feature & size
        # print(trans_feat[0].shape)   #torch.Size([B, T, 20, 36, 512])
        trans_feat = self.sc(trans_feat[0], t, output_size)
        # print(trans_feat.shape)   #torch.Size([T, 128, 60, 108])
        trans_feat = trans_feat.view(b, t, -1, h, w)


        enc_feat = enc_feat + trans_feat    #torch.Size([B, T, 128, 60, 108])

        # decode frames from features
        output = self.decoder(enc_feat.view(b * t, c, h, w))   #torch.Size([7, 3, 240, 432])
        # output = torch.tanh(output)

        _, c, h, w = output.size()
        # print(output.shape)
        output = output.view(b, t, c, h, w)
        return output, [flows_backward, flows_forward]

        # return output

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module


class LTAM(nn.Module):
    def __init__(self, stride=4):
        super().__init__()

        self.stride = stride
        self.fusion = nn.Conv2d(3 * 128, 128, 3, 1, 1, bias=True)

    def forward(self, curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2,
                sparse_feat_set_s3, location_feat):
        """Compute the long-range trajectory-aware attention.

        Args:
            anchor_feat (tensor): Input feature with shape (n, c, h, w)
            sparse_feat_set_s1 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s2 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s3 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            location_feat (tensor): Input location map with shape (n, 2*t, h//4, w//4)

        Return:
            fusion_feature (tensor): Output fusion feature with shape (n, c, h, w).
        """

        n, c, h, w = anchor_feat.size()
        t = sparse_feat_set_s1.size(1)
        feat_len = int(c * self.stride * self.stride)
        feat_num = int((h // self.stride) * (w // self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.contiguous().view(n, t, 2, h // self.stride, w // self.stride).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w // self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h // self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        output_s1 = F.grid_sample(
            sparse_feat_set_s1.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s2 = F.grid_sample(
            sparse_feat_set_s2.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s3 = F.grid_sample(
            sparse_feat_set_s3.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                 (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)

        index_output_s1 = F.grid_sample(
            index_feat_set_s1.contiguous().view(-1, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride)),
            grid_flow.contiguous().view(-1, (h // self.stride), (w // self.stride), 2), mode='nearest',
            padding_mode='zeros', align_corners=True)  # (nt) * (c*4*4) * (h//4) * (w//4)
        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = F.unfold(curr_feat, kernel_size=(self.stride, self.stride), padding=0, stride=self.stride)
        # n * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * (c*4*4)
        curr_feat = curr_feat.permute(0, 2, 1)
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3)  # n * (h//4*w//4) * (c*4*4) * 1

        # cross-scale attention * 4
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        index_output_s1 = index_output_s1.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                            (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len,
                                                                                                  feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3)  # n * (h//4*w//4) * t * (c*4*4)
        # [ n * (h//4*w//4) * t * (c*4*4) ]  *  [ n * (h//4*w//4) * (c*4*4) * 1 ]  -->  n * (h//4*w//4) * t
        matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3)  # n * (h//4*w//4) * t
        matrix_index = matrix_index.view(n, feat_num, t)  # n * (h//4*w//4) * t
        corr_soft, corr_index = torch.max(matrix_index, dim=2)  # n * (h//4*w//4)
        # n * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        corr_soft = corr_soft.unsqueeze(1).expand(-1, feat_len, -1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        corr_soft = F.fold(corr_soft, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s1 = output_s1.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s1 = F.unfold(output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s1 = torch.gather(output_s1.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4)  --> n * (c*4*4) * (h//4*w//4)
        output_s1 = output_s1.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s1 = F.fold(output_s1, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s2 = output_s2.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s2 = F.unfold(output_s2, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s2 = torch.gather(output_s2.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s2 = output_s2.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s2 = F.fold(output_s2, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s3 = output_s3.contiguous().view(n * t, (c * self.stride * self.stride), (h // self.stride),
                                                (w // self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s3 = F.unfold(output_s3, kernel_size=(1, 1), padding=0, stride=1).view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        output_s3 = torch.gather(output_s3.contiguous().view(n, t, feat_len, feat_num), 1,
                                 corr_index.view(n, 1, 1, feat_num).expand(-1, -1, feat_len,
                                                                           -1))  # n * 1 * (c*4*4) * (h//4*w//4)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        output_s3 = output_s3.squeeze(1)
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s3 = F.fold(output_s3, output_size=(h, w), kernel_size=(self.stride, self.stride), padding=0,
                           stride=self.stride)

        out = torch.cat([output_s1, output_s2, output_s3], dim=1)   #384
        out = self.fusion(out)
        out = out * corr_soft
        out += anchor_feat

        return out


if __name__ == '__main__':
    input = torch.randn(1,15,3,240,432).cuda()
    net = InpaintGenerator().cuda()
    print("VSR(REDS) have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters()) / 1000000.0))
    out= net(input)