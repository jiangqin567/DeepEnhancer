
def validation_init(opts, config_dict, loaded_model, val_loader):
    if "metrics" in config_dict['val']:
        calculate_metric = True
        PSNR = 0.0
        SSIM = 0.0
    else:
        calculate_metric = False

    loaded_model.eval()
    # test_clip_par_folder = config_dict['datasets']['val']['dataroot_lq'].split('/')[-1]

    for val_data in val_loader:  ### Once load all frames
        val_frame_num = config_dict['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = []

        clip_name, frame_name = val_data['key'][0].split('/')
        test_clip_par_folder = val_data['video_name'][0]  ## The video name

        frame_name_list = val_data['name_list']

        part_output = None
        for i in range(0, all_len, opts.temporal_stride):
            # print(i)
            current_part = {}
            current_part['lq'] = val_data['lq'][:, i:min(i + val_frame_num, all_len), :, :, :]
            current_part['gt'] = val_data['gt'][:, i:min(i + val_frame_num, all_len), :, :, :]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][i:min(i + val_frame_num, all_len)]
            part_lq = current_part['lq'].cuda()

            # if part_output is not None:
            #     part_lq[:,:val_frame_num-opts.temporal_stride,:,:,:] = part_output[:,opts.temporal_stride-val_frame_num:,:,:,:]

            with torch.no_grad():
                part_output, _ = loaded_model(part_lq)



            if i == 0:
                all_output.append(part_output.detach().cpu().squeeze(0))
            else:
                restored_temporal_length = min(i + val_frame_num, all_len) - i - (val_frame_num - opts.temporal_stride)
                all_output.append(part_output[:, 0 - restored_temporal_length:, :, :, :].detach().cpu().squeeze(0))

            del part_lq

            if (i + val_frame_num) >= all_len:
                break
        #############

        val_output = torch.cat(all_output, dim=0)
        gt = val_data['gt'].squeeze(0)
        lq = val_data['lq'].squeeze(0)

        if config_dict['datasets']['val']['normalizing']:
            val_output = (val_output + 1) / 2
            gt = (gt + 1) / 2
            lq = (lq + 1) / 2

        print(gt.shape)
        print(val_output.shape)
        torch.cuda.empty_cache()

        gt_imgs = []
        sr_imgs = []

        for j in range(len(val_output)):
            gt_imgs.append(tensor2img(gt[j]))
            sr_imgs.append(tensor2img(val_output[j]))


        ### Save the image
        for id, sr_img in enumerate(sr_imgs):
            save_place = os.path.join(opts.save_place, opts.name,
                                      'test_results_' + str(opts.temporal_length) + "_" + str(opts.which_iter),
                                      test_clip_par_folder, clip_name, frame_name_list[id][0])
            dir_name = os.path.abspath(os.path.dirname(save_place))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(save_place, sr_img)

        ### To Video directly TODO: currently only support 1-depth sub-folder test clip [√]
        # if test_clip_par_folder==os.path.basename(opts.input_video_url):
        #     input_clip_url = os.path.join(opts.input_video_url, clip_name)
        # else:
        #     input_clip_url = os.path.join(opts.input_video_url, test_clip_par_folder, clip_name)
        #
        # restored_clip_url = os.path.join(opts.save_place, opts.name, 'test_results_'+str(opts.temporal_length)+"_"+str(opts.which_iter), test_clip_par_folder, clip_name)
        # video_save_url = os.path.join(opts.save_place, opts.name, 'test_results_'+str(opts.temporal_length)+"_"+str(opts.which_iter), test_clip_par_folder, clip_name+'.avi')
        # frame_to_video(input_clip_url, restored_clip_url, video_save_url)
        ###

        if calculate_metric:
            PSNR_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]
            SSIM_this_video = [calculate_ssim(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]
            PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
            SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

    if calculate_metric:
        PSNR /= len(val_loader)
        SSIM /= len(val_loader)

        log_str = f"Validation on {opts.input_video_url}\n"
        log_str += f'\t # PSNR: {PSNR:.4f}\n'
        log_str += f'\t # SSIM: {SSIM:.4f}\n'

        print(log_str)

def validation_v0(opts, config, netG, val_loader):

    interval_length = 13

    if "metrics" in config['val']:
        calculate_metric = True
        PSNR = 0.0
        SSIM = 0.0
    else:
        calculate_metric = False

    for val_data in val_loader:  ### Once load all frames

        # val_frame_num = config['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = torch.zeros_like(val_data['gt'])

        keyframe_idx = list(range(0, all_len, interval_length + 1))

        if keyframe_idx[-1] == (all_len - 1):
            keyframe_idx = keyframe_idx[:-1]
        clip_name, frame_name = val_data['key'][0].split('/')
        print(clip_name)
        # print(val_data['lq'].shape)
        # print(val_data['gt'].shape)

        for k in keyframe_idx:
            current_part = {}
            current_part['lq'] = val_data['lq'][:, k:k + interval_length + 2, :, :]
            current_part['gt'] = val_data['gt'][:, k:k + interval_length + 2, :, :, :]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][k:k + interval_length + 2]
            part_lq = current_part['lq'].cuda()
            part_gt = current_part['gt'].cuda()
            # print(.part_lq.shape)
            # print(.part_gt.shape)

            netG.eval()
            with torch.no_grad():
                part_output, _ = netG(part_lq)
                # print("1111",.part_lq.shape)
                # print("2222", .part_gt.shape)
                # print("3333", .part_output.shape)
                # part_output = torch.cat([part_gt[:, :, :1, :, :], part_output_ab], dim=2)
                # all_output.append(part_output.detach().cpu().squeeze(0))
                all_output[:, k:k + interval_length + 2, :, :, :] = part_output.detach().cpu().squeeze(0)

                del part_lq
                del part_gt
                del part_output
        #############
        # val_output = torch.cat(all_output, dim=0)
        val_output = all_output.squeeze(0)  #torch.Size([71, 3, 240, 432]
        gt_rgb_img = val_data['gt'].squeeze(0)

        val_output = lab2rgb(val_output)  #torch.Size([71, 3, 240, 432])
        gt_rgb_img = lab2rgb(gt_rgb_img)

        sr_rgb_img = [tensor2img_v1(np.clip(val_output[i, ...] * 255., 0, 255), np.uint8) for i in
                      range(val_output.size(0))]
        gt_rgb_img = [tensor2img_v1(np.clip(gt_rgb_img[i, ...] * 255., 0, 255), np.uint8) for i in
                      range(gt_rgb_img.size(0))]


        # import matplotlib.pyplot as plt
        # plt.imshow(sr_rgb_img[0])
        # plt.show()
        # break

        # cv2.imshow("gt.png",gt_rgb_img[0])
        # cv2.waitKey(0)
        # break

        if opts.save_image:
            for id, sr_img in zip(val_data['frame_list'], sr_rgb_img):
                save_place = os.path.join(opts.save_dir,
                                          config['datasets']['val']['name'],
                                          clip_name, str(id.item()).zfill(8) + '.png')
                dir_name = os.path.abspath(os.path.dirname(save_place))
                os.makedirs(dir_name, exist_ok=True)
                cv2.imwrite(save_place, sr_img)
                # sr = Image.fromarray(sr_img)
                # sr.save(save_place)


        if calculate_metric:
            PSNR_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(sr_rgb_img, gt_rgb_img)]
            SSIM_this_video = [calculate_ssim(sr, gt) for sr, gt in zip(sr_rgb_img, gt_rgb_img)]
            PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
            SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

    if calculate_metric:
        PSNR /= len(val_loader)
        SSIM /= len(val_loader)

        log_str = f"Validation on {opts.gt_video_url}\n"
        log_str += f'\t # PSNR: {PSNR:.4f}\n'
        log_str += f'\t # SSIM: {SSIM:.4f}\n'

        print(log_str)


def validation(opts, config, netG, val_loader):
    interval_length = 13

    if "metrics" in config['val']:
        calculate_metric = True
        PSNR = 0.0
        SSIM = 0.0
    else:
        calculate_metric = False

    for val_data in val_loader:  ### Once load all frames

        # val_frame_num = config['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = torch.zeros_like(val_data['gt'])

        keyframe_idx = list(range(0, all_len, interval_length + 1))

        if keyframe_idx[-1] == (all_len - 1):
            keyframe_idx = keyframe_idx[:-1]
        clip_name, frame_name = val_data['key'][0].split('/')
        print(clip_name)
        # print(val_data['lq'].shape)
        # print(val_data['gt'].shape)

        for k in keyframe_idx:
            current_part = {}
            current_part['lq'] = val_data['lq'][:, k:k + interval_length + 2, :, :]
            current_part['gt'] = val_data['gt'][:, k:k + interval_length + 2, :, :, :]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][k:k + interval_length + 2]
            part_lq = current_part['lq'].cuda()
            part_gt = current_part['gt'].cuda()
            # print(.part_lq.shape)
            # print(.part_gt.shape)

            netG.eval()
            with torch.no_grad():
                part_output, _ = netG(part_lq)
                # print("1111",.part_lq.shape)
                # print("2222", .part_gt.shape)
                # print("3333", .part_output.shape)
                # part_output = torch.cat([part_gt[:, :, :1, :, :], part_output_ab], dim=2)
                # all_output.append(part_output.detach().cpu().squeeze(0))
                all_output[:, k:k + interval_length + 2, :, :, :] = part_output.detach().cpu().squeeze(0)

                del part_lq
                del part_gt
                del part_output
        #############
        # val_output = torch.cat(all_output, dim=0)
        val_output = all_output.squeeze(0)  # torch.Size([71, 3, 240, 432]
        gt = val_data['gt'].squeeze(0)

        if config_dict['datasets']['val']['normalizing']:
            val_output = (val_output + 1) / 2
            gt = (gt + 1) / 2

        torch.cuda.empty_cache()

        gt_imgs = []
        sr_imgs = []

        for j in range(len(val_output)):
            gt_imgs.append(tensor2img(gt[j]))
            sr_imgs.append(tensor2img(val_output[j]))

        # import matplotlib.pyplot as plt
        # plt.imshow(sr_rgb_img[0])
        # plt.show()
        # break

        # cv2.imshow("gt.png",gt_rgb_img[0])
        # cv2.waitKey(0)
        # break

        if opts.save_image:
            for id, sr_img in zip(val_data['frame_list'], sr_imgs):
                save_place = os.path.join(opts.save_dir,
                                          config['datasets']['val']['name'],
                                          clip_name, str(id.item()).zfill(8) + '.png')
                dir_name = os.path.abspath(os.path.dirname(save_place))
                os.makedirs(dir_name, exist_ok=True)
                cv2.imwrite(save_place, sr_img)
                # sr = Image.fromarray(sr_img)
                # sr.save(save_place)

        if calculate_metric:
            PSNR_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]
            SSIM_this_video = [calculate_ssim(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]
            PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
            SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

    if calculate_metric:
        PSNR /= len(val_loader)
        SSIM /= len(val_loader)

        log_str = f"Validation on {opts.gt_video_url}\n"
        log_str += f'\t # PSNR: {PSNR:.4f}\n'
        log_str += f'\t # SSIM: {SSIM:.4f}\n'

        print(log_str)


def validation_any_resolution(opts, config_dict, loaded_model, val_loader):
    logger = opts.logger

    if "metrics" in config_dict['val']:
        calculate_metric = True
        PSNR = 0.0
        SSIM = 0.0
    else:
        calculate_metric = False

    loaded_model.eval()
    # test_clip_par_folder = config_dict['datasets']['val']['dataroot_lq'].split('/')[-1]
    metrics_video = {}

    for val_data in val_loader:  ### Once load all frames
        val_frame_num = config_dict['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = []
        clip_name, frame_name = val_data['key'][0].split('/')
        print(clip_name)
        test_clip_par_folder = val_data['video_name'][0]  ## The video name

        frame_name_list = val_data['name_list']
        part_output = None

        for i in range(0, all_len, opts.temporal_stride):
            # print(i)
            current_part = {}
            current_part['lq'] = val_data['lq'][:, i:min(i + val_frame_num, all_len), :, :, :]

            if len(val_data['gt']) > 0:
                current_part['gt'] = val_data['gt'][:, i:min(i + val_frame_num, all_len), :, :, :]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][i:min(i + val_frame_num, all_len)]
            part_lq = current_part['lq'].cuda()
            # if part_output is not None:
            #     part_lq[:,:val_frame_num-opts.temporal_stride,:,:,:] = part_output[:,opts.temporal_stride-val_frame_num:,:,:,:]
            h, w = val_data['lq'].shape[3], val_data['lq'].shape[4]

            with torch.no_grad():

                ################################################################
                mod_size_h = 60
                mod_size_w = 108
                ######################RNN_Swin####################################
                # mod_size_h = 64
                # mod_size_w = 64
                ####################################################################
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                part_lq = torch.cat(
                    [part_lq, torch.flip(part_lq, [3])],
                    3)[:, :, :, :h + h_pad, :]
                part_lq = torch.cat(
                    [part_lq, torch.flip(part_lq, [4])],
                    4)[:, :, :, :, :w + w_pad]
                part_output, _ = loaded_model(part_lq)
                part_output = part_output[:, :, :, :h, :w]
                # print(part_output.shape)

            if i == 0:
                all_output.append(part_output.detach().cpu().squeeze(0))
            else:
                restored_temporal_length = min(i + val_frame_num, all_len) - i - (val_frame_num - opts.temporal_stride)
                all_output.append(part_output[:, 0 - restored_temporal_length:, :, :, :].detach().cpu().squeeze(0))

            del part_lq

            if (i + val_frame_num) >= all_len:
                break
        #############
        val_output = torch.cat(all_output, dim=0)
        if len(val_data['gt']) > 0:
            gt = val_data['gt'].squeeze(0)
        lq = val_data['lq'].squeeze(0)

        if config_dict['datasets']['val']['normalizing']:
            val_output = (val_output + 1) / 2
            if len(val_data['gt']) > 0:
                gt = (gt + 1) / 2
            lq = (lq + 1) / 2
        torch.cuda.empty_cache()

        gt_imgs = []
        sr_imgs = []

        for j in range(len(val_output)):
            if len(val_data['gt']) > 0:
                gt_imgs.append(tensor2img(gt[j]))
            sr_imgs.append(tensor2img(val_output[j]))

        ### Save and evaluate the image
        for id, sr_img in enumerate(sr_imgs):
            save_path = os.path.join(opts.save_place, opts.name,
                                     test_clip_par_folder, clip_name, frame_name_list[id][0])

            dir_name = os.path.abspath(os.path.dirname(save_path))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(save_path, sr_img)

        ### To Video directly TODO: currently only support 1-depth sub-folder test clip [√]
        # if test_clip_par_folder==os.path.basename(opts.input_video_url):
        #     input_clip_url = os.path.join(opts.input_video_url, clip_name)
        # else:
        #     input_clip_url = os.path.join(opts.input_video_url, test_clip_par_folder, clip_name)
        #
        # restored_clip_url = os.path.join(opts.save_place, opts.name, 'test_results_'+str(opts.temporal_length)+"_"+str(opts.which_iter), test_clip_par_folder, clip_name)
        # video_save_url = os.path.join(opts.save_place, opts.name, 'test_results_'+str(opts.temporal_length)+"_"+str(opts.which_iter), test_clip_par_folder, clip_name+'.avi')
        # frame_to_video(input_clip_url, restored_clip_url, video_save_url)
        ###

        if calculate_metric and len(val_data['gt']) > 0:
            PSNR_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]
            SSIM_this_video = [calculate_ssim(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]

            metrics_video.update({clip_name: [sum(PSNR_this_video) / len(PSNR_this_video),
                                              sum(SSIM_this_video) / len(SSIM_this_video)]})

            # log_str = f"{clip_name}\n"
            # log_str += f'\t # PSNR: {sum(PSNR_this_video) / len(PSNR_this_video):.4f}\n'
            # log_str += f'\t # SSIM: {sum(SSIM_this_video) / len(SSIM_this_video):.4f}\n'

            PSNR += sum(PSNR_this_video) / len(PSNR_this_video)
            SSIM += sum(SSIM_this_video) / len(SSIM_this_video)

    # print(metrics_video)
    if calculate_metric and len(val_data['gt']) > 0:
        PSNR /= len(val_loader)
        SSIM /= len(val_loader)

        log_str = f"Validation on {opts.input_video_url}\n"

        for k, v in metrics_video.items():
            log_str += f'\t # {k}: psnr: {v[0]:.4f}, ssim: {v[1]:.4f}\n'
        log_str += f'Average:\n'
        log_str += f'\t # PSNR: {PSNR:.4f}\n'
        log_str += f'\t # SSIM: {SSIM:.4f}\n'

        logger.info(log_str)
        print(log_str)