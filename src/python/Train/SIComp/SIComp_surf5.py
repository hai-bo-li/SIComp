import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import warnings
import visdom
from torch.optim import Adam
from tqdm import tqdm
from SIComp.src.python.utils_all.pytorch_ssim import *
from SIComp.src.python.utils_all.Datasets import FlowFormerDataset
from SIComp.src.python.core.FlowFormer import build_flowformer
from SIComp.src.python.configs.SIComp.SIComp_surf5_config import get_cfg
from SIComp.src.python.utils_all.differential_color_function import *
import lpips
from SIComp.src.python.All_Model import *
from SIComp.src.python.utils_all.utils import *
def calculate_image_metrics(pred, target):
    # l1
    mae = l1_fun(pred, target).item()
    #l2
    mse = l2_fun(pred, target).item()
    # RMSE calculation
    rmse = math.sqrt(mse * 3)  # Multiply by 3 because the average is computed over 3 channels
    # PSNR calculation
    if mse == 0:
        psnr = float('inf')  # Avoid division-by-zero errors
    else:
        psnr = 10 * math.log10(1 / mse)
    # ssim calculation
    ssim = ssim_fun(pred, target).item()
    # Compute DeltaE
    xl_batch = rgb2lab_diff(pred, device)
    yl_batch = rgb2lab_diff(target, device)
    diff_map = ciede2000_diff(xl_batch, yl_batch, device)
    diff_map = diff_map.mean().item()
    # Compute LPIPS
    lpips = loss_fn_lpips(pred, target)
    lpips = lpips.mean().item()  # Take the mean to obtain a scalar loss
    return mae, rmse, psnr, ssim, diff_map, lpips


def computeLoss(prj_pred, prj_train, loss_option):
    train_loss = 0
    # l1
    if 'l1' in loss_option:
        l1_loss = l1_fun(prj_pred, prj_train)
        train_loss += l1_loss
    # l2
    l2_loss = l2_fun(prj_pred, prj_train)
    if 'l2' in loss_option:
        train_loss += l2_loss
    # ssim
    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))
        train_loss += ssim_loss

    if 'diff' in loss_option:
        xl_batch = rgb2lab_diff(prj_pred, device)
        yl_batch = rgb2lab_diff(prj_train, device)
        diff_map = ciede2000_diff(xl_batch, yl_batch, device)
        color_loss = diff_map.mean()
        train_loss += 0.005 * color_loss

    if 'lpips' in loss_option:
        lpips_loss = loss_fn_lpips(prj_pred, prj_train)
        train_loss += lpips_loss.mean()  # Take the mean to obtain a scalar loss
    return train_loss


def flow_test_loadData(dataset_root, setup_name, data_type='crop'):
    # data paths
    # This helper only loads a single setup
    data_root = fullfile(dataset_root, setup_name)
    cam_ref_path = fullfile(str(data_root), 'cam/{}/ref'.format(data_type))
    cam_valid_path = fullfile(str(data_root), 'cam/{}/test'.format(data_type))
    prj_valid_path = fullfile(dataset_root, 'test')
    print("Loading data from '{}'".format(data_root))
    # training data
    cam_surf = Get_dataLoader(cam_ref_path, index=[0, 31, 62, 93, 124], cfg=cfg)  # ref/img_0126.png is cam-captured surface image i.e., s when img_gray.png i.e., x0 projected
    # validation data
    cam_valid = Get_dataLoader(cam_valid_path, cfg=cfg)
    prj_valid = Get_dataLoader(prj_valid_path, cfg=cfg)
    return cam_surf, cam_valid, prj_valid


def evaluate(model, data_loader, steps, save_folder):
    save_vis_test_image_path = os.path.join(save_folder, f"test_step_{steps}.png")
    model.eval()
    metrics = {
        'mae': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0,
        'diff_map': 0, 'lpips': 0
    }
    total_batches = 0
    with torch.no_grad():
        # Wrap data_loader with tqdm
        for i_batch, data_blob in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
            cam_crop, prj_GT, surfs_crop = [x.cuda() for x in data_blob]
            # Run model prediction
            warped_predict = model(prj_GT, cam_crop, surfs_crop)
            # Compute metrics
            mae, rmse, psnr, ssim, diff_map, lpips = calculate_image_metrics(warped_predict, prj_GT)
            # Accumulate each metric value
            metrics['mae'] += mae
            metrics['rmse'] += rmse
            metrics['psnr'] += psnr
            metrics['ssim'] += ssim
            metrics['diff_map'] += diff_map  # Accumulate diff_map
            metrics['lpips'] += lpips  # Accumulate lpips
            total_batches += 1
            if i_batch == 0:
                visdom_display_save(vis, cam_crop, warped_predict, prj_GT, cfg.batch_size, steps, 'test', max_nrow=5,
                                    save_path=save_vis_test_image_path)
    # Compute the average of all metrics
    avg_mae = metrics['mae'] / total_batches
    avg_rmse = metrics['rmse'] / total_batches
    avg_psnr = metrics['psnr'] / total_batches
    avg_ssim = metrics['ssim'] / total_batches
    avg_diff_map = metrics['diff_map'] / total_batches  # Average diff_map
    avg_lpips = metrics['lpips'] / total_batches  # Average lpips
    # Return all averaged values
    return avg_mae, avg_rmse, avg_psnr, avg_ssim, avg_diff_map, avg_lpips


def train(cfg):
    stop_training = False
    SIComp_surf5.train()
    # Load the training datasets
    train_datasets = [
        FlowFormerDataset(datasets_module.SIComp_dataset1_root, datasets_module.SIComp_dataset1_lists, phase='train', num_train=500,
                          transforms=data_transforms,
                          surf_index=cfg.surf_indices),
        FlowFormerDataset(datasets_module.CompenNet_plus_plus_root, datasets_module.CompenNet_plus_plus_lists, phase='train',
                          num_train=500,
                          transforms=data_transforms, surf_index=cfg.surf_indices),
        FlowFormerDataset(datasets_module.SIComp_dataset2_root, datasets_module.SIComp_dataset2_lists, phase='train', num_train=500,
                          transforms=data_transforms,
                          surf_index=cfg.surf_indices),
        FlowFormerDataset(datasets_module.CompenHR_root, datasets_module.CompenHR_lists, phase='train', num_train=500,
                          transforms=data_transforms,
                          surf_index=cfg.surf_indices)
    ]
    validate_dataset = FlowFormerDataset(datasets_module.Validate_root, datasets_module.Validate_lists,
                                     phase='test',
                                     num_train=200,
                                     transforms=data_transforms,
                                     surf_index=cfg.surf_indices)
    train_loader, validate_loader = create_dataloader(train_datasets=train_datasets, test_dataset=validate_dataset, cfg=cfg)
    FlowFormer_Opt = Adam(SIComp_surf5.FlowFormer.parameters(), lr=cfg.trainer.flow_learning_rate,
                          weight_decay=cfg.trainer.flow_weight_decay)
    SCNet_Opt = Adam(SIComp_surf5.CompenNeSt.parameters(), lr=cfg.trainer.cmp_learning_rate, weight_decay=cfg.trainer.cmp_weight_decay)
    # Set learning rate schedulers
    scheduler_flowformer = get_scheduler(FlowFormer_Opt, cfg.trainer.flow_scheduler)
    scheduler_SCNet = get_scheduler(SCNet_Opt, cfg.trainer.comp_scheduler)
    total_steps = 0
    # Create the Visdom window for visualizing validation metrics
    metrics = ["test_l1", "test_RMSE", "train_loss", "test_ssim", "0.01*test_diff", "test_lpips"]
    win_metrics = vis.line(
        X=np.array([0]),
        Y=np.array([[0] * len(metrics)]),
        opts=dict(
            title=f"{name}",
            xlabel="Iterations",
            ylabel="Metrics",
            legend=metrics
        )
    )

    for epoch in tqdm(range(cfg.trainer.num_epochs), desc='train_epoch'):
        for i_batch, data_blob in enumerate(train_loader):
            if i_batch == 0:
                print('Training...')
            FlowFormer_Opt.zero_grad()
            SCNet_Opt.zero_grad()
            cam_crop, prj_GT, surfs_crop = [x.to(device) for x in data_blob]
            step = epoch * len(train_loader) + i_batch
            warped_predict = SIComp_surf5(prj_GT, cam_crop, surfs_crop, step=step)
            # warped_predict = OmniCompNet_full(prj_GT, cam_crop, surfs_crop)
            # If _CN.l1_pretrain = True, use pure l1 for the first 4000 steps,
            # then switch to 'l1+ssim+diff+lpips'
            if cfg.trainer.l1_pretrain:
                if total_steps <= cfg.trainer.l1_pretrain_num:
                    train_loss = computeLoss(warped_predict, prj_GT, 'l1')
                else:
                    train_loss = computeLoss(warped_predict, prj_GT, cfg.train_loss)
            else:
                train_loss = computeLoss(warped_predict, prj_GT, cfg.train_loss)

            train_loss.backward()
            FlowFormer_Opt.step()
            SCNet_Opt.step()

            # Update schedulers (if iteration-based schedulers are used)
            if isinstance(scheduler_flowformer, LambdaLR):
                scheduler_flowformer.step()
            else:
                scheduler_flowformer.step()
            if isinstance(scheduler_SCNet, LambdaLR):
                scheduler_SCNet.step()
            else:
                scheduler_SCNet.step()

            if total_steps % cfg.trainer.train_vis_num == 0:
                save_vis_train_image_path = os.path.join(save_vis_img_folder, f"train_step_{total_steps}.png")
                visdom_display_save(vis, cam_crop, warped_predict, prj_GT, cfg.batch_size, total_steps, 'train', max_nrow=5,
                                    save_path=save_vis_train_image_path)

            if total_steps % cfg.test.test_vis_num == 0:
                test_mae, test_rmse, test_psnr, test_ssim, test_diff_map, test_lpips = evaluate(SIComp_surf5, validate_loader,
                                                                                                total_steps, save_vis_img_folder)

                save_test_log_one_line(
                    save_metrics_folder, model_name, cfg.batch_size, cfg.trainer.flow_learning_rate, cfg.train_loss,
                    1, 1, 1, 1, 1, test_psnr, test_rmse, test_ssim, test_diff_map, test_lpips, f'{name}.txt',
                    f"{name}.xlsx"
                )
                save_config_and_model_info(cfg, SCNet, save_metrics_folder)

                vis.line(
                    X=np.array([[total_steps] * len(metrics)]),
                    Y=np.array([[test_mae, test_rmse, train_loss.item(), test_ssim, 0.01 * test_diff_map, test_lpips]]),
                    win=win_metrics,
                    update='append'
                )

                save_vis_line_path = os.path.join(save_vis_line_folder, f"vis_line_{total_steps}.png")
                save_vis_line(vis, win_metrics, save_vis_line_path)
            # Save checkpoint helper call
            total_steps += 1
            save_one_model_checkpoint(SIComp_surf5, total_steps, cfg.trainer.save_num, save_checkpoint_folder, current_time, is_final=False)
            if total_steps >= cfg.trainer.Max_iters:
                stop_training = True
                break
        if stop_training:
            break
        if not isinstance(scheduler_flowformer, LambdaLR):
            scheduler_flowformer.step()
        if not isinstance(scheduler_SCNet, LambdaLR):
            scheduler_SCNet.step()

    save_one_model_checkpoint(SIComp_surf5, total_steps, cfg.trainer.save_num, save_checkpoint_folder, current_time, is_final=True)


if __name__ == '__main__':
    cfg = get_cfg()
    resetRNGseed(cfg.trainer.randseed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = 'SIComp_surf5'
    print(f"train_loss: {cfg.train_loss}\nflow_learning_rate: {cfg.trainer.flow_learning_rate}\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    choice = "sicomp"
    # choice = input("Please input: lab2 lab3 SL_lab2 SL_lab3 sl_local local: ").strip()
    datasets_module = import_datasets_module(choice)
    vis = visdom.Visdom(port=cfg.vis_port)
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    ssim_fun = SSIM().to(device)
    # Stage selection
    Train_Model = True
    Compensation_Model = False
    Test_Model = False
    FlowFormer = build_flowformer(cfg)
    if cfg.restore_ckpt is not None:
        load_state_dict_without_module(cfg.restore_ckpt, FlowFormer)
    FlowFormer.to(device)
    # Create model instances
    SCNet = SCNet_surf5().to(device)
    model_name = SCNet.__class__.__name__
    load_state_dict_without_module(cfg.SCNet_pretrain, SCNet)
    SIComp_surf5 = Connection(FlowFormer, SCNet, warp_func=warp_images).to(device)
    save_vis_line_folder = create_folder_with_time(f"../../{name}/img", current_time)
    save_metrics_folder = create_folder_with_time(f"../../{name}/log", current_time)
    save_vis_img_folder = create_folder_with_time(f"../../{name}/img", current_time)
    save_checkpoint_folder = create_folder_with_time(f'../../{name}/checkpoint', current_time)

    print('-------------------------------------- Training Options -----------------------------------')
    if Train_Model:
        train(cfg)
    print('All dataset done!')
    torch.cuda.empty_cache()
