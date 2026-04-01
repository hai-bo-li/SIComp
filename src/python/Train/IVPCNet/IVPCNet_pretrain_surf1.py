import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import warnings
import visdom
from torch.optim import Adam
from tqdm import tqdm
from SIComp.src.python.All_Model import *
from SIComp.src.python.utils_all import Datasets as New_dataset
from SIComp.src.python.configs.IVPCNet_config.SCNet_pretrain_surf1_config import get_cfg
from SIComp.src.python.utils_all.differential_color_function import *
import lpips
from SIComp.src.python.utils_all.utils import *

def calculate_image_metrics(pred, target):
    # l1
    mae = l1_fun(pred, target).item()
    # l2
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
    train_loss = 0.
    l1_loss = 0.
    ssim_loss = 0.
    lpips_loss = 0.
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
    return train_loss, l1_loss, l2_loss, ssim_loss, lpips_loss




def evaluate_SL(model, valid_dataloader, steps):
    save_vis_test_image_path = os.path.join(save_vis_img_folder, f"test_step_{steps}.png")
    model.eval()
    total_metrics = {
        'mae': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0,
        'diff_map': 0, 'lpips': 0  # Add diff_map and lpips
    }
    total_batches = 0
    with torch.no_grad():
        for i_batch, data_blob in enumerate(valid_dataloader):
            cam_valid = data_blob['cam'].to(device)
            prj_GT = data_blob['prj'].to(device)
            cam_surf = data_blob['surf'].to(device)
            predicted = model(cam_valid, cam_surf)
            mae, rmse, psnr, ssim, diff_map, lpips = calculate_image_metrics(predicted,prj_GT)
            # Accumulate all metric values
            total_metrics['mae'] += mae
            total_metrics['rmse'] += rmse
            total_metrics['psnr'] += psnr
            total_metrics['ssim'] += ssim
            total_metrics['diff_map'] += diff_map  # Accumulate diff_map
            total_metrics['lpips'] += lpips  # Accumulate lpips
            total_batches += 1
            if i_batch == 0:
                visdom_display_save(vis, cam_valid, predicted, prj_GT, cfg.batch_size, steps, 'test', max_nrow=5,
                                    save_path=save_vis_test_image_path)

    # Compute the average of all metrics
    avg_mae = total_metrics['mae'] / total_batches
    avg_rmse = total_metrics['rmse'] / total_batches
    avg_psnr = total_metrics['psnr'] / total_batches
    avg_ssim = total_metrics['ssim'] / total_batches
    avg_diff_map = total_metrics['diff_map'] / total_batches  # Average diff_map
    avg_lpips = total_metrics['lpips'] / total_batches  # Average lpips

    return avg_mae, avg_rmse, avg_psnr, avg_ssim, avg_diff_map, avg_lpips

def train_SL(cfg):
    stop_training = False
    SCNet.train()
    train_datasets = [
        New_dataset.CompenNetMultiDataset(datasets_module.SIComp_dataset1_root, datasets_module.SIComp_dataset1_lists, 'train', 'warpSL',
                                          surf_idx=cfg.surf_indices, transforms=data_transforms),
        New_dataset.CompenNetMultiDataset(datasets_module.Compensated_dataset_root, datasets_module.Compensated_data_lists, 'train',
                                          'warpSL', surf_idx=cfg.surf_indices, transforms=data_transforms, CMP=True),
        New_dataset.CompenNetMultiDataset(datasets_module.CompenNet_dataset_root, datasets_module.CompenNet_data_lists, 'train', 'warpSL',
                                          surf_idx=cfg.surf_indices, transforms=data_transforms),
        New_dataset.CompenNetMultiDataset(datasets_module.SIComp_dataset2_root, datasets_module.SIComp_dataset2_lists, 'train',
                                          'warpSL', surf_idx=cfg.surf_indices, transforms=data_transforms),
        New_dataset.CompenNetMultiDataset(datasets_module.CompenNet_plus_plus_root, datasets_module.CompenNet_plus_plus_lists,
                                          'train',
                                          'warpSL',
                                          surf_idx=cfg.surf_indices, transforms=data_transforms),
        New_dataset.CompenNetMultiDataset(datasets_module.Dark_dataset_root, datasets_module.Dark_data_lists, 'train',
                                          'warpSL',
                                          surf_idx=cfg.surf_indices, transforms=data_transforms),
    ]
    valid_dataset = New_dataset.CompenNetMultiDataset(datasets_module.Validate_root, datasets_module.Validate_lists, 'valid', 'warpSL',
                                                     surf_idx=cfg.surf_indices, transforms=data_transforms)
    train_loader, valid_loader = create_dataloader(train_datasets=train_datasets, test_dataset=valid_dataset, cfg=cfg)
    SCNet_Opt = Adam(SCNet.parameters(), lr=cfg.trainer.cmp_learning_rate, weight_decay=cfg.trainer.cmp_weight_decay)
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
            prj_GT = data_blob['prj'].to(device)
            cam_train = data_blob['cam'].to(device)
            cam_surf = data_blob['surf'].to(device)
            # predict and compute loss
            predicted = SCNet(cam_train, cam_surf)
            train_loss, l1_loss, l2_loss, ssim_loss, lpips_loss = computeLoss(predicted, prj_GT, cfg.train_loss)
            train_rmse = math.sqrt(l2_loss.item() * 3)
            SCNet_Opt.zero_grad()
            train_loss.backward()
            SCNet_Opt.step()

            scheduler_SCNet.step()
            if total_steps % cfg.trainer.train_vis_num == 0:
                save_vis_train_image_path = os.path.join(save_vis_img_folder, f"train_step_{total_steps}.png")
                visdom_display_save(vis, cam_train, predicted, prj_GT, cfg.batch_size, total_steps, 'train', max_nrow=5,
                                    save_path=save_vis_train_image_path)

            if total_steps % cfg.test.test_vis_num == 0:
                test_mae, test_rmse, test_psnr, test_ssim, test_diff_map, test_lpips = evaluate_SL(SCNet, valid_loader,
                                                                                                   steps=total_steps)

                save_test_log_one_line(
                    save_metrics_folder, SCNet.__class__.__name__, cfg.batch_size, cfg.trainer.cmp_learning_rate, cfg.train_loss,
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
            save_one_model_checkpoint(SCNet, total_steps, cfg.trainer.save_num, save_checkpoint_folder, current_time, is_final=False)
            if total_steps >= cfg.trainer.Max_iters:
                stop_training = True
                break
        if stop_training:
            break
    save_one_model_checkpoint(SCNet, total_steps, cfg.trainer.save_num, save_checkpoint_folder, current_time, is_final=True)
    del cam_train
    del prj_GT
    del cam_surf
    torch.cuda.empty_cache()



if '__main__' == __name__:
    cfg = get_cfg()
    resetRNGseed(cfg.trainer.randseed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    name = 'SCNet_pretrain_surf1'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    choice = "sl_local"
    # choice = input("Please input: lab2  lab3  sl_lab2  sl_lab3  sl_local  local: ").strip()
    datasets_module = import_datasets_module(choice)
    vis = visdom.Visdom(port=cfg.vis_port)
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    ssim_fun = pytorch_ssim.SSIM().to(device)
    data_transforms = {'surf': None,
                       'cam': cv_transforms.Compose([cv_transforms.ToTensor()]),
                       'prj': cv_transforms.Compose([cv_transforms.ToTensor()])}
    SCNet = SCNet_surf1()
    SCNet = SCNet.to(device)
    save_vis_line_folder = create_folder_with_time(f"../../{name}/img", current_time)
    save_metrics_folder = create_folder_with_time(f"../../{name}/log", current_time)
    save_vis_img_folder = create_folder_with_time(f"../../{name}/img", current_time)
    save_checkpoint_folder = create_folder_with_time(f'../../{name}/checkpoint', current_time)
    print('-------------------------------------- Training Options -----------------------------------')
    train_SL(cfg)
    torch.cuda.empty_cache()
    print('Training done!')
