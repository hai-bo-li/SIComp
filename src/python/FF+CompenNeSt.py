import warnings
from tqdm import tqdm
from core.FlowFormer import build_flowformer
from configs.local.FF_CompenNeSt_Compensation_Image_config import get_cfg
from All_Model import *
from utils_all.Datasets import FlowFormerDataset
from utils_all.pytorch_ssim import *
from utils_all.differential_color_function import *
from lpips import lpips
import argparse
from utils_all.utils import *
import time

def calculate_image_metrics(pred, target):

    with torch.no_grad():
        # l1
        mae = l1_fun(pred, target).item()
        # l2
        mse = l2_fun(pred, target).item()
        # RMSE calculation
        rmse = math.sqrt(mse)
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


def flow_test_loadData(dataset_root, setup_name, cfg, data_type='crop'):
    data_root = fullfile(dataset_root, setup_name)
    cam_ref_path = fullfile(str(data_root), 'cam/{}/ref'.format(data_type))
    cam_valid_path = fullfile(str(data_root), 'cam/{}/test'.format(data_type))
    prj_valid_path = fullfile(dataset_root, 'test')
    print("Loading data from '{}'".format(data_root))
    cam_surf = Get_dataLoader(cam_ref_path, index=cfg.surf_indices, cfg=cfg)
    cam_valid = Get_dataLoader(cam_valid_path, cfg=cfg)
    prj_valid = Get_dataLoader(prj_valid_path, cfg=cfg)
    return cam_surf, cam_valid, prj_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["valid", "actual"], required=False, default='valid',
                        help="mode: 'valid' or 'actual'")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    cfg = get_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    ssim_fun = SSIM().to(device)

    name = cfg.name
    model_name = cfg.model_name
    save_metrics_folder = create_folder_with_time(f"../../{name}/log", current_time)

    choice = "test_local"
    datasets_module = import_datasets_module(choice)
    print('------------------------------------ Data Processing  ---------------------------')
    # apply_mask_and_crop_images_actualMetric(datasets_module.valid_data_root, datasets_module.valid_data_lists, metrics=False, im_mask=True)
    total_images_saved = 0
    FlowFormer = build_flowformer(cfg)
    CompenNeSt = CompenNeSt()
    print('successful load the pretrain file')

    if args.mode == "actual":
        FF_CompenNeSt = FlowCompensationModel(FlowFormer, CompenNeSt, warp_func=warp_images, actual_compensation=True).to(device)
        FF_CompenNeSt.load_state_dict(torch.load(cfg.FF_CompenNeSt_pretrain_path))
        FF_CompenNeSt.eval()
        print('------------------------------------ Start actual compensation  ---------------------------')
        target_idx = 33  # Use frame 33 from the training set to compute the optical flow

        for data_name in datasets_module.valid_data_lists:
            print(f"\n>>> Processing Actual Dataset: [{data_name}]")

            # --- Step 1: extract the reference training frame (index 33) for the current dataset and compute fixed flow ---
            flow_init_dataset = FlowFormerDataset(
                datasets_module.valid_data_root, [data_name],
                phase='train',
                num_train=target_idx + 1,
                transforms=data_transforms, surf_index=cfg.surf_indices
            )
            blob = flow_init_dataset[target_idx]
            cam_train = blob[0].unsqueeze(0).to(device)  # [1, C, H, W]
            gt_train = blob[1].unsqueeze(0).to(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device=device)
                start_time = time.time()

            # --- Step 2: compute and cache the fixed optical flow for the current dataset ---
            with torch.no_grad():
                h_t, w_t = cam_train.shape[-2:]
                # If the size does not match, use the high-resolution flow predictor
                if (h_t, w_t) != FF_CompenNeSt.train_size:
                    f_cached = predict_flow_highres(
                        FF_CompenNeSt.flow_model, gt_train, cam_train,
                        target_size=(h_t, w_t), train_size=FF_CompenNeSt.train_size
                    )
                else:
                    f_cached = FF_CompenNeSt.flow_model(gt_train, cam_train)

                # Store the optical flow in the model's fixed_flow attribute
                FF_CompenNeSt.fixed_flow = f_cached.detach()
                # --- Optional: visualize and save the optical flow map ---
                # Build the save path
                prj_cmp_path = os.path.join(datasets_module.valid_data_root, data_name, f'prj/cmp/{model_name}')
                # if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)
                # flow_vis_path = os.path.join(prj_cmp_path, "fixed_flow_visualization.png")
                # # Make sure prj_cmp_path already exists
                # if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)
                #
                # save_flow_to_vismap(OmniCompNet_full.fixed_flow, flow_vis_path, max_flow=20)
                #
                # print(f">>> Standard optical flow visualization saved to: {flow_vis_path}")
            print(f">>> Dataset [{data_name}] flow fixed using Training Frame Index {target_idx}.")

            # --- Step 3: prepare test data and ground-truth desire images ---
            desire_test_path = os.path.join(datasets_module.valid_data_root, data_name, 'cam/crop/desire')
            desire_test_loader = Get_dataLoader(desire_test_path, cfg=cfg)
            assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)

            test_dataset = FlowFormerDataset(
                datasets_module.valid_data_root,
                [data_name],
                phase='test',
                num_train=5,  # This usually refers to the number of reference training frames and does not affect test loading
                transforms=data_transforms,
                surf_index=cfg.surf_indices
            )
            test_loader = DataLoader(test_dataset, cfg.batch_size, shuffle=False, num_workers=0)
            total_images_saved = 0

            # --- Step 4: start the compensation loop ---
            with torch.no_grad():
                # Use zip to align the test data and desire ground truth
                for data_blob, desire_test in tqdm(zip(test_loader, desire_test_loader),
                                                   total=len(test_loader), desc=f"Actual CMP {data_name}"):
                    cam_crop, prj_GT, surfs_crop = [x.to(device) for x in data_blob]
                    desire_test = desire_test.permute(0, 3, 1, 2).float().div(255).to(device)

                    # Core prediction: the model should already be configured to use fixed_flow
                    # Calling forward here avoids running the internal FlowFormer again
                    prj_cmp_test = FF_CompenNeSt(prj_GT, cam_crop, surfs_crop, desire_test)

                    # Save the compensated projection images
                    flow_saveImgs(prj_cmp_test, prj_cmp_path, start_idx=total_images_saved)
                    total_images_saved += prj_cmp_test.size(0)

                # --- Step 5: summarize results for the current dataset ---
                print(f'>>> Compensation images for [{data_name}] saved to {prj_cmp_path}')

                elapsed = time.time() - start_time
                hours, rem = divmod(elapsed, 3600)
                minutes, seconds = divmod(rem, 60)
                time_lapse = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

                if device.type == 'cuda':
                    max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                else:
                    max_mem = 0.0

                print(f"Dataset [{data_name}] Status -> Time: {time_lapse}, Max VRAM: {max_mem:.2f} MB")
    elif args.mode == "valid":
        print('------------------------------------ Start Valid -------------------------------')
        # 1. Initialize the model
        FF_CompenNeSt = FlowCompensationModel(FlowFormer, CompenNeSt, warp_func=warp_images).to(device)
        FF_CompenNeSt = FF_CompenNeSt.to(device)

        # 2. Load the weights
        FF_CompenNeSt.load_state_dict(torch.load(cfg.FF_CompenNeSt_pretrain_path))
        FF_CompenNeSt.eval()
        # 3. Start the regular validation loop
        all_metrics = {'mse': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0, 'deltaE': 0, 'lpips': 0}
        dataset_count = 0
        target_idx = 33
        for data_name in datasets_module.valid_data_lists:
            print(f"\n>>> Processing Dataset: [{data_name}]")
            # --- Step 1: extract only the specified reference training frame for the current dataset ---
            # Set num_train to target_idx + 1 to ensure the index is valid with no extra redundancy
            flow_init_dataset = FlowFormerDataset(
                datasets_module.valid_data_root, [data_name],
                phase='train',
                num_train=target_idx + 1,
                transforms=data_transforms, surf_index=cfg.surf_indices
            )
            # Read the specific frame directly by index to avoid DataLoader overhead
            blob = flow_init_dataset[target_idx]
            cam_train = blob[0].unsqueeze(0).to(device)  # [1, C, H, W]
            gt_train = blob[1].unsqueeze(0).to(device)
            # --- Step 2: compute and cache the fixed optical flow for the current dataset ---
            with torch.no_grad():
                h_t, w_t = cam_train.shape[-2:]
                if (h_t, w_t) != FF_CompenNeSt.train_size:
                    f_cached = predict_flow_highres(
                        FF_CompenNeSt.flow_model, gt_train, cam_train,
                        target_size=(h_t, w_t), train_size=FF_CompenNeSt.train_size
                    )
                else:
                    f_cached = FF_CompenNeSt.flow_model(gt_train, cam_train)

                # Store the dataset-specific optical flow in fixed_flow
                FF_CompenNeSt.fixed_flow = f_cached.detach()

            print(f">>> Dataset [{data_name}] flow fixed using Frame Index {target_idx}.")
            total_images_saved = 0
            test_path = os.path.join(datasets_module.valid_data_root, data_name, f'prj/test/{model_name}')
            # Load the current test set
            test_dataset = FlowFormerDataset(
                datasets_module.valid_data_root,
                [data_name],
                phase='test',  # Test phase
                num_train=100,
                transforms=data_transforms,
                surf_index=cfg.surf_indices
            )
            test_loader = DataLoader(test_dataset, cfg.batch_size, shuffle=False, num_workers=0)

            start_time = time.time()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device=device)

            with torch.no_grad():
                total_metrics = {'mse': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0, 'deltaE': 0, 'lpips': 0}
                batch_count = 0

                for i_batch, data_blob in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing {data_name}:"):
                    cam_crop, prj_GT, surfs_crop = [x.to(device) for x in data_blob]

                    # Calling forward here will automatically use fixed_flow instead of running FlowFormer
                    predicted = FF_CompenNeSt(GT=prj_GT, x=cam_crop, s=surfs_crop)

                    # Metric computation
                    mse, rmse, psnr, ssim, deltaE, lpips = calculate_image_metrics(predicted, prj_GT)

                    # Accumulate each metric
                    total_metrics['mse'] += mse
                    total_metrics['rmse'] += rmse
                    total_metrics['psnr'] += psnr
                    total_metrics['ssim'] += ssim
                    total_metrics['deltaE'] += deltaE
                    total_metrics['lpips'] += lpips

                    # Save images
                    flow_saveImgs(predicted, test_path, start_idx=total_images_saved)
                    # # 4. Key fix: save lossless tensors (.pt)
                    # # At this point total_images_saved has not been updated yet,
                    # # so idx starts from 00000 and aligns perfectly with GT
                    # pt_save_path = os.path.join(test_path, 'tensors')
                    # os.makedirs(pt_save_path, exist_ok=True)
                    #
                    # for j in range(predicted.size(0)):
                    #     idx = total_images_saved + j  # 这里的 idx 现在是正确的了
                    #     torch.save(predicted[j].detach().cpu(), os.path.join(pt_save_path, f"{idx:05d}.pt"))

                    # 5. Update the index at the end
                    total_images_saved += predicted.size(0)
                    batch_count += 1

                # Average metrics for the current dataset
                avg_mse = total_metrics['mse'] / batch_count
                avg_rmse = total_metrics['rmse'] / batch_count
                avg_psnr = total_metrics['psnr'] / batch_count
                avg_ssim = total_metrics['ssim'] / batch_count
                avg_deltaE = total_metrics['deltaE'] / batch_count
                avg_lpips = total_metrics['lpips'] / batch_count
                # time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
                elapsed = time.time() - start_time
                hours, rem = divmod(elapsed, 3600)
                minutes, seconds = divmod(rem, 60)
                time_lapse = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
                print("Elapsed time:", time_lapse)
                if device.type == 'cuda':
                    max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # Convert to MB
                else:
                    max_mem = 0.0
                print(f"valid_time lapse : {time_lapse},valid_memory : {max_mem}")
                print(
                    f'[{data_name}] Average metrics: MSE={avg_mse:.4f}, RMSE={avg_rmse:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, deltaE={avg_deltaE:.4f}, LPIPS={avg_lpips:.4f}')
                print(f'{avg_mse:.4f} {avg_rmse:.4f}  {avg_psnr:.4f}  {avg_ssim:.4f}  {avg_deltaE:.4f}  {avg_lpips:.4f}')
                print(f'Test images saved to: {test_path}')

                # Save the log for each dataset
                log_surrogate(save_metrics_folder, current_time, data_name, model_name, cfg.train_loss, cfg.num_train,
                              cfg.batch_size,
                              avg_psnr, avg_rmse, avg_ssim, avg_deltaE, avg_lpips)
                # Accumulate averages across all datasets
                all_metrics['mse'] += avg_mse
                all_metrics['rmse'] += avg_rmse
                all_metrics['psnr'] += avg_psnr
                all_metrics['ssim'] += avg_ssim
                all_metrics['deltaE'] += avg_deltaE
                all_metrics['lpips'] += avg_lpips
                dataset_count += 1

                # Aggregate log over all datasets
            if dataset_count > 0:
                mean_mse = all_metrics['mse'] / dataset_count
                mean_rmse = all_metrics['rmse'] / dataset_count
                mean_psnr = all_metrics['psnr'] / dataset_count
                mean_ssim = all_metrics['ssim'] / dataset_count
                mean_deltaE = all_metrics['deltaE'] / dataset_count
                mean_lpips = all_metrics['lpips'] / dataset_count

                print('\n==== Average metrics across datasets ====')
                print(
                    f'MSE={mean_mse:.4f}, RMSE={mean_rmse:.4f}, PSNR={mean_psnr:.4f}, SSIM={mean_ssim:.4f}, deltaE={mean_deltaE:.4f}, LPIPS={mean_lpips:.4f}')
                log_surrogate(save_metrics_folder, current_time, "ALL", model_name, cfg.train_loss, cfg.num_train, cfg.batch_size,
                              mean_psnr, mean_rmse, mean_ssim, mean_deltaE, mean_lpips)
            print('-------------------------------------- Done! --------------------------------------/n')
