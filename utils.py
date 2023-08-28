import os, argparse
import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported Value Encountered.')
    
def reverse_normalize(scaler=(0, 27163.332477808)):
    if scaler:
        transform = transforms.Compose([
            transforms.Lambda(lambda p: (p + 1)/2),
            transforms.Lambda(lambda p: p*(scaler[1]-scaler[0])+scaler[0])
        ])
    else:
        transform = transforms.Compose([])
    return transform

def calculate_baseline_score(valid_loader, min_max_scaler, device, psnr_threshold=50):
    running_ssim, running_psnr = 0.0, 0.0
    total_num = 1e-16
    for data, gt in tqdm(valid_loader):
        mini_batch_size = gt.shape[0]

        data, gt = data.to(device, dtype=torch.float), gt.to(device, dtype=torch.float)

        for i in range(mini_batch_size):
            pet = reverse_normalize(scaler=min_max_scaler[-1])(data[i][-1]).detach().cpu().numpy()
            pet_cor = reverse_normalize(scaler=min_max_scaler[-1])(gt[i][0]).detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(pet, pet_cor, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
            if psnr <= psnr_threshold:
                running_psnr += psnr
                running_ssim += structural_similarity(pet, pet_cor, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
                total_num += 1
    orig_psnr = running_psnr / total_num
    orig_ssim = running_ssim / total_num

    return orig_psnr, orig_ssim

def calculate_eval_metrics(model, loader, min_max_scaler, device, psnr_threshold=50):
    """  validation phase  """
    running_ssim = 0.0
    running_psnr = 0.0
    sample_num = 1e-16
    model.eval()

    for batch_idx, (batch_data, batch_gt) in enumerate(loader):
        mini_batch_size = batch_gt.shape[0]
        # sample_num += mini_batch_size
        batch_data, batch_gt = batch_data.to(device, dtype=torch.float), batch_gt.to(device, dtype=torch.float)
        output = model(batch_data)
        
        for k, i in enumerate(range(mini_batch_size)):
            pred = reverse_normalize(scaler=min_max_scaler[-1])(output[i][0]).detach().cpu().numpy()
            gt = reverse_normalize(scaler=min_max_scaler[-1])(batch_gt[i][0]).detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(pred, gt, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
            if psnr <= psnr_threshold:
                running_psnr += psnr
                running_ssim += structural_similarity(pred, gt, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])

    optimized_psnr = running_psnr / sample_num
    optimized_ssim = running_ssim / sample_num

    print(f"optimized score: psnr:{optimized_psnr} | ssim:{optimized_ssim}")
    return optimized_psnr, optimized_ssim


if __name__ == "__main__":
    pass


    
