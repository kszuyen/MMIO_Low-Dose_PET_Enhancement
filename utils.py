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

# def calculate_psnr(img1, img2):
#     # img1 and img2 have range [0, 255]
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))

# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()


# def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')

# def orig_psnr_ssim(loader):
#         running_ssim = 0.0
#         running_psnr = 0.0
#         running_psnr_2 = 0.0
#         running_ssim_2 = 0.0

#         print("calculating original score:")
#         for data, gt in tqdm(loader):
#             mini_batch_size = gt.shape[0]

#             data, gt = data.to(device, dtype=torch.float), gt.to(device, dtype=torch.float)

#             for i in range(mini_batch_size):
#                 # pet = reverse_normalize(scaler=scaler[2])(data[i][2]).detach().cpu().numpy()
#                 # pet_cor = reverse_normalize(scaler=scaler[3])(gt[i][0]).detach().cpu().numpy()
#                 pet = cv2.normalize(data[i][2].detach().cpu().numpy(), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#                 pet_cor = cv2.normalize(gt[i][0].detach().cpu().numpy(), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
#                 running_psnr += calculate_psnr(pet, pet_cor)
#                 running_ssim += calculate_ssim(pet, pet_cor)

#                 running_psnr_2 += peak_signal_noise_ratio(pet, pet_cor, data_range=255)
#                 running_ssim_2 += structural_similarity(pet, pet_cor, data_range=255)


#         psnr = running_psnr / len(valid_dataset)
#         ssim = running_ssim / len(valid_dataset)
#         psnr_2 = running_psnr_2 / len(valid_dataset)
#         ssim_2 = running_ssim_2 / len(valid_dataset)
#         print(f"psnr:{psnr} | ssim:{ssim}")
#         print(f"skimage: psnr:{psnr_2} | ssim:{ssim_2}")
#         return psnr, ssim

if __name__ == "__main__":
    pass
    # from dataset import NTUH_dataset, reverse_normalize
    # from torch.utils.data import DataLoader
    # import torch
    # from tqdm import tqdm
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # root_dir = '/home/kszuyen/project/2d_ct_mr'
    # min_max_scaler = [(-3024.0, 1344.625), (0.0, 3164.0), (-0.41372260451316833, 41806.524908423424), (0.0, 26270.221499204636)]

    # valid_dataset = NTUH_dataset(
    #     root_dir=root_dir,
    #     dataset_type="valid",
    #     min_max_scaler=min_max_scaler,
    #     DataAugmentation=False
    # )
    # valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # orig_psnr_ssim(valid_loader)

    