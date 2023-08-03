import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os, sys, argparse, pprint
import matplotlib.pyplot as plt
import utils
from dataset import NTUH_dataset, reverse_normalize
from unet import UNET
from tqdm import tqdm
from calculate_min_max_scaler import calculate_min_max_scaler
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", "-P", type=str)
    parser.add_argument("--data_dir", help="2d data directory", type=str, default="/home/kszuyen/project/2d_data_EarlyFrame")
    parser.add_argument("--load_model", help="Load trained model if True", type=utils.str2bool, default='true')
    # parser.add_argument("--image_size", help="training image size", type=int, default=256)
    parser.add_argument("--batch_size", help="training batch size", type=int, default=16)
    parser.add_argument("--learning_rate", help="training learning rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", help="training epochs", type=int, default=300)
    parser.add_argument("--fold", help="specify current fold (1~10)", type=int, default=1)
    parser.add_argument("--plot", type=utils.str2bool, default='false')
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--case", 
        help="""
        1: PT only
        2: CT & PT
        3: MR & PT
        4: CT, MR & PT
        """,
        type=int, default=4
    )
    return parser

def load_config(args):
    cfg = utils.dotdict(dict())
    if args.project_name:
        cfg.project_name = args.project_name
    else:
        print('Please specify current project name by adding argument "--project_name".')
        sys.exit()
    cfg.load_model = args.load_model
    # cfg.data_dir = args.data_dir
    cfg.data_dir = os.path.join(DIR_PATH, f"2d_data_{args.project_name}_fold{args.fold}")
    cfg.case = args.case
    cfg.out_channels = 1
    # cfg.image_size = args.image_size
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.num_epochs = args.num_epochs
    cfg.fold = args.fold
    cfg.plot = args.plot
    cfg.cuda = args.cuda
    # min_max_scaler = args.scaler # min and max value of CT, MR, PT

    root_dir = os.path.join(DIR_PATH, "results", cfg.project_name, f"fold{cfg.fold}")
    models_dir = os.path.join(root_dir, "models_file")
    plots_dir = os.path.join(root_dir, "plots")

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
        os.mkdir(models_dir)
        os.mkdir(plots_dir)
    
    if args.case == 1:
        cfg.mod_name = "pt_only"
        cfg.in_channels = 1
    elif args.case == 2:
        cfg.mod_name = "ct_pt"
        cfg.in_channels = 2
    elif args.case == 3:
        cfg.mod_name = "mr_pt"
        cfg.in_channels = 2
    elif args.case == 4:
        cfg.mod_name = "ct_mr_pt"
        cfg.in_channels = 3
    else:
        print("""Case error, please select case 1, 2, 3 or 4:
        1: PT only
        2: CT & PT
        3: MR & PT
        4: CT, MR & PT
        """)
        sys.exit()
    cfg.output_dir = os.path.join(plots_dir, cfg.mod_name)
    utils.make_dir(cfg.output_dir)
    cfg.ckpt_dir = os.path.join(models_dir, f"{cfg.mod_name}.pth")
    cfg.best_ckpt_dir = os.path.join(models_dir, f"{cfg.mod_name}_best.pth")

    return cfg

def plot_fig(data, output, ground_truth, scaler, output_dir, 
             orig_psnr, new_psnr, orig_ssim, new_ssim, 
             CT=True, MR=True, PT=True, num_sample=3):

    fig = plt.figure()
    c = 2 + CT + MR + PT
    for i in range(num_sample):
        k, j = 1, 0
        if CT:
            fig.add_subplot(num_sample,c,k+i*c)
            plt.imshow(reverse_normalize(scaler=scaler[0])(data[i][j]).detach().cpu().numpy(), cmap="gray")
            plt.title("CT")
            plt.axis('off')
            j, k = j + 1, k + 1
        if MR:
            fig.add_subplot(num_sample,c,k+i*c)
            plt.imshow(reverse_normalize(scaler=scaler[1])(data[i][j]).detach().cpu().numpy(), cmap="gray")
            plt.title("MR")
            plt.axis('off')
            j, k = j + 1, k + 1

        if PT:
            fig.add_subplot(num_sample,c,k+i*c)
            plt.imshow(reverse_normalize(scaler=scaler[2])(data[i][j]).detach().cpu().numpy())
            plt.title("PT")
            plt.axis('off')
            k += 1
        fig.add_subplot(num_sample,c,k+i*c)
        plt.imshow(reverse_normalize(scaler=scaler[2])(output[i][0]).detach().cpu().numpy())
        plt.title("Predict")
        plt.axis('off')
        fig.add_subplot(num_sample,c,(k+1)+i*c)
        plt.imshow(reverse_normalize(scaler=scaler[2])(ground_truth[i][0]).detach().cpu().numpy())
        plt.title("PT_Coreg_Avg")
        plt.axis('off')
    fig.suptitle(f"Epoch: {epoch}\npsnr:{new_psnr:.3f} | ssim:{new_ssim:.3f}\n(original score: psnr:{orig_psnr:.3f} | ssim:{orig_ssim:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch{epoch}.png"))
    plt.close()

def plot_loss_curve(train_loss, valid_loss, epoch, case, output_dir):
    x = np.linspace(0, epoch, epoch)
    plt.plot(x, train_loss, linestyle='-', label="Train")
    plt.plot(x, valid_loss, linestyle='--', label="Validation")
    plt.legend()
    plt.suptitle(f"Case {case}: Loss curve")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

def plot_eval_metric_curve(orig_psnr, psnr_list, orig_ssim, ssim_list, epoch, case, output_dir):
    x = np.linspace(0, epoch, epoch)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.axhline(orig_psnr, linestyle=':', color=color, label="original PSNR")
    ax1.plot(x, psnr_list, linestyle='-.', color=color, label="PSNR")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.axhline(orig_ssim, linestyle=':', color=color, label="original SSIM")
    ax2.plot(x, ssim_list, linestyle='--', color=color, label="SSIM")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend(loc='lower right')
    fig.suptitle(f"Case {case}: PSNR & SSIM")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eval_metric_curve.png"))
    plt.close()

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    cfg = load_config(args)
    print("Training with config:")
    pprint.pprint(cfg)

    """ set cuda """
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    elif cfg.cuda == 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cuda:0')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}.")
    """  load model  """
    model = UNET(in_ch=cfg.in_channels, out_ch=cfg.out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss() # MSSSIM(), nn.L1Loss(), nn.MSELoss()
    if cfg.load_model and os.path.exists(cfg.ckpt_dir):
        print(f"Loading checkpoint from {cfg.ckpt_dir}...")
        ckpt = torch.load(cfg.ckpt_dir, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        orig_psnr, orig_ssim = ckpt['orig_psnr'], ckpt['orig_ssim']
        best_psnr, best_ssim = ckpt['psnr'], ckpt['ssim']
        train_loss, valid_loss = ckpt['train_loss'], ckpt['valid_loss']
        ssim_list, psnr_list = ckpt['ssim_list'], ckpt['psnr_list']
        min_max_scaler = ckpt['min_max_scaler']
        best_epoch = torch.load(cfg.best_ckpt_dir, map_location=device)['epoch']
        
    else:
        start_epoch = best_epoch = 0
        best_ssim = 0.0
        best_psnr = 0.0
        train_loss, valid_loss = [], []
        ssim_list, psnr_list = [], []

        min_max_scaler = calculate_min_max_scaler(cfg.data_dir)

    train_dataset = NTUH_dataset(
        root_dir=cfg.data_dir,
        dataset_type="train",
        min_max_scaler=min_max_scaler,
        DataAugmentation=True,
        case=cfg.case
    )
    valid_dataset = NTUH_dataset(
        root_dir=cfg.data_dir,
        dataset_type="val",
        min_max_scaler=min_max_scaler,
        DataAugmentation=False,
        case=cfg.case
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=True)
    
    if not (cfg.load_model and os.path.exists(cfg.ckpt_dir)):
        def calculate_orig_score():
            print("calculating original score:")
            running_ssim, running_psnr = 0.0, 0.0
            for data, gt in tqdm(valid_loader):
                mini_batch_size = gt.shape[0]

                data, gt = data.to(device, dtype=torch.float), gt.to(device, dtype=torch.float)

                for k, i in enumerate(range(mini_batch_size)):
                    pet = reverse_normalize(scaler=min_max_scaler[-1])(data[i][-1]).detach().cpu().numpy()
                    pet_cor = reverse_normalize(scaler=min_max_scaler[-1])(gt[i][0]).detach().cpu().numpy()

                    running_psnr += peak_signal_noise_ratio(pet, pet_cor, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
                    running_ssim += structural_similarity(pet, pet_cor, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])

            orig_psnr = running_psnr / len(valid_dataset)
            orig_ssim = running_ssim / len(valid_dataset)
            return orig_psnr, orig_ssim
        orig_psnr, orig_ssim = calculate_orig_score()

    print(f"Original score: psnr:{orig_psnr} | ssim:{orig_ssim}")
    for epoch in range(start_epoch+1, cfg.num_epochs+1):
        print(f"Epoch {epoch}/Case {cfg.case}/Fold {cfg.fold}")
        """  training phase  """
        running_loss = 0.0
        model.train()

        loop = tqdm(train_loader, total=len(train_loader))
        loop.set_description("Train")

        for batch_idx, (data, ground_truth) in enumerate(loop):
            mini_batch_size = ground_truth.shape[0]

            data, ground_truth = data.to(device, dtype=torch.float), ground_truth.to(device, dtype=torch.float)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, ground_truth)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * mini_batch_size
            
        train_loss.append(running_loss / len(train_dataset))

        """  validation phase  """
        running_ssim = 0.0
        running_psnr = 0.0
        running_loss = 0.0
        model.eval()

        loop = tqdm(valid_loader, total=len(valid_loader))
        loop.set_description("Valid")

        for batch_idx, (data, ground_truth) in enumerate(loop):
            mini_batch_size = ground_truth.shape[0]

            data, ground_truth = data.to(device, dtype=torch.float), ground_truth.to(device, dtype=torch.float)

            output = model(data)
            loss = criterion(output, ground_truth)
            
            running_loss += loss.item() * mini_batch_size

            for k, i in enumerate(range(mini_batch_size)):
                pred = reverse_normalize(scaler=min_max_scaler[-1])(output[i][0]).detach().cpu().numpy()
                gt = reverse_normalize(scaler=min_max_scaler[-1])(ground_truth[i][0]).detach().cpu().numpy()

                running_psnr += peak_signal_noise_ratio(pred, gt, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])
                running_ssim += structural_similarity(pred, gt, data_range=min_max_scaler[-1][1]-min_max_scaler[-1][0])

        valid_loss.append(running_loss / len(valid_dataset))

        new_psnr = running_psnr / len(valid_dataset)
        new_ssim = running_ssim / len(valid_dataset)
        psnr_list.append(new_psnr)
        ssim_list.append(new_ssim)
        print("====================================================================")
        print(f"original score: psnr:{orig_psnr} | ssim:{orig_ssim}")
        print(f"new score:      psnr:{new_psnr} | ssim:{new_ssim}")
        print("====================================================================")

        # save checkpoint
        torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'orig_psnr': orig_psnr,
                        'orig_ssim': orig_ssim,
                        'psnr': best_psnr,
                        'ssim': best_ssim,
                        'psnr_list': psnr_list,
                        'ssim_list': ssim_list,
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'min_max_scaler': min_max_scaler
                        }, cfg.ckpt_dir
                        )
        # if achieved best score, save best checkpoint
        if new_psnr>=best_psnr and new_ssim>=best_ssim:
            best_psnr, best_ssim = new_psnr, new_ssim
            best_epoch = epoch
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'orig_psnr': orig_psnr,
                        'orig_ssim': orig_ssim,
                        'psnr': best_psnr,
                        'ssim': best_ssim,
                        'psnr_list': psnr_list,
                        'ssim_list': ssim_list,
                        'train_loss': train_loss,
                        'valid_loss': valid_loss,
                        'min_max_scaler': min_max_scaler
                        }, cfg.best_ckpt_dir
                        )
            print("Model saved at new best score!")
        else:
            print(f"Current best ckpt: epoch {best_epoch}")

        if cfg.plot:
            if cfg.case == 1:
                plot_fig(data, output, ground_truth, min_max_scaler, cfg.output_dir, 
                orig_psnr, new_psnr, orig_ssim, new_ssim, 
                CT=False, MR=False, PT=True)
            elif cfg.case == 2:
                plot_fig(data, output, ground_truth, min_max_scaler, cfg.output_dir, 
                orig_psnr, new_psnr, orig_ssim, new_ssim, 
                CT=True, MR=False, PT=True)
            elif cfg.case == 3:
                plot_fig(data, output, ground_truth, min_max_scaler, cfg.output_dir, 
                orig_psnr, new_psnr, orig_ssim, new_ssim, 
                CT=False, MR=True, PT=True)
            elif cfg.case == 4:
                plot_fig(data, output, ground_truth, min_max_scaler, cfg.output_dir, 
                orig_psnr, new_psnr, orig_ssim, new_ssim, 
                CT=True, MR=True, PT=True)


        plot_loss_curve(
            train_loss, valid_loss, 
            epoch=epoch, case=cfg.case, output_dir=cfg.output_dir)
        plot_eval_metric_curve(
            orig_psnr, psnr_list, orig_ssim, ssim_list, 
            epoch=epoch, case=cfg.case, output_dir=cfg.output_dir)
        plt.close('all')

    print(f"Training complete\nbest score: psnr:{best_psnr} | ssim:{best_ssim}")
        