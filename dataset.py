from torch.utils.data import Dataset
import os, sys
import numpy as np
from torchvision import transforms
import random
from scipy.ndimage import rotate

class NTUH_dataset(Dataset):
    def __init__(self, root_dir, dataset_type="train", case=4, min_max_scaler=None, DataAugmentation=False, resize_image_size=False):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.case = case
        self.train_files = os.listdir(os.path.join(root_dir, dataset_type, "data"))
        self.DataAugmentation = DataAugmentation
        self.min_max_scaler = min_max_scaler
        self.resize_image_size = resize_image_size
        if case == 1: # PT only
            self.min_max_scaler = [self.min_max_scaler[2]]
        elif case == 2: # CT & PT
            self.min_max_scaler = [self.min_max_scaler[0], self.min_max_scaler[2]]
        elif case == 3: # MR & PT
            self.min_max_scaler = [self.min_max_scaler[1], self.min_max_scaler[2]]
        elif case == 4: # CT, MR & PT
            pass
        else:
            print("Error: case not in 1, 2, 3 or 4")
            sys.exit()
        
    def __len__(self):
        return len(self.train_files)
    
    def parse_data(self, data, case):
        CT = data[:,:,0]
        MR = data[:,:,1]
        PT = data[:,:,2]
        if case == 1: # PT only
            data = np.expand_dims(PT, axis=2)
        elif case == 2: # CT & PT
            data = np.stack([CT, PT], axis=2)
        elif case == 3: # MR & PT
            data = np.stack([MR, PT], axis=2)
        elif case == 4: # CT, MR & PT
            data = data
        else:
            print("Error: case not in 1, 2, 3 or 4")
            sys.exit()
        return data
    
    def __getitem__(self, index):
        data_name = self.train_files[index]

        data = np.load(os.path.join(self.root_dir, self.dataset_type, "data", data_name))
        ground_truth = np.load(os.path.join(self.root_dir, self.dataset_type, "ground_truth", data_name))

        data = self.parse_data(data, self.case)
        h = data.shape[0]

        ### data augmentation ###
        if self.DataAugmentation:
            # random flipping
            if random.choice([0, 1]):
                data, ground_truth = np.flipud(data), np.flipud(ground_truth)
            
            # translation
            for directions in (['up', 'down'], ['left', 'right']):
                shift, d = random.randint(0, int(h*0.02)), random.choice(directions)
                if shift:
                    data, ground_truth = translate(data, shift, d), translate(ground_truth, shift, d)
            # rotation
            if random.choice([0, 1]):
                angle = random.randint(-5, 5)
                data, ground_truth = rotate_img(data, angle), rotate_img(ground_truth, angle)
        ### data augmentation output shape
        # print(data.shape, ground_truth.shape)
        ###

        # data, ground_truth = torch.tensor(data).permute(2,0,1), torch.tensor(ground_truth.copy()).unsqueeze(0)
        data, ground_truth = transforms.ToTensor()(data.copy()), transforms.ToTensor()(ground_truth.copy())

        if self.min_max_scaler:
            for i in range(data.shape[0]):
                data[i,:,:] = transforms.Lambda(lambda p: (p-self.min_max_scaler[i][0])/(self.min_max_scaler[i][1]-self.min_max_scaler[i][0]))(data[i,:,:])
            ground_truth = transforms.Lambda(lambda p: (p-self.min_max_scaler[-1][0])/(self.min_max_scaler[-1][1]-self.min_max_scaler[-1][0]))(ground_truth)
            transform = transforms.Lambda(lambda p: (p * 2) - 1)
            data, ground_truth = transform(data), transform(ground_truth)

        if self.resize_image_size:
            resize = transforms.Resize(self.resize_image_size)
            data, ground_truth= resize(data), resize(ground_truth)
        
        if self.dataset_type == "test":
            patient = data_name.split('.')[0].split('_')[0]
            slc = int(data_name.split('.')[0].split('_')[1])
            
            return data, ground_truth, patient, slc
        else:
            return data, ground_truth

"""  functions for numpy array image augmentations:
https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd  """

def translate(img, shift=10, direction='right', roll=False):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be down|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img

def rotate_img(img, angle=10, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img

if __name__ == "__main__":
    train_dataset = NTUH_dataset(
        root_dir="/home/kszuyen/MMIO_Low-Dose_PET_Enhancement/2d_data_LowDose_with_T1_fold1",
        dataset_type="train",
        min_max_scaler=[(-63.38726043701172, 131.9373779296875), (0.0, 3361.0), (-4348.736393213272, 51885.97814440727)],
        DataAugmentation=False,
        case=4
    )
    data, gt = train_dataset[0]
    print(data.shape, gt.shape)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # fig.add_subplot(2, 2, 1)
    # plt.imshow(data[0])
    # fig.add_subplot(2, 2, 2)
    # plt.imshow(data[1])
    # fig.add_subplot(2, 2, 3)
    # plt.imshow(data[2])
    # fig.add_subplot(2, 2, 4)
    # plt.imshow(gt[0])
    # plt.savefig('1.png')
