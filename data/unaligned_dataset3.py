import math

import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset
import random
import util.util as util
import torch.nn.functional as F
def pad_tensor(input):

    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]

def  addnoise(input):
    r, g, b = input[0], input[1], input[2]  # ת����0-1
    y = 1 - (0.299 * r + 0.587 * g + 0.114 * b)

    min_std, max_std = 0, 5/255.
    std = torch.rand(1) * (max_std - min_std) + min_std
    noise = torch.FloatTensor(input.shape)
    torch.normal(mean=0.0,std=std,out=noise)
    # min_lam, max_lam = 1/255, 5/255
    # lam = torch.rand(1) * (max_lam - min_lam) + min_lam
    # noised = torch.poisson(lam * input) / lam
    output = input + y * noise
    output = torch.clamp(output, min=0, max=1)
    return output
def  color_deviation(input):
    a, b, c = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
    output = torch.concat([torch.unsqueeze(a * input[0],dim=0), torch.unsqueeze(b * input[1],dim=0), torch.unsqueeze(c * input[2],dim=0)],dim=0)
    return output

def genMask(input):
    h = input.size(1)
    w = input.size(2)
    # size = 32
    # # rate = 0.75
    # num = int((w * h / (size * size)))
    # pos = random.sample(range(num), k=1)
    # h_num = int(h // size)
    # mask = torch.ones([h, w])
    # for i in pos:
    #     x, y = size * int(i // h_num), size * int(i % h_num)
    #     mask[x: x + size, y: y + size] = 0
    # x, y = size * int(pos // h_num), size * int(pos % h_num)
    mask1 = torch.zeros([1, h, w])
    mask2 = torch.zeros([1, h, w])
    mask3 = torch.zeros([1, h, w])
    mask4 = torch.zeros([1, h, w])
    mask1[:, ::2, ::2] = 1
    mask2[:, ::2, 1::2] = 1
    mask3[:, 1::2, ::2] = 1
    mask4[:, 1::2, 1::2] = 1
    return mask1,mask2,mask3,mask4

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.transform = get_transform(opt)
        if opt.phase == 'train':
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
            self.B_imgs, self.B_paths = store_dataset(self.dir_B)
            self.B_size = len(self.B_paths)
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A_lsrw')
        else:
            self.dir_A = opt.dataroot

        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.A_size = len(self.A_paths)
        print(self.A_size)


    def __getitem__(self, index):
        A_img = self.A_imgs[index % self.A_size]
        A_path = self.A_paths[index % self.A_size]
        if self.opt.phase == 'train':
            B_img = self.B_imgs[index % self.B_size]
            B_path = self.B_paths[index % self.B_size]
            B_img = self.transform(B_img)
        A_img = self.transform(A_img)





        if self.opt.resize_or_crop == 'no':
            input_img = A_img

            A_img_gamma = (A_img.clone() + 1) / 2.
            ra, ga, ba = A_img_gamma[0], A_img_gamma[1], A_img_gamma[2]  # ת����0-1
            ya = (0.299 * ra + 0.587 * ga + 0.114 * ba)
            A_img_gamma = ya * A_img_gamma + (1 - ya) * torch.pow(A_img_gamma, 1 / 2.1)  # �Ľ���gammaУ��

            A_img_gamma = A_img_gamma * 2 - 1

        else:

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx1 = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx1 = torch.LongTensor(idx1)
                A_img = A_img.index_select(2, idx1)


                if self.opt.phase == 'train':
                    idx2 = [i for i in range(B_img.size(2) - 1, -1, -1)]
                    idx2 = torch.LongTensor(idx2)
                    B_img = B_img.index_select(2, idx2)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx1 = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx1 = torch.LongTensor(idx1)
                A_img = A_img.index_select(1, idx1)
                if self.opt.phase == 'train':
                    idx2 = [i for i in range(B_img.size(1) - 1, -1, -1)]
                    idx2 = torch.LongTensor(idx2)
                    B_img = B_img.index_select(1, idx2)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:

                input_img = (A_img.clone() + 1) / 2.
                if self.opt.light == 1:
                    gamma = random.uniform(1.1, 1.8)
                    input_img = torch.pow(input_img, gamma)  # ������ȶԱȶȵ���
                elif self.opt.light == 2:
                    times = random.randint(200,400)/100.
                    input_img = input_img/times
                else:
                    gamma = random.uniform(0.5, 1.5)  # ���ȡ������[0.8,1.7)
                    input_img = torch.pow(input_img, gamma)  # ������ȶԱȶȵ���
                input_img = input_img * 2 - 1

            else:
                input_img = A_img.clone()



            # if self.opt.newgamma:
            #     A_img_gamma = (A_img.clone() + 1) / 2.
            #     ra, ga, ba = A_img_gamma[0], A_img_gamma[1], A_img_gamma[2]  # ת����0-1
            #     ya = (0.299 * ra + 0.587 * ga + 0.114 * ba)
            #     A_img_gamma = torch.clamp(A_img_gamma * 0.5/torch.mean(ya), 0, 1)
            #     A_img_gamma = A_img_gamma * 2 - 1
            #
            # else:
            # if self.opt.nogamma:
            #     A_img_gamma = A_img.clone()
            # else:
            A_img_gamma = (A_img.clone() + 1) / 2.
            ra, ga, ba = A_img_gamma[0], A_img_gamma[1], A_img_gamma[2]  # ת����0-1
            ya = (0.299 * ra + 0.587 * ga + 0.114 * ba)
            A_img_gamma = torch.clamp(ya * A_img_gamma + (1 - ya) * torch.pow(A_img_gamma, 1 / 2.1), 0, 1)  # �Ľ���gammaУ��
            A_img_gamma = A_img_gamma * 2 - 1

            # mask1,mask2,mask3,mask4 = genMask(A_img)

        if self.opt.phase == 'train':
            return {'A': A_img_gamma,'B': B_img, 'A_img': A_img, 'input_img': input_img,
                        'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A_img_gamma,  'A_img': A_img, 'input_img': input_img,
                    'A_paths': A_path, }

    def __len__(self):
        if self.opt.phase == 'train':
            return max(self.A_size, self.B_size)
        else:
            return self.A_size

    def name(self):
        return 'UnalignedDataset'

