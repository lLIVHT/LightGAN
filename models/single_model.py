import copy

import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
import torch.nn.functional as F
# from . import networks
from pytorch_msssim import MS_SSIM
import matplotlib.pyplot as plt
from PIL import Image

class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        from . import networks as networks
        nb = opt.batchSize
        size = opt.fineSize
        self.base_seed = 42
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.A_img = self.Tensor(nb, opt.input_nc, size, size)


        self.epoch1 = (opt.niter + opt.niter_decay)//4
        self.epoch2 = 2 * self.epoch1
        self.epoch3 = 3 * self.epoch1

        self.loss_G = 0
        self.loss_G_A = 0
        self.loss_D_A = 0
        self.loss_D_fea = 0
        self.loss_D_P = 0
        self.loss_G_g = 0
        self.loss_G_fea = 0
        if self.isTrain:
            self.ssim = MS_SSIM(data_range=1.0)
            self.colorloss = networks.Colorloss()
            if opt.vgg > 0 or opt.fea == 1:
                self.vgg_loss = networks.PerceptualLoss(opt)
                self.vgg_loss = self.vgg_loss.cuda()  # 先放到 GPU
                # self.vgg_loss = nn.DataParallel(self.vgg_loss, device_ids=opt.gpu_ids)
                self.vgg = networks.load_vgg16("./model", self.gpu_ids)
                self.vgg.eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
                self.dino_loss = networks.DinoFeatureLoss()
                self.dino_loss = self.dino_loss.cuda()
                # self.dino_loss = nn.DataParallel(self.dino_loss, device_ids=opt.gpu_ids)
                # self.clip_loss = networks.CLIPSemanticLoss().cuda()
                # self.yolo_loss = networks.YOLOv8FeatureLoss().cuda()

        skip = True if opt.skip > 0 else False

        self.netG_A_s = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, "student", opt.norm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)

        if self.isTrain:
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                skip=skip, opt=opt)
            self.load_network(self.netG_A, 'G_A', opt.which_epoch, self.gpu_ids,
                              save_path="./checkpoints/lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror_p2/latest_net_G_A.pth")
        if self.isTrain:
            use_sigmoid = True
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)

            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
            if self.opt.fea == 1:
                self.netD_fea = networks.define_D(512, opt.ndf,
                                                  opt.which_model_netD_fea,
                                                  opt.n_layers_patchD, opt.norm, True, self.gpu_ids, True)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A_s, 'G_A_s', which_epoch, self.gpu_ids)
            if self.isTrain:
                self.load_network(self.netG_A, 'G_A', which_epoch, self.gpu_ids, save_path="./checkpoints/lr_net1_11_gammavgg_relu4_3_SELU_2DA_2fea_ls_1g1d_mirror_p2/latest_net_G_A.pth")
                self.load_network(self.netD_A, 'D_A', which_epoch, self.gpu_ids)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch, self.gpu_ids)
                if self.opt.fea == 1:
                    self.load_network(self.netD_fea, 'D_fea', which_epoch, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt.lr
            self.old_lr_D = opt.lr_D
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)


            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A_s.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            if self.opt.fea == 1:
                self.optimizer_D_fea = torch.optim.Adam(self.netD_fea.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A_s)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
            if self.opt.fea == 1:
                networks.print_network(self.netD_fea)
        if opt.isTrain:
            self.netG_A_s.train()
        else:
            self.netG_A_s.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        if self.opt.phase == 'train':
            input_B = input['B']
        input_A = input['A']

        input_img = input['input_img']
        A_img= input['A_img']
        self.A_img.resize_(A_img.size()).copy_(A_img)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        if self.opt.phase == 'train':
            self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']
        self.input_img0 = self.input_img.clone()

    def predict(self):
        with torch.no_grad():
            # import time
            # inp_tensor = torch.rand(1, 3, 1200, 900).cuda()
            # # inp_tensor1 = torch.rand(1, 1, 1200, 900).cuda()  # NOTE: this is the shape for ACDC images
            # import time
            # inp_tensor = torch.rand(1, 3, 1200, 900).cuda()
            # start_time = time.time()
            # _ = self.netG_A_s.forward(inp_tensor)
            # used_time = time.time() - start_time
            # print('per Time (S) = {:.4f}'.format(used_time))
            # from thop import profile
            # # 取出未包装的模型，保持它在 GPU 上
            # model_for_profile = self.netG_A_s.module if isinstance(self.netG_A_s, torch.nn.DataParallel) else self.netG_A_s

            # flops, params = profile(model_for_profile, inputs=(inp_tensor,))
            # G_flops = flops * 1e-9
            # M_params = params * 1e-6
            # # 测速时移回 GPU，用原始调用方式
            # model_for_profile = model_for_profile.to(inp_tensor.device)
            # # # get time
            # start_time = time.time()
            # count = 100
            # for i in range(count):
            #     _ = model_for_profile(inp_tensor)
            # used_time = time.time() - start_time
            # ave_time = used_time / count
            # # print score
            # print('FLOPs (G) = {:.4f}'.format(G_flops))
            # print('Params (M) = {:.4f}'.format(M_params))
            # print('Time (S) = {:.4f}'.format(ave_time))

            vis = False
            self.real_A = Variable(self.input_A)
            self.real_img = Variable(self.input_img)

            if vis:
               self.fake_B, _, _ , maps = self.netG_A_s.forward(self.real_img, vis=vis)
               fake_B = util.tensor2im(self.fake_B.data)
               # maps = util.atten2im(maps.data)
               maps = (maps * 2 - 1)
               maps = util.tensor2im(maps.data)

               return OrderedDict([('fake_B', fake_B), ('maps', maps)])
            else:
                self.fake_B, _, _ = self.netG_A_s.forward(self.real_img, it=1)
                fake_B = util.tensor2im(self.fake_B.data)
                return OrderedDict([('fake_B', fake_B)])
    def Visualize_feature_maps(self):

        self.real_A = Variable(self.input_A)
        self.real_img = Variable(self.input_img)
        model = self.netG_A_s
        # 选择你感兴趣的层（可以是卷积层，激活层等）
        layer_names = ["module.conv0","module.bn2_1","module.bn2_3","module.bn2_2", "module.bn4_1","module.bn4_3","module.bn4_2",
                       "module.bn5_1","module.bn5_3","module.bn5_2","module.bn8_1","module.bn8_3","module.bn8_2"]  # ResNet的层名，可以根据需要选择
        intermediate_outputs = []
        hooks = []

        # 钩子函数：提取中间层的输出
        def hook_fn(module, input, output):
            intermediate_outputs.append(output)

        # 注册钩子以提取特定层的输出
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)


        # 前向传播
        with torch.no_grad():
            self.fake_B,_,_ = self.netG_A_s.forward(self.real_img)

        # 选择要显示的通道
        def select_channels(layer_output, num_channels=16):
            # 计算每个通道的平均激活值
            # layer_output = layer_output.unsqueeze(0)  # 去掉batch维度，得到 [num_channels]
            channel_mean = layer_output.mean(dim=(1, 2))  # [num_channels]


            # 按照响应强度排序，选择前num_channels个通道
            top_channels = torch.topk(channel_mean, num_channels, dim=0).indices
            return top_channels

        # 可视化特征图
        for i, layer_output in enumerate(intermediate_outputs):
            print(f"Layer {layer_names[i]} output shape: {layer_output.shape}")

            # 获取该层的特征图
            feature_maps = layer_output[0]  # [num_channels, height, width]，即选择batch size为1的输出

            # 选择前16个响应最强的通道
            top_channels = select_channels(feature_maps, num_channels=16)

            # 可视化选定的特征图
            plt.figure(figsize=(15, 15))
            for j, channel in enumerate(top_channels):
                name = layer_names[i].split(".")[-1]
                plt.subplot(4, 4, j + 1)
                plt.imshow(feature_maps[channel].cpu().detach().numpy(), cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"feature maps from layer: {name}",fontsize=40)
            # 使用tight_layout()去除留白
            plt.tight_layout(pad=0.5)  # 减小pad来减少间距
            plt.subplots_adjust(hspace=0.005, wspace=0.05)  # 增加行间距和列间距
            # 保存图片到文件
            output_dir = "output_fea_images"

            os.makedirs(output_dir, exist_ok=True)  # 创建保存图片的文件夹（如果不存在）

            # 保存为 PNG 格式
            save_path = os.path.join(output_dir, f"{layer_names[i]}_feature_maps.png")
            plt.savefig(save_path, bbox_inches='tight')  # 使用 bbox_inches='tight' 去除多余空白
            plt.savefig(save_path)
            print(f"Saved feature maps of layer {layer_names[i]} to {save_path}")

            plt.show()

        # 清理钩子
        for hook in hooks:
            hook.remove()
        # real_A = util.tensor2im(self.real_A.data)
        # fake_B = util.tensor2im(self.fake_B.data)
        # return OrderedDict([('fake_B', fake_B)])

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, fake1=None):
        # Real
        pred_real = netD.forward(real.detach())
        pred_fake = netD.forward(fake.detach())
        if fake1 is not None:
            pred_fake1 = netD.forward(fake1.detach())
            loss_D_fake1 = self.criterionGAN(pred_fake1, False)

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        if fake1 is not None:
            loss_D = loss_D_real + 0.5 * loss_D_fake + 0.5 * loss_D_fake1
        else:
            loss_D = loss_D_real + loss_D_fake
        return loss_D

    def backward_D_A(self, epoch, iteration):
        discriminator_id =1
        seed = self.base_seed + epoch * 10000 + iteration * 10 + discriminator_id
        # CPU 随机
        random.seed(seed)

        # CUDA 随机：确保在正确设备上设置
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        # fake_B = self.fake_B
        if self.epoch3 <= epoch:
            fake_B = self.fake_B1
        if self.epoch2 <= epoch < self.epoch3:
            fake_B = self.fake_B2
        if self.epoch1 <= epoch < self.epoch2:
            fake_B = self.fake_B3
        if epoch < self.epoch1:
            fake_B = self.fake_B4
        if self.opt.realBf:
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, None)
        else:
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A.backward()

    def backward_D_fea(self, epoch, iteration):
        discriminator_id = 2
        seed = self.base_seed + epoch * 10000 + iteration * 10 + discriminator_id
        # CPU 随机
        random.seed(seed)
        # CUDA 随机：确保在正确设备上设置
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        # fake_patch_1 = self.fake_patch_1
        if self.epoch3 <= epoch:
            fake_patch_1 = self.fake1_patch_1
        if self.epoch2 <= epoch < self.epoch3:
            fake_patch_1 = self.fake2_patch_1
        if self.epoch1 <= epoch < self.epoch2:
            fake_patch_1 = self.fake3_patch_1
        if epoch < self.epoch1:
            fake_patch_1 = self.fake4_patch_1
        self.loss_D_fea = 0
        for i in range(self.opt.patchD_3):

            fake_B_fea, real_B_fea = self.vgg_loss.vgg_fea(self.vgg, fake_patch_1[i],
                                                                  self.real_patch_1[i])
            if self.opt.realBf:
                # real_Bf_fea = self.vgg_loss.vgg_fea1(self.vgg, self.realf_patch_1[i])

                loss_D_fea = self.backward_D_basic(self.netD_fea, real_B_fea, fake_B_fea,None)
            else:
                loss_D_fea = self.backward_D_basic(self.netD_fea, real_B_fea, fake_B_fea)
            self.loss_D_fea = self.loss_D_fea + loss_D_fea / float(self.opt.patchD_3)
        self.loss_D_fea.backward()

    def backward_D_P(self):
        self.loss_D_P = 0
        # fake_patch_1 = self.fake_patch_1
        if self.opt.input_2:
            c = random.randint(0, 1)
            if c == 0:
                fake_patch_1 = self.fake_patch_1
            elif c == 1:
                fake_patch_1 = self.fake0_patch_1
        else:
            fake_patch_1 = self.fake_patch_1
        # else:
        #     fake_patch_1 = self.realf_patch_1
        for i in range(self.opt.patchD_3):
            if self.opt.realBf:

                loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch_1[i], fake_patch_1[i],self.realf_patch_1[i].detach())
            else:
                loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch_1[i], fake_patch_1[i])
            self.loss_D_P = self.loss_D_P + loss_D_P / float(self.opt.patchD_3)

        self.loss_D_P.backward()


    def pool(self, input):
        # pd = nn.ReflectionPad2d(1)
        # pool = nn.AvgPool2d(kernel_size=3, stride=1)
        # output = pool(pd(input))

        pool = nn.MaxPool2d(kernel_size=2)
        # output = F.interpolate(pool(input), scale_factor=2, mode='bilinear')
        return pool(input)

    def Mask_pool(self, input, rate):
        pool_input = F.interpolate(self.pool(input), scale_factor=2, mode='bilinear')

        output = input * (1 - rate) + pool_input * rate
        # output = self.randMask(output, 0.5)

        return output
    def randMask(self, input, rate):

        h = input.size(2)
        w = input.size(3)
        size = 8
        num = int((w * h / (size * size)))
        pos = random.sample(range(num), k=int(num * rate))
        h_num = int(h // size)
        w_num = int(w // size)
        # mask = torch.ones([input.size(0), 1, h, w]).cuda()
        # pool_input = (input + self.pool(input)) / 2.

        for i in pos:
            x, y = size * int(i // h_num), size * int(i % w_num)
            refer = input[:, :, x, y].clone()
            input[:, :, x: x + size, y: y + size] = refer.unsqueeze(2).unsqueeze(3)
        # input = input * mask
        return input

    def randmove(self,input):
        # h_off, w_off = torch.randint(high=2, low=-1, size=[1]) * padding, torch.randint(high=2, low=-1, size=[1]) * padding
        # pad = nn.ReflectionPad2d(padding=padding)
        # input_pad = pad(input)
        image = input
        b, c, h, w = input.size()
        device = image.device
        num_pixels = h * w
        # shullfe_indices = []
        # for _ in range(b):
        #     channel_indices = []
        #     for _ in range(c):
        #         channel_indices.append(torch.randperm(num_pixels, device=device))
        #         shullfe_indices.append(torch.stack(channel_indices))
        # shullfe_indices = torch.stack(shullfe_indices)
        batch_channel_indices = torch.randperm(num_pixels, device=device).unsqueeze(0).unsqueeze(0)
        shuffle_indices = batch_channel_indices.repeat(b, c, 1)
        shuffle_indices = torch.stack([idx[torch.randperm(c,device=device)] for idx in shuffle_indices])

        images_flat = image.view(b, c, num_pixels)

        shuffle_flat = images_flat.gather(dim=2, index = shuffle_indices)
        shuffled_images = shuffle_flat.view(b, c, h, w)
        return shuffled_images

        # h_offset, w_offset = padding + h_off, padding + w_off
        # output = input_pad[:, :, h_offset:h + h_offset, w_offset:w + w_offset]
        # return output
    def color_deviation(self,input):
        a, b, c = random.uniform(0.1, 1), random.uniform(0.1, 1), random.uniform(0.1, 1)
        output = torch.concat([torch.unsqueeze(a * input[:, 0, :, :], dim=1), torch.unsqueeze(b * input[:, 1, :, :], dim=1),
                               torch.unsqueeze(c * input[:, 2, :, :], dim=1)], dim=1)
        return output
    def addnoise(self, input, min, max):
        device = input.device
        min_std, max_std = min / 255, max / 255.
        std = (torch.rand(1) * (max_std - min_std) + min_std).to(device)
        # noise = torch.rand_like(input.shape,input) * std
        noise = torch.rand(input.shape, device = device) * std
        output = input + noise
        # output = noise.cuda()
        return output
    def randnaddnoise(self, input, min, max):
        device = input.device
        min_std, max_std = min / 255, max / 255.
        std = (torch.rand(1) * (max_std - min_std) + min_std).to(device)
        # noise = torch.randn_like(input) * std
        noise = torch.randn(input.shape, device = device) * std
        output = input + noise
        # output = noise.cuda()
        return output

    def gaussian_blur(self, images, kernel_size=11, sigma=3.0):
        from torch.nn.functional import conv2d
        B, C, H, W = images.shape
        device = images.device

        # 生成高斯核（单通道）
        ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # 归一化
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)  # [C, 1, K, K]

        # 边缘填充后卷积
        pad = kernel_size // 2
        images_pad = F.pad(images, (pad, pad, pad, pad), mode="reflect")
        blurred = conv2d(images_pad, kernel, groups=C, padding=0)  # [B, C, H, W]
        return blurred

    def forward(self, epoch, iteration):
        current_seed = self.base_seed + epoch * 10000 + iteration
        # CPU 随机
        random.seed(current_seed)

        # CUDA 随机：确保在正确设备上设置
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
        torch.manual_seed(current_seed)

        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.A_img = Variable(self.A_img)



        self.input_img = (self.input_img + 1) / 2.

        if self.opt.addcolor:
            if random.random() < 0.5:
                self.input_img = self.color_deviation(self.input_img)

        # if self.opt.niter == 150:
        #     rate = max(1 - epoch / 50 * 0.25, 0)
        # else:
        rate = max(0.75 - epoch / self.epoch1 * 0.25, 0)
        # rate = 0.75
        if self.opt.pool:

            # rate = max((epoch - self.epoch2) / self.epoch2 * 0.75, 0)
            # rate = random.random()
            if rate:
                self.input_img = self.Mask_pool(self.input_img, rate)
                # self.input_img = self.gaussian_blur(self.input_img, sigma=3.0 * rate)
                # self.input_img = self.randMask(self.input_img, rate)

        if self.opt.addnoise:
            # rate = max(0.75 - epoch / self.epoch2 * 0.75, 0)
            if random.random() < 0.5:
                self.input_img = self.randnaddnoise(self.input_img, 0, 5)
            self.input_img = self.addnoise(self.input_img, rate * 50, rate * 50)

        self.input_img = torch.clamp(self.input_img, min=0, max=1)
        self.input_img = self.input_img * 2 - 1


        if self.opt.input_2 :
            with torch.no_grad():
                self.real_img0 = Variable(self.input_img0)
                self.fake_B0 , self.x5_10 ,self.x5_20 = self.netG_A.forward(self.real_img0, vis=False)

        c = random.randint(0, 1)
        if c == 0:
            self.real_img = Variable(self.input_img)
            self.real_img00 = Variable(self.input_img0)
        else:
            self.real_img = Variable(self.input_img0)
            self.real_img00 = Variable(self.input_img)
        self.fake_B1, self.x5_11, self.x5_21 = self.netG_A_s.forward(input = self.real_img, vis=False, it=1)
        if epoch  < self.epoch3:
            self.fake_B2, self.x5_12, self.x5_22 = self.netG_A_s.forward(input = self.fake_B1, vis=False, it=2)
        if epoch  < self.epoch2:
            self.fake_B3, self.x5_13, self.x5_23 = self.netG_A_s.forward(input = self.fake_B2, vis=False, it=3)
        if epoch  < self.epoch1:
            self.fake_B4, self.x5_14, self.x5_24 = self.netG_A_s.forward(input = self.fake_B3, vis=False, it=4)

        self.fake_B10, self.x5_110, self.x5_210 = self.netG_A_s.forward(input = self.real_img00, vis=False, it=1)
        if epoch < self.epoch3:
            self.fake_B20, self.x5_120, self.x5_220 = self.netG_A_s.forward(input = self.fake_B10, vis=False, it=2)
        if epoch < self.epoch2:
            self.fake_B30, self.x5_130, self.x5_230 = self.netG_A_s.forward(input = self.fake_B20, vis=False, it=3)
        if epoch < self.epoch1:
            self.fake_B40, self.x5_140, self.x5_240 = self.netG_A_s.forward(input = self.fake_B30, vis=False, it=4)

        if self.opt.patchD or self.opt.patch_vgg:
            self.real_patch_1 = []
            self.realA_patch_1 = []
            self.fake0_patch_1 = []
            self.fake1_patch_1 = []
            self.fake2_patch_1 = []
            self.fake3_patch_1 = []
            self.fake4_patch_1 = []
            self.A_img_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))

                self.realA_patch_1.append(self.real_A[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                          w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                self.fake0_patch_1.append(self.fake_B0[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                          w_offset_1:w_offset_1 + self.opt.patchSize])
                self.A_img_patch_1.append(self.A_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                                 w_offset_1:w_offset_1 + self.opt.patchSize])
                self.fake1_patch_1.append(self.fake_B1[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                if epoch < self.epoch3:
                    self.fake2_patch_1.append(self.fake_B2[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                              w_offset_1:w_offset_1 + self.opt.patchSize])
                if epoch < self.epoch2:
                    self.fake3_patch_1.append(self.fake_B3[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                              w_offset_1:w_offset_1 + self.opt.patchSize])
                if epoch < self.epoch1:
                    self.fake4_patch_1.append(self.fake_B4[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                              w_offset_1:w_offset_1 + self.opt.patchSize])

    def backward_G(self, epoch):
        self.loss_G_A = 0
        self.loss_mirror = 0
        self.loss_vgg_b = 0
        self.loss_dino = 0
        mirror_reg = 0
        mirror1 = 0
        w0 = 1.5
        if self.opt.colorloss:
            w = 1

            mirror1 = 0
            mirror12 = 0
            if epoch >= self.epoch3:
                w1 = 1
                mirror12 = torch.mean((self.x5_210 - self.x5_21)**2)

                mirror1 = w1 * (torch.mean(torch.abs(self.fake_B1 - self.fake_B0)) + 1 - self.ssim((self.fake_B1+1)/2., (self.fake_B0+1)/2.))
                mirror1 += w1 * (
                            torch.mean(torch.abs(self.fake_B10 - self.fake_B0)) + 1 - self.ssim((self.fake_B10 + 1) / 2.,
                                                                                               (self.fake_B0 + 1) / 2.))
                # mirror1 = w1 * (torch.mean((self.fake_B1 - self.fake_B0) ** 2))
                mirror_reg = 3 * w1 * (self.colorloss.colorloss3(self.fake_B1, self.A_img, self.fake_B0, w0))
            else:
                w1 = 1

                # mirror12 = torch.mean((self.x5_22 - self.x5_21) ** 2) + torch.mean((self.x5_220 - self.x5_210) ** 2)

                mirror_reg = 3 * w1 * (self.colorloss.colorloss3(self.fake_B1, self.A_img, self.A_img, w0))


            if epoch < self.epoch3:
                if epoch >= self.epoch2:
                    w2 = 2
                    w = 1 / 2
                    mirror12 += torch.mean((self.x5_220 - self.x5_22)**2)

                    mirror1 += w2 * (torch.mean(torch.abs(self.fake_B2 - self.fake_B0)) + 1 - self.ssim((self.fake_B2+1)/2., (self.fake_B0+1)/2.))
                    mirror1 += w2 * (torch.mean(torch.abs(self.fake_B20 - self.fake_B0)) + 1 - self.ssim(
                        (self.fake_B20 + 1) / 2., (self.fake_B0 + 1) / 2.))
                    # mirror1 += w2 * (torch.mean((self.fake_B2 - self.fake_B0) ** 2))
                    mirror_reg += 3 * w2 * (self.colorloss.colorloss3(self.fake_B2, self.A_img, self.fake_B0, w0))
                else:
                    w2 = 1
                    # mirror12 += torch.mean((self.x5_23 - self.x5_22) ** 2) + torch.mean((self.x5_230 - self.x5_220) ** 2)

                    mirror_reg += 3 * w2 * (self.colorloss.colorloss3(self.fake_B2, self.A_img, self.A_img, w0))

            if epoch < self.epoch2:
                if epoch >= self.epoch1:
                    w3 = 3
                    w = 1 / 3
                    mirror12 += torch.mean((self.x5_230 - self.x5_23)**2)
                    mirror1 += w3 * (torch.mean(torch.abs(self.fake_B3 - self.fake_B0)) + 1 - self.ssim((self.fake_B3+1)/2., (self.fake_B0+1)/2.))
                    mirror1 += w3 * (torch.mean(torch.abs(self.fake_B30 - self.fake_B0)) + 1 - self.ssim(
                        (self.fake_B30 + 1) / 2., (self.fake_B0 + 1) / 2.))
                    # mirror1 += w3 * (torch.mean((self.fake_B3 - self.fake_B0) ** 2))
                    mirror_reg += 3 * w3 * (self.colorloss.colorloss3(self.fake_B3, self.A_img, self.fake_B0, w0))
                else:
                    w3 = 1
                    # mirror12 += torch.mean((self.x5_24 - self.x5_23) ** 2) + torch.mean((self.x5_240 - self.x5_230) ** 2)

                    mirror_reg += 3 * w3 * (self.colorloss.colorloss3(self.fake_B3, self.A_img, self.A_img, w0))

            if epoch < self.epoch1:
                w4 = 4
                w = 1 / 4
                mirror12 += torch.mean((self.x5_240 - self.x5_24)**2)
                mirror1 +=  w4 * (torch.mean(torch.abs(self.fake_B4 - self.fake_B0)) + 1 - self.ssim((self.fake_B4+1)/2., (self.fake_B0+1)/2.))
                mirror1 +=  w4 * (
                            torch.mean(torch.abs(self.fake_B40 - self.fake_B0)) + 1 - self.ssim((self.fake_B40 + 1) / 2.,
                                                                                               (self.fake_B0 + 1) / 2.))
                # mirror1 += w4 * (torch.mean((self.fake_B4 - self.fake_B0) ** 2))
                mirror_reg += 3 * w4 * (self.colorloss.colorloss3(self.fake_B4, self.A_img, self.fake_B0, w0))


            self.loss_mirror = 12 * w * (mirror_reg +  mirror1 + mirror12)
            progress = epoch / (self.opt.niter + self.opt.niter_decay)
            self.w_DA = 0.5#1.3 + 0.4 * vgg_ratio#torch.abs(4- self.loss_dino.detach() - self.loss_vgg_b.detach())
            self.w_Dp = 0.5#1.3 + 0.4 * vgg_ratio #torch.abs(4 - self.loss_dino.detach() - self.loss_vgg_b.detach())
            self.w_Dfea = 0.5#1.3 + 0.4 * vgg_ratio#torch.abs(4 - self.loss_dino.detach() - self.loss_vgg_b.detach())
            self.w_vgg = 0.5

            if self.epoch3 <= epoch:
                fake_B = self.fake_B1
                fake_patch_1 = self.fake1_patch_1
            if self.epoch2 <= epoch < self.epoch3:
                fake_B = self.fake_B2
                fake_patch_1 = self.fake2_patch_1
            if self.epoch1 <= epoch < self.epoch2:
                fake_B = self.fake_B3
                fake_patch_1 = self.fake3_patch_1
            if epoch < self.epoch1:
                fake_B = self.fake_B4
                fake_patch_1 = self.fake4_patch_1

            pred_fake = self.netD_A.forward(fake_B)

            self.loss_G_g = self.criterionGAN(pred_fake, True)

            self.loss_G_A = self.loss_G_A + self.w_DA * self.loss_G_g
            self.loss_G_fea = 0
            if self.opt.fea:
                for i in range(self.opt.patchD_3):
                    self.fake_B_fea = self.vgg_loss.vgg_fea1(self.vgg, fake_patch_1[i])
                    pred_fake_fea = self.netD_fea.forward(self.fake_B_fea)
                    loss_G_fea = (self.criterionGAN(pred_fake_fea, True))
                    self.loss_G_fea += loss_G_fea / float(self.opt.patchD_3)
                self.loss_G_A = self.loss_G_A + self.w_Dfea * self.loss_G_fea

            loss_G_A = 0

            if self.opt.patchD and self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    pred_fake_patch = self.netD_P.forward(self.fake_patch_1[i])
                    if self.opt.input_2:
                        pred_fake0_patch = self.netD_P.forward(self.fake0_patch_1[i])
                        loss_G_p = (self.criterionGAN(pred_fake_patch, True)) + (self.criterionGAN(pred_fake0_patch, True))
                    else:
                        loss_G_p = (self.criterionGAN(pred_fake_patch, True))
                    loss_G_A = loss_G_p / float(self.opt.patchD_3)
                    self.loss_G_A = self.loss_G_A + self.w_Dfea * loss_G_A

                self.loss_G_A += self.w_Dp * loss_G_A / float(self.opt.patchD_3)
        self.w_vgg =0.2 #原来是0.5，且下面不乘w2,3,4


        if self.opt.vgg > 0:
            
            if epoch < 0:
                vgg_w = 0
            else:
                import math
                # sigmoid: 从 ~0 到 ~1，中心在 0.5
                vgg_ratio = 1 / (1 + math.exp((progress - 0.5) * 6))
                vgg_w = 0.3 + 0.4 * vgg_ratio   # [0.6, 0.35]#self.w_vgg# + mirror1.detach() + mirror_reg.detach()#min(self.w_vgg, 0.2 + (epoch - 100) / 150)
                dino_w =  0.7 - 0.4 * vgg_ratio  # [0.3, 0.55]# + mirror2.detach() + mirror_reg.detach()# min(self.w_vgg, 0.2 + (epoch - 100) / 150)
            if epoch >= self.epoch3:
                target1 = self.fake_B0#self.A_img#self.real_B#
            else:
                target1 = None#self.fake_B2.detach()
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake_B1,
                                                              self.A_img, target1, w1) * self.opt.vgg if self.opt.vgg > 0 else 0

            self.loss_dino = self.dino_loss(self.A_img, self.fake_B0, self.fake_B1, 1) * self.opt.vgg if self.opt.vgg > 0 else 0
            if epoch < self.epoch3:
                if epoch >= self.epoch2:
                    target2 = self.fake_B0#self.A_img#self.real_B#
                else:
                    target2 = None#self.fake_B3.detach()
                self.loss_vgg_b += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake_B2,
                                                                  self.A_img, target2, w2) * self.opt.vgg if self.opt.vgg > 0 else 0
                self.loss_dino += self.dino_loss(self.A_img, self.fake_B0, self.fake_B2, 1) * self.opt.vgg if self.opt.vgg > 0 else 0
            if epoch < self.epoch2:
                if epoch >= self.epoch1:
                    target3 = self.fake_B0#self.A_img#self.real_B#
                else:
                    target3 = None#self.fake_B4.detach()
                self.loss_vgg_b += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake_B3,
                                                                  self.A_img, target3, w3) * self.opt.vgg if self.opt.vgg > 0 else 0

                self.loss_dino += self.dino_loss(self.A_img, self.fake_B0, self.fake_B3, 1) * self.opt.vgg if self.opt.vgg > 0 else 0
            if epoch < self.epoch1:
                target4 = self.fake_B0#self.A_img#self.real_B#
                self.loss_vgg_b += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake_B4,
                                                                  self.A_img, target4, w4) * self.opt.vgg if self.opt.vgg > 0 else 0
                self.loss_dino += self.dino_loss(self.A_img, self.fake_B0, self.fake_B4, 1) * self.opt.vgg if self.opt.vgg > 0 else 0



            loss_vgg_patch = 0
            loss_dino_patch = 0

            for i in range(self.opt.patchD_3):
                if epoch >= self.epoch3:
                    target1 = self.fake0_patch_1[i]

                else:
                    target1 = None#self.fake2_patch_1[i].detach()
                loss_vgg_patch += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake1_patch_1[i],
                                                                  self.A_img_patch_1[i],
                                                                  target1, w1) * self.opt.vgg
            if epoch < self.epoch3:
                for i in range(self.opt.patchD_3):
                    if epoch >= self.epoch2:
                        target2 = self.fake0_patch_1[i]
                    else:
                        target2 = None#self.fake3_patch_1[i].detach()
                    loss_vgg_patch += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake2_patch_1[i],
                                                                      self.A_img_patch_1[i],
                                                                      target2, w2) * self.opt.vgg
            if epoch < self.epoch2:
                for i in range(self.opt.patchD_3):
                    if epoch >= self.epoch1:
                        target3 = self.fake0_patch_1[i]
                    else:
                        target3 = None#self.fake4_patch_1[i].detach()
                    loss_vgg_patch += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake3_patch_1[i],
                                                                      self.A_img_patch_1[i],
                                                                      target3, w3) * self.opt.vgg
            if epoch < self.epoch1:
                for i in range(self.opt.patchD_3):
                    target4 = self.fake0_patch_1[i]
                    loss_vgg_patch += self.vgg_loss.compute_vgg_loss3(self.vgg, self.fake4_patch_1[i],
                                                                      self.A_img_patch_1[i],
                                                                      target4, w4) * self.opt.vgg




            self.loss_vgg_b += loss_vgg_patch / float(self.opt.patchD_3)
            self.loss_vgg_b = self.loss_vgg_b * vgg_w * w0 * w
            # self.loss_dino += loss_dino_patch / float(self.opt.patchD_3)
            self.loss_dino = self.loss_dino * dino_w * w0 * w

        self.loss_G = self.loss_G_A + self.loss_vgg_b + self.loss_dino + self.loss_mirror

        self.loss_G.backward()

    def optimize_parameters(self, epoch, iteration):

        self.forward(epoch, iteration)
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        if self.opt.fea == 1:
            self.set_requires_grad([self.netD_A, self.netD_fea], True)
        if self.opt.patchD:
            self.set_requires_grad([self.netD_A, self.netD_P], True)
        self.optimizer_D_A.zero_grad()
        self.backward_D_A(epoch, iteration)
        self.optimizer_D_A.step()
        if self.opt.patchD:
            self.optimizer_D_P.zero_grad()
            self.backward_D_P(epoch, iteration)
            self.optimizer_D_P.step()
        if self.opt.fea:
            self.optimizer_D_fea.zero_grad()
            self.backward_D_fea(epoch, iteration)
            self.optimizer_D_fea.step()


    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        D_fea = self.loss_D_fea.item() if self.opt.fea else 0  # if self.opt.fea and epoch>200 else 0
        G_A = self.loss_G_A.item()
        loss_mirror = self.loss_mirror.item() if self.opt.colorloss else 0
        vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
        dino = self.loss_dino.item() if self.opt.vgg > 0 else 0
        # if epoch < 100:
        return OrderedDict(
            [('D_A', D_A), ('G_A', G_A), ("vgg", vgg),("dino", dino), ("D_P", D_P), ("D_fea", D_fea), ("loss_mirror", loss_mirror)])




    def get_current_visuals(self, epoch):
        fake_B1 = util.tensor2im(self.fake_B1.data)
        input_img = util.tensor2im(self.input_img.data)
        input_img0 = util.tensor2im(self.input_img0.data)
        fake_B0 = util.tensor2im(self.fake_B0.data)
        if epoch < self.epoch3:
            fake_B2 = util.tensor2im(self.fake_B2.data)
        if epoch < self.epoch2:
            fake_B3 = util.tensor2im(self.fake_B3.data)
        if epoch < self.epoch1:
            fake_B4 = util.tensor2im(self.fake_B4.data)

        if self.epoch3 <= epoch <= 800:
            return OrderedDict(
                [('fake_B1', fake_B1),  ('fake_B0', fake_B0), ('input_img', input_img), ('input_img0', input_img0)])
        if self.epoch2 <= epoch < self.epoch3:
            return OrderedDict(
                [('fake_B2', fake_B2),('fake_B1', fake_B1), ('fake_B0', fake_B0), ('input_img', input_img), ('input_img0', input_img0)])
        if self.epoch1 <= epoch < self.epoch2:
            return OrderedDict(
                [('fake_B3', fake_B3),('fake_B2', fake_B2),('fake_B1', fake_B1), ('fake_B0', fake_B0), ('input_img', input_img), ('input_img0', input_img0)])
        if 0 <= epoch < self.epoch1:
            return OrderedDict(
                [('fake_B4', fake_B4),('fake_B3', fake_B3),('fake_B2', fake_B2),('fake_B1', fake_B1), ('fake_B0', fake_B0), ('input_img', input_img), ('input_img0', input_img0)])


    def save(self, label):
        # self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_A_s, 'G_A_s', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        # self.save_network(self.dino_loss.mapping, 'Dino_map', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)
        if self.opt.fea:
            self.save_network(self.netD_fea, 'D_fea', label, self.gpu_ids)

    def update_learning_rate(self):

        if self.opt.new_lr:
            lr = self.old_lr / 2
            lr_D = self.old_lr_D / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd

            lrd_D = self.opt.lr_D / self.opt.niter_decay
            lr_D = self.old_lr_D - lrd_D
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr_D
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr_D
        if self.opt.fea:
            for param_group in self.optimizer_D_fea.param_groups:
                param_group['lr'] = lr_D

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
