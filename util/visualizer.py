import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from torch.utils.tensorboard import SummaryWriter  # 新增 TensorBoard 支持


class Visualizer():
    def __init__(self, opt):
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port  # 保留端口信息但不再使用

        # 初始化 TensorBoard
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard')
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('Creating web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # 显示/保存当前结果
    def display_current_results(self, visuals, epoch):
        # 保存图像到 HTML 目录
        if self.use_html:
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # 更新网站
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, max(0, epoch - 5), -1):  # 只显示最近5个epoch
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, _ in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    if os.path.exists(os.path.join(self.img_dir, img_path)):
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)

                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

        # 添加图像到 TensorBoard
        for label, image_numpy in visuals.items():
            # 转换图像格式 (H,W,C) -> (C,H,W) 并归一化
            img_tensor = np.transpose(image_numpy, (2, 0, 1))
            img_tensor = img_tensor.astype(np.float32) / 255.0

            # 添加到 TensorBoard
            self.writer.add_image(f'{label}/epoch_{epoch}', img_tensor, epoch, dataformats='CHW')

    # 绘制当前错误
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        # 添加标量数据到 TensorBoard
        for tag, value in errors.items():
            self.writer.add_scalar(f'Loss/{tag}', value, epoch + counter_ratio)

    # 打印当前错误
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # 保存图像到磁盘
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            if label == "fake_B":
                image_name = '%s.png' % name
            else:
                image_name = '%s_%s.png' % (name, label)

            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)

    def save_images_demo(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            image_name = '%s.jpg' % name
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)

    def close(self):
        self.writer.close()