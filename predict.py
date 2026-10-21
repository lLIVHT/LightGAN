import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html


def set_parse(opt):
    opt.dataroot = 'dataset/test/testA'
    opt.results_dir = 'results'
    # opt.dataroot = '/mnt/sdb/hap/test_datasets/ve_lol/VE-LOL-L-Cap-Low_test/low'
    # opt.results_dir = 've_lol'
    # opt.dataroot = '/mnt/sdb/hap/test_datasets/NPE' #MEF,DICM,LIME,VV,NPE
    # opt.results_dir = 'NPE'
    opt.name = 'lightweight'
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.resize_or_crop='no'
    opt.grad_attention = 0
    opt.single=20260712
    opt.gf =0
    opt.BSN = 0
    opt.mirrorc = 0
    opt.mirrorb = 0
    opt.grad = 0
    opt.flag = 0
    opt.net = 'networks'
    opt.niter = 100
    opt.niter_decay = 100
opt = TestOptions().parse()
set_parse(opt)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join("./ablation/", opt.name, '%s' % (opt.results_dir))
webpage = html.HTML(web_dir, 'Experiment = %s, dir = %s' % (opt.name, opt.results_dir))
# test
if __name__=='__main__':
    print(len(dataset))
    for i, data in enumerate(dataset):
        model.set_input(data)
        visuals = model.predict()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()