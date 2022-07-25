import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.image as mpimg
import argparse
import math
import os
import random
import time
from multiprocessing import cpu_count

import GPUtil
import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from llm import hybridModel
# from utils.network_4conv_CA_SE_CA_copy import hybridModel
from sklearn.utils import shuffle
from timm import create_model as creat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_grad_cam import GradCAM
print('CPU核的数量：', cpu_count())
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

transform = transforms.Compose([transforms.ToTensor()])



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    # model = models.resnet50(pretrained=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpus_num = torch.cuda.device_count()
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    model = hybridModel().to(device)

    model = nn.DataParallel(model)
    model = model.cuda()
    
    pth = '/home/Newdisk/zhangxiuwei/sunyi/dfnet/model/model_nir/pretain_model/CA_SE_CA_model_nir_80_0.7490.pkl'

    
    model.load_state_dict(torch.load(pth))
    if isinstance(model,torch.nn.DataParallel):
	    model = model.module
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    target_layers = [model.conv_layer1]

    imageRgbPath = '/home/Newdisk/zhangxiuwei/sunyi/nirscene/country/set/train_dataset_5/co/000010_co-3403.jpg'
    imageIrPath = '/home/Newdisk/zhangxiuwei/sunyi/nirscene/country/set/train_dataset_5/ir/000010_ir-8105.jpg'
    img_co = mpimg.imread(imageRgbPath)
    img_ir = mpimg.imread(imageIrPath)

    img_tmp = np.float32(img_co) / 255

    label = 1
    img_co = transform(img_co)
    img_ir = transform(img_ir)
 
    

    img1 = img_co.float().unsqueeze(0)  # 32,1,64,64
    img2 = img_ir.float().unsqueeze(0)
    # label = label.squeeze().float()  # 32,1
    # print(img1.shape)
    input_tensor = torch.cat((img1,img2),0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)
    # label = label.to(device)  # 32

    # rgb_img = np.float32(rgb_img) / 255
    

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)
   
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        img1 = cv2.cvtColor(img_tmp,cv2.COLOR_GRAY2RGB)
    
        cam_image = show_cam_on_image(img1, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor,target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_gb.jpg', gb)
    cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)


