# from concurrent.futures import process


import numpy as np
import cv2
import argparse
import yaml
import torch
import torch.nn as nn


from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import functools
import cropper
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from util import tensor2im
from utils.functions import cv_draw_landmark, cv_draw_landmark_only, get_suffix, crop_img, tensor2im, inpaint_img, color_transfer
from models.stgan import Generator



def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    print(cfg)
    gpu_mode = args.mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

    align_crop = cropper.align_crop_opencv

    # face_factor = 0.5
    face_factor = 0.5
    align_type = 'similarity'  # choices=['affine', 'similarity']
    order = 3 # 'The order of interpolation.' choices=[0, 1, 2, 3, 4, 5]
    mode = 'edge' # ['constant', 'edge', 'symmetric', 'reflect', 'wrap']
    attrs = ["Bangs", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open", "Mustache", "Pale_Skin", "Young"]
    # input_label = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    # input_label = torch.eye(len(attrs))
    input_label = torch.Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    g_conv_dim = 64
    g_layers = 5
    shortcut_layers = 3
    use_stu = True
    one_more_conv = True
    image_size = 128
    crop_size = 178
    align_crop = cropper.align_crop_opencv
    trans = transforms.Compose([
        transforms.CenterCrop([crop_size, crop_size]),
        transforms.Resize([image_size, image_size], Image.BICUBIC),
        # transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    netG = Generator(len(attrs), g_conv_dim, g_layers, shortcut_layers, use_stu=use_stu, one_more_conv=one_more_conv)

    G_checkpoint = torch.load("weights/stgan/G_60000.pth.tar", map_location=torch.device('cpu'))
    G_to_load = {k.replace('module.', ''): v for k, v in G_checkpoint['state_dict'].items()}
    netG.load_state_dict(G_to_load)
    netG.eval()

    standard_landmark = np.genfromtxt("standard_landmark_68pts.txt", dtype=np.float).reshape(68, 2)

    # define a video capture object
    vid = cv2.VideoCapture(0)
    # vid = cv2.VideoCapture('/Users/xingdi/netmind-face/3DDFA_V2/cropped_align_celeba_0.5/%06d.jpg')
    # vid = cv2.VideoCapture('/Users/xingdi/Desktop/Obama.webm')
    # vid = cv2.VideoCapture('/Users/xingdi/Desktop/webcam_xing_0214_noglasses.mov')
    fc = 0

    dense_flag = args.opt in ('3d',)
    pre_ver = None
    ver = None


    while(True):


        
        # print(fc)
        # Capture the video frame
        # by frame
        ret, img = vid.read()

        # print(np.any(img))
        # if np.any(img):
        #     continue

        frame_bgr = img.copy()


        if fc == 0:
            boxes = face_boxes(frame_bgr)
            if boxes:
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                ver, ld = tddfa.recon_vers_ld(param_lst, roi_box_lst, dense_flag=dense_flag)
                ver = ver[0]
                ld = ld[0]

                # refine
                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver,ld = tddfa.recon_vers_ld(param_lst, roi_box_lst, dense_flag=dense_flag)
                ver = ver[0]
                ld = ld[0]
            else:
                continue


        else:

            if not ver.any():

                param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

                roi_box = roi_box_lst[0]
                # todo: add confidence threshold to judge the tracking is failed
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame_bgr)
                    if boxes:
                        boxes = [boxes[0]]
                        # print(boxes)
                        param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

                ver,ld = tddfa.recon_vers_ld(param_lst, roi_box_lst, dense_flag=dense_flag)
                ver = ver[0]
                ld = ld[0]
            else:
                boxes = face_boxes(frame_bgr)

                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                ver, ld = tddfa.recon_vers_ld(param_lst, roi_box_lst, dense_flag=dense_flag)
                ver = ver[0]
                ld = ld[0]

                # refine
                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver,ld = tddfa.recon_vers_ld(param_lst, roi_box_lst, dense_flag=dense_flag)
                ver = ver[0]
                ld = ld[0]

        pre_ver = ver


            # print(ver.shape)
            # print(ld.shape)

        # if args.opt == '2d_sparse_only':
        #     res = cv_draw_landmark_only(frame_bgr, ld)
        # elif args.opt == '2d_sparse':
        #     res = cv_draw_landmark(frame_bgr, ver)
        # elif args.opt == '3d':
        #     res = render(frame_bgr, [ver], tddfa.tri, alpha=1.0, with_bg_flag=False)
        # else:
        #     raise ValueError(f'Unknown opt {args.opt}')



        src_landmarks = ld[:2,:]

        src_landmarks = np.stack([src_landmarks[0,:], src_landmarks[1,:]],1) 

        cropped_img, tformed_landmarks = align_crop(frame_bgr,
                                                 src_landmarks,
                                                 standard_landmark,
                                                 crop_size=(256, 256),
                                                 face_factor=face_factor,
                                                 align_type=align_type,
                                                 order=order,
                                                 mode=mode)

        
        img_id = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # print(img_id.shape)
        img_id = trans(Image.fromarray(img_id)).unsqueeze(0)

        # print(img_id.shape)
        img_o  = netG(img_id.repeat((input_label.shape[0], 1,1,1 )), input_label)

        # img_o = tensor2ims(img_o)
        # print(torch.cat(img_o,img_id), 0))
        img_o = make_grid(torch.cat((img_id, img_o), 0), nrow = len(attrs) + 1, normalize=False)
        img_o = img_o.numpy()
        # print(img_o.shape)
        # print(np.max(img_o))

        img_o = (np.transpose(img_o, (1, 2, 0)) + 1) / 2.0 * 255.0
        img_o = img_o.astype(np.uint8)
        # print(np.max(img_o))
        # print(img_o.shape)
        # print(np.max(img_o))

        img_o = cv2.cvtColor(img_o, cv2.COLOR_RGB2BGR)


        # # Display the resulting frame 
        cv2.imshow("change", img_o)

        fc += 1


      
        k = cv2.waitKey(1)
        if (k & 0xff == ord('q')):
            break
      
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--folder_dir', type=str)
    parser.add_argument('-w', '--writer_dir', default='./cropped_imgs/', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='3d', choices=['2d_sparse', '2d_sparse_only', '3d'])

    args = parser.parse_args()
    main(args)