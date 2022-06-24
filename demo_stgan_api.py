import numpy as np
import cv2
import yaml
import torch
from PIL import Image
from torchvision import transforms
import cropper
from torchvision.utils import make_grid
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from models.stgan import Generator
import time


class FacialAttributeEditing:

    def __init__(self):
        self.opt = '3d'
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        gpu_mode = 'cpu' == 'gpu'
        self.face_boxes = FaceBoxes()
        self.tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        self.face_factor = 0.5
        self.align_type = 'similarity'  # choices=['affine', 'similarity']
        self.order = 3  # 'The order of interpolation.' choices=[0, 1, 2, 3, 4, 5]
        self.mode = 'edge'  # ['constant', 'edge', 'symmetric', 'reflect', 'wrap']
        self.attrs = ["Bangs", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Male",
                      "Mouth_Slightly_Open",
                      "Mustache", "Pale_Skin", "Young"]
        self.input_label = torch.Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        self.align_crop = cropper.align_crop_opencv
        self.trans = transforms.Compose([
            transforms.CenterCrop([crop_size, crop_size]),
            transforms.Resize([image_size, image_size], Image.BICUBIC),
            # transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.netG = Generator(len(self.attrs), g_conv_dim, g_layers, shortcut_layers, use_stu=use_stu,
                              one_more_conv=one_more_conv)
        G_checkpoint = torch.load("weights/stgan/G_60000.pth.tar", map_location=torch.device('cpu'))
        G_to_load = {k.replace('module.', ''): v for k, v in G_checkpoint['state_dict'].items()}
        self.netG.load_state_dict(G_to_load)
        self.netG.eval()
        self.standard_landmark = np.genfromtxt("standard_landmark_68pts.txt", dtype=np.float).reshape(68, 2)

    def edit(self, img_path, output_path):
        path = img_path
        dense_flag = self.opt in ('3d',)
        img = cv2.imread(path)
        frame_bgr = img.copy()

        boxes = self.face_boxes(frame_bgr)
        if boxes:
            boxes = [boxes[0]]
            param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)
            ver, ld = self.tddfa.recon_vers_ld(param_lst, roi_box_lst, dense_flag=dense_flag)
            ver = ver[0]
            # refine
            param_lst, roi_box_lst = self.tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver, ld = self.tddfa.recon_vers_ld(param_lst, roi_box_lst, dense_flag=dense_flag)
            ver = ver[0]
            ld = ld[0]
        else:
            return False

        src_landmarks = ld[:2, :]

        src_landmarks = np.stack([src_landmarks[0, :], src_landmarks[1, :]], 1)

        cropped_img, tformed_landmarks = self.align_crop(frame_bgr,
                                                         src_landmarks,
                                                         self.standard_landmark,
                                                         crop_size=(256, 256),
                                                         face_factor=self.face_factor,
                                                         align_type=self.align_type,
                                                         order=self.order,
                                                         mode=self.mode)
        # 原图的脸部
        # cv2.imwrite('results0.jpg', cropped_img)
        img_id = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        img_id = self.trans(Image.fromarray(img_id)).unsqueeze(0)

        img_o = self.netG(img_id.repeat((self.input_label.shape[0], 1, 1, 1)), self.input_label)
        img_o = make_grid(torch.cat((img_id, img_o), 0), nrow=len(self.attrs) + 1, normalize=False)
        img_o = img_o.numpy()

        img_o = (np.transpose(img_o, (1, 2, 0)) + 1) / 2.0 * 255.0
        img_o = img_o.astype(np.uint8)

        img_o = cv2.cvtColor(img_o, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, img_o)
        return True


facial_attribute_editing = FacialAttributeEditing()

if __name__ == '__main__':
    t2 = time.time()
    path = '/Users/renboyan/Documents/m.jpeg'
    output_path = '/Users/renboyan/Documents/m-edit.jpeg'
    facial_attribute_editing.edit(path, output_path)
    print(time.time() - t2)
    print()
    t3 = time.time()
    path = '/Users/renboyan/Documents/g.png'
    output_path = '/Users/renboyan/Documents/g-edit.png'
    facial_attribute_editing.edit(path, output_path)
    print(time.time() - t3)
    print()
