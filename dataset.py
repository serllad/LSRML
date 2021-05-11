"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image
import random
import SimpleITK as sitk
import glob
import os
import cv2
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision.transforms as transforms
from utils import _get_coutour_sample
transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]
)


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    'HueSaturationValue:色相饱和度值'
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)  # 由四个点就可以得到变化矩阵
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


class Datafor_metalearning():
    def __init__(self, root, site_list, is_train=True):
        self.num_site=len(site_list)
        self.dataloader = {}
        self.site_list=site_list
        for key in site_list:
            Dataset=Data(root,key,site_list,is_train)
            self.dataloader[key]=get_generator(torch.utils.data.DataLoader(

            Dataset,

            batch_size=2,

            shuffle=True,

            num_workers=4))


    def __getitem__(self, item):
        s_list=np.random.permutation(self.site_list)
        meta_train=s_list[:2]
        meta_test=s_list[-1:]
        #print(meta_train,meta_test)
        imgs_train=[]
        masks_train=[]
        domain=[]
        for i in meta_train:
            imgs,masks,d=next(self.dataloader[i])
            imgs_train.append(imgs)
            masks_train.append(masks)
            domain.append(d)

        imgs_test = []
        masks_test = []
        for i in meta_test:
            imgs,masks,_=next(self.dataloader[i])
            imgs_test.append(imgs)
            masks_test.append(masks)

        # imgs_test,masks_test = next(self.dataloader[meta_test[0]])
        imgs_train=torch.cat(imgs_train,dim=0)
        masks_train=torch.cat(masks_train,dim=0)
        imgs_test = torch.cat(imgs_test, dim=0)
        masks_test = torch.cat(masks_test, dim=0)
        domain=torch.cat(domain)
        return imgs_train,masks_train,imgs_test,masks_test,domain

    # def __len__(self):
    #     return min([len(self.dataloader(x) for x in self.dataloader])

def get_generator(a):
    for img,mask,d in a:
        yield img,mask,d
def RandomIntensityChange(img, shift=0.1, scale=0.1, u=0.5):
    if np.random.random()<u:
        # B,H,W,C
        shift_factor = np.random.uniform(-shift, shift, size=[1, 1, 1])  # [-0.1,+0.1]
        scale_factor = np.random.uniform(1.0 - scale, 1.0 + scale, size=[1, 1, 1])  # [0.9,1.1)
        # shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,1,1,img.shape[3],img.shape[4]]) # [-0.1,+0.1]
        # scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,1,1,img.shape[3],img.shape[4]]) # [0.9,1.1)
        return img * scale_factor + shift_factor
    else:
        return img
import math
def randomSharp(img,u=0.5):
    if np.random.random() < u:
        blurred_strength=0.25+1.25*np.random.random()
        blurred = cv2.GaussianBlur(img, (5, 5), blurred_strength)
        strength = math.floor(np.random.random() * 20 + 10)

        img=img.astype(np.int)
        blurred=blurred.astype(np.int)

        out = strength/100.0 * np.fabs(blurred - img) + blurred
        return out
    else:
        return img
def random_gamma_correct(image, u=0.5):

    if np.random.random()<u:
        gamma=np.random.random()*1+0.2

        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    else:
        return image
class Data(data.Dataset):

    def __init__(self, root, site,site_list, is_train=True):
        # self.transforms = eval(transforms or 'Identity()')


        all_patient = glob.glob(root + '/imgs/%s/*' % site)
        all_patient.sort()
        self.imgs=[]
        self.labels = []
        for i in all_patient:
            self.imgs += glob.glob(os.path.join(i, '*'))

        for i in self.imgs:
            i = i.replace('imgs', 'masks')
            self.labels.append(i)
        self.imgs.sort()
        self.labels.sort()
        self.istrain = is_train
        self.d = {j: i for i, j in enumerate(site_list)}
    def __getitem__(self, index):
        img_path = self.imgs[index]
        domain=self.get_domain(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256)).astype(np.uint8)

        mask_path = self.labels[index]
        mask = np.array(Image.open(mask_path))

        mask = cv2.resize(mask, (256, 256))
        mask = mask.astype(np.uint8)

        if self.istrain:
            # img = randomHueSaturationValue(img,
            #                                hue_shift_limit=(-30, 30),
            #                                sat_shift_limit=(-5, 5),
            #                                val_shift_limit=(-15, 15))
            #img=randomSharp(img)
            #img=random_gamma_correct(img)
            # print(img.max(),img.min())
            img=random_gamma_correct(img)
            img=randomSharp(img)
            img, mask = randomShiftScaleRotate(img, mask,
                                               shift_limit=(-0.1, 0.1),
                                               scale_limit=(-0.2, 0.2),
                                               aspect_limit=(-0.1, 0.1),rotate_limit=(-0, 0))
            img, mask = randomHorizontalFlip(img, mask)
            #img=random_gamma_correct(img)
            img = img.transpose(2, 0, 1).astype(np.float32)
            #img=randomSharp(img).astype(np.float32)
        else:
            img = img.transpose(2, 0, 1).astype(np.float32)
        # img, mask = randomVerticleFlip(img, mask)
        # img, mask = randomRotate90(img, mask)
        # img = img.transpose(2, 0, 1).astype(np.float32)

        mask = np.expand_dims(mask, axis=2)
        # img = transform(img)
        mean=img.mean()
        std=img.std()
        img=(img-mean)/std
        # img=RandomIntensityChange(img).astype(np.float32)
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0

        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        mask_onehot = torch.zeros((2, mask.shape[1], mask.shape[2]))
        mask_onehot = mask_onehot.scatter_(0, torch.from_numpy(mask).long(), 1)
        # mask = torch.from_numpy(mask)
        if domain is None:
            domain=-1
        #print(img_path,domain)
        return img, mask_onehot,domain
    def get_domain(self,img_path):
        domain=img_path.split('/')[3]
        domain_id=self.d.get(domain)
        return domain_id
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    root='D:/DataSets'
    sites=['HK','UCL','I2CVB','ISBI','ISBI_1.5','Task05_Prostate']
    for epoch in range(100):
        train_generator = Datafor_metalearning(root,sites)
        # print(len)
        length=0
        for idx,i in enumerate(train_generator):
            #
            length+=1
        print(length)
        # print(idx)
