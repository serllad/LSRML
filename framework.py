import torch
import torch.nn as nn
from torch.autograd import Variable as V
import cv2
import math
import numpy as np
from loss import loss_VAE
from utils import _get_compactness_cost
from cenet import Domain_Discr,VAE
def alpha(x,max_iter):
    if x>=max_iter:
        return 1
    else:
        return x/max_iter
class MyFrame():
    def __init__(self, net, loss, inner_lr=2e-4,d_lr=1e-3,clip_value=10, lmbda=1,evalmode=False,mylog=None):
        self.net = net().cuda()
        self.Discriminator=Domain_Discr().cuda()
        self.vae=VAE(in_channels=516).cuda()

        # self.net = net()
        self.inner_lr=inner_lr
        self.outer_lr=d_lr
        self.mylog=mylog
        c = 0
        for i in self.net.parameters():
            c += i.numel()

        print('parameters:', c)

        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        # self.get_optimizer(lr)

        #self.inner_optimizer = torch.optim.SGD(params=self.net.parameters(), lr=inner_lr,momentum=0.9)
        self.inner_optimizer = torch.optim.Adam(params=list(self.net.parameters())[96:]+list(self.vae.parameters()), lr=inner_lr,weight_decay=1e-8)
        # self.outer_optimizer = torch.optim.Adam(params=self.net.parameters(), lr=outer_lr)
        self.d_optimizer=torch.optim.Adam(params=list(self.Discriminator.parameters()), lr=d_lr,weight_decay=1e-8)

        # self.optimizer=adabound.AdaBound(params=self.net.parameters(),lr=lr,final_lr=0.01)
        self.loss = loss()
        self.lmbda=lmbda
        self.clip_value=clip_value
        self.old_lr = inner_lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
    # def get_optimizer(self,lr):
    #     self.encoder_optimizer=torch.optim.Adam(params=list(self.net.parameters())[:108],lr=lr/4.0)
    #     self.decoder_optimizer = torch.optim.Adam(params=list(self.net.parameters())[108:], lr=lr, weight_decay=1e-7)
    def set_input(self, img, mask,test_img,test_mask,target_img,target_mask,domain_id=None, img_id=None):
        self.meta_train_img = img
        self.meta_train_mask = mask
        self.meta_test_img = test_img
        self.meta_test_mask = test_mask
        self.domain_id = domain_id
        self.target_img = target_img
        self.target_mask = target_mask
        # self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.tocuda(volatile=True)
        mask = self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        return mask

    def tocuda(self, volatile=False):
        'volatile属性默认为False，如果设为True，所有依赖它的节点volatile属性都为True。volatile属性为True的节点不会求导，优先级比requires_grad高。'
        self.meta_train_img = V(self.meta_train_img.cuda(),requires_grad=True ,volatile=volatile)
        self.meta_train_mask = V(self.meta_train_mask.cuda(),requires_grad=True, volatile=volatile)
        self.meta_test_img = V(self.meta_test_img.cuda(),requires_grad=True, volatile=volatile)
        self.meta_test_mask = V(self.meta_test_mask.cuda(),requires_grad=True, volatile=volatile)
        self.domain_id = self.domain_id.cuda()
        self.target_mask = V(self.target_mask.cuda(), requires_grad=True, volatile=volatile)
        self.target_img = V(self.target_img.cuda(), requires_grad=True, volatile=volatile)

    def get_lambda(self, input, max_iteration):
        return 2.0 / (1 + math.exp(-10 * alpha(input, max_iteration))) - 1
    def optimize(self,iteration,max_iteration):
        self.tocuda()

        pred_meta_train,fea = self.net.forward(self.meta_train_img)
        alpha = self.get_lambda(iteration, max_iteration)
        #compact_loss, _, _, _ = _get_compactness_cost(pred_meta_train, self.meta_train_mask)
        pred_domain = self.Discriminator(fea, alpha)
        ce_loss = nn.CrossEntropyLoss()(pred_domain, self.domain_id)
        loss =self.loss(pred_meta_train,self.meta_train_mask)+ce_loss
        grad = torch.autograd.grad(loss, list(self.net.parameters())[96:]+list(self.Discriminator.parameters()))
        #[clip_grad_norm(x,max_norm=self.clip_value) for x in list(self.net.parameters())[96:]]

        # print(',,/.......................',grad)

        fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, list(self.net.parameters())[96:])))
        fast_weights=list(self.net.parameters())[:96]+fast_weights

        # pred_meta_test=self.net.forward_ce_net(self.meta_test_img,fast_weights)

        # compact_loss,_,_,_=_get_compactness_cost(pred_meta_test,self.meta_test_mask)
        reconstruct_loss, pred_meta_test = self.construct_loss(fast_weights,iteration)
        loss_meta_test=self.lmbda*(self.loss(pred_meta_test,self.meta_test_mask)+reconstruct_loss)

        self.inner_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        loss_meta_test.backward(retain_graph=True)
        #[clip_grad_norm(x, max_norm=self.clip_value) for x in list(self.net.parameters())[96:]]
        
        # clip_grad_norm(list(self.net.parameters())[96:], max_norm=self.clip_value)
        pred_meta_train,fea = self.net.forward(self.meta_train_img)

        pred_domain = self.Discriminator(fea, alpha)
        ce_loss = nn.CrossEntropyLoss()(pred_domain, self.domain_id)
        seg_train_loss=self.loss(pred_meta_train, self.meta_train_mask)
        #compact_loss, _, _, _ = _get_compactness_cost(pred_meta_train, self.meta_train_mask)
        loss = seg_train_loss+ce_loss
        loss.backward()
        self.inner_optimizer.step()
        self.d_optimizer.step()
        return seg_train_loss.data,ce_loss.data,loss_meta_test.data
    def construct_loss(self,fast_weights,iteration):
        fea, pred_meta_test = self.net.forward_ce_net(self.meta_test_img, fast_weights)
        # pred_img, mean, logvar = self.vae(fea)
        # pred_meta_test_detach=pred_meta_test.detach()
        # construct_gt = self.get_construct_gt(self.meta_test_img, pred_meta_test_detach)
        # reconstruct_loss_meta_test = loss_VAE(self.mylog, construct_gt.shape, mean, logvar, construct_gt, pred_img,
        #                             pred_meta_test_detach, 0.1, 0.1)
        fea2,pred_target=self.net.forward_ce_net(self.target_img, fast_weights)
        pred_target=pred_target.detach()
        pred_img_target, mean_target, logvar_target = self.vae(fea2)
        construct_gt_target = self.get_construct_gt(self.target_img, pred_target)
        reconstruct_loss_target = loss_VAE(self.mylog, construct_gt_target.shape, mean_target, logvar_target,
                                           construct_gt_target, pred_img_target,
                                            pred_target, 0.1, 0.1)
        return reconstruct_loss_target,pred_meta_test
    def get_construct_gt(self,img,mask):#img.shape(B,3,256,256),mask.shape(B,2,256,256)
        img=img[:,0,:,:].unsqueeze(1)
        return torch.cat([img,img],1)*mask
    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.inner_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = new_lr
        # for param_group in self.encoder_optimizer.param_groups:
        #     param_group['lr'] = new_lr/4.0
        # print(mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
