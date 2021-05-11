import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from util import GetBoundary

import cv2
import numpy as np
from torch.nn.modules.loss import _Loss
def loss_VAE(mylog,input_shape, z_mean, z_log_var,y_true,y_pred,mask, weight_L2=0.1, weight_KL=0.1):
    # y_pred=y_pred*mask
    B,C, H, W = input_shape
    n = C * H * W
    loss_L2 = torch.mean(torch.square(y_true - y_pred), dim=(2, 3))
    fore_ = torch.sum(mask[:, 1])
    back_ = torch.sum(mask[:, 0])
    weights = torch.cat([(back_ / (fore_ + back_)).unsqueeze(1), (fore_ / (fore_ + back_)).unsqueeze(1)],dim=-1).cuda()

    loss_L2 = torch.sum(loss_L2 * weights, dim=-1)
    # loss_L2 = K.mean(K.square(y_true - y_pred), axis=(1, 2, 3))  # original axis value is (1,2,3).
    # loss_L2=torch.mean(torch.square(y_true-y_pred),dim=(1,2,3))
    # weight_
    # loss_L2 = torch.square(y_true - y_pred)
    # loss_KL = (1 / n) * K.sum(
    #     K.exp(z_var) + K.square(z_mean) - 1. - z_var,
    #     axis=-1
    # )
    loss_KL=(1/n)*torch.sum(torch.exp(z_log_var)+torch.square(z_mean)-1-z_log_var,dim=-1)
    print('loss_KL:%s---loss_L2:%s'%(loss_KL,loss_L2),file=mylog)
    return torch.mean(weight_L2 * loss_L2 + weight_KL * loss_KL)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, target,pred):

        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1 - pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha

        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        

        return loss

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=1, weight_map=None, eps=1e-8):
        dice_loss = soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map)
        return dice_loss


def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def soft_dice_loss(prediction, soft_ground_truth, num_class=1, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
    # dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss
    dice_score = torch.mean(-torch.log(dice_score))
    return dice_score

class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):

        y_ce_true = y_true.squeeze(dim=1).long()


        a = self.ce_loss(y_pred, y_ce_true)

        return a
class multi_class_dice_loss(nn.Module):
    def __init__(self):
        super(multi_class_dice_loss, self).__init__()

    def forward(self,prediction, gt, num_class=2):
        predict = prediction.permute(0, 2, 3, 1)
        pred = predict.contiguous().view(-1, num_class)
        # pred = F.softmax(pred, dim=1)
        soft_ground_truth=gt.permute(0, 2, 3, 1)
        b,x,y,c=soft_ground_truth.shape
        ground = soft_ground_truth.reshape((b*x*y,c))

        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
        dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
        dice_loss = 1.0 - torch.mean(dice_score)
        return dice_loss
        # dice_score = torch.mean(dice_score)
        # return dice_score

class dice_loss2(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss2, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b



def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        return a


import torch
import torch.nn as nn

class DiceLoss(Function):

    def __init__(self, *args, **kwargs):
        # print('args',args)
        pass

    def forward(self, predict, target):
        eps = 0.00001
        # target = target.squeeze(axis=1)
        #b:batch_size, z:depth, y:height w:width
        b, z, y, x = predict.shape
        predict=predict.view(b,z,-1)
        target_ = target.view(b, -1)
        # target_=torch.dot(target,target_)
        result_ = predict.argmax(1)# dim 2 is of length 2. Reduce the length to 1 and label it with the class with highest probability

        result = result_.cuda().float()
        target = target_.cuda().float()
        self.save_for_backward(result, target)

        self.intersect = torch.zeros(predict.shape[0]).cuda()
        self.union = torch.zeros(predict.shape[0]).cuda()
        dice = torch.zeros(predict.shape[0]).cuda()

        for i in range(predict.shape[0]):
            self.intersect[i] = torch.dot(result[i, :], target[i, :])
            result_sum = torch.dot(result[i, :])
            target_sum = torch.dot(target[i, :])
            self.union[i] = result_sum + target_sum

            dice[i] = 2 * self.intersect[i] / (self.union[i] + eps)
            # print('union: {}\t intersect: {}\t dice_coefficient: {:.7f}'.format(str(self.union[i]), str(self.intersect[i]), dice[i]))

        sum_dice = torch.sum(dice)

        return sum_dice

    def backward(self, grad_output):
        # print('grad_output:',grad_output)
        input, target = self.saved_tensors
        intersect, union = self.intersect, self.union

        grad_input = torch.zeros(target.shape[0], 2, target.shape[1])#batch,2,448*448
        grad_input = grad_input.cuda()
        for i in range(input.shape[0]):
            part1 = torch.div(target[i, :], union[i])
            part2 = intersect[i] / (union[i] * union[i])
            part2 = torch.mul(input[i, :], part2)
            dice = torch.add(torch.mul(part1, 2), torch.mul(part2, -4)).cuda()

            grad_input[i, 0, :] = torch.mul(dice, grad_output.item())
            grad_input[i, 1, :] = torch.mul(dice, -grad_output.item())
        grad_input=grad_input.view(input.shape[0],2,448,448)
        # print(grad_input.shape)
        return grad_input, None
# combined with cross entropy loss, instance level
class LossVariance(nn.Module):
    """ The instances in target should be labeled
    """
    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):
        batch_size = input.size(0)
        # print(input.shape,target.shape)
        loss = 0
        for k in range(batch_size):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]
            # print('unique_vals:',unique_vals)
            sum_var = 0
            for val in unique_vals:
                instance = input[k][:, target[k][0] == val]
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
        # print(type(loss))
        loss /= batch_size
        return loss

def dice_loss(input, target):
    return DiceLoss()(input, target)

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i,:, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss

def dice_error(input, target):

    eps = 0.00001

    b, z, y, x = input.shape
    if z==1:
        predict = input.view(b, z, -1)
        predict = predict.squeeze(1).float()
    else:
        predict = input.view(b, -1).float()
    target = target.reshape(b, z*x*y).float()

    # predict = predict.argmax(1)  # dim 2 is of length 2. Reduce the length to 1 and label it with the class with highest probability

    # result = predict.cuda().float()
    # target = target_.cuda().float()


    intersect = torch.zeros(predict.shape[0]).cuda()
    union = torch.zeros(predict.shape[0]).cuda()
    dice = torch.zeros(predict.shape[0]).cuda()

    for i in range(predict.shape[0]):
        intersect[i] = torch.dot(predict[i, :], target[i, :])
        result_sum = torch.sum(predict[i, :])
        target_sum = torch.sum(target[i, :])
        union[i] = result_sum + target_sum

        dice[i] = 2 * intersect[i] / (union[i] + eps)

    mean_dice = torch.mean(dice)
    return mean_dice
class bce_and_dice(nn.Module):
    def __init__(self,batch):
        super(bce_and_dice,self).__init__()
        self.dice=dice_loss2(batch)
    def __call__(self,y_true,y_pred):
        return 1/2*nn.BCELoss()(y_pred, y_true)+self.dice(y_true,y_pred)

class weightedCE(nn.Module):
    def __init__(self):
        super(weightedCE,self).__init__()
    def forward(self,label,output):
        w=torch.sum(label)/label.shape.numel()#1
        weight=w*(1-label)+(1-w)*label

        loss=torch.nn.BCELoss(weight)(output,label)
        return loss

# class boundary_loss(nn.Module):
#     def __init__(self,batch):
#         super(boundary_loss,self).__init__()
#     def forward(self,y,x):
#
#         x1,x2=x
#         # print(x1.dtype,y.dtype)
#         loss=nn.BCELoss()(x1,y)
#
#         b=GetBoundary(y.cpu().numpy().squeeze(1))
#
#         b=torch.from_numpy(b).cuda()
#
#         loss+=weightedCE()(b,x2.squeeze(1))
#         return loss

# class FocalLoss(nn.Module):
#
#     def __init__(self, class_num=1, alpha=0.5, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#
#         # self.alpha = torch.autograd.Variable(alpha)
#         self.alpha=alpha
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, target,inputs):
#         print(inputs.grad_fn)
#         p=inputs.squeeze(1)
#         target=target.squeeze(1)
#         # p=(1-inputs)*(1-targets)+inputs*targets#计算pt
#         # alpha=(1-targets)*self.alpha+targets*(1-self.alpha)
#         # batch_loss = -alpha * (torch.pow((1 - p), self.gamma)) *p.log()
#         # return batch_loss.mean()
#         # print(p.max(),p.min())
#         # print(inputs.max())
#         loss = -(1 - p) ** self.gamma * (target * torch.log(p)) - \
#                p ** self.gamma * ((1 - target) * torch.log(1 - p))
#         # print(loss.max(),loss.min())
#         print(loss.grad_fn)
#         return loss.mean()
# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#         The losses are averaged across observations for each minibatch.
#
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#
#
#     """
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)#data.new device，dtype是一样的
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         #print(class_mask)
#
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P*class_mask).sum(1).view(-1,1)
#
#         log_p = probs.log()
#         #print('probs size= {}'.format(probs.size()))
#         #print(probs)
#
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
#         #print('-----bacth_loss------')
#         #print(batch_loss)
#
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss


