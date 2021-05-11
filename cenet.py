import torch
import torch.nn as nn
from torchvision import models
import torchvision
import torch.nn.functional as F

from functools import partial

# import Constants

nonlinearity = partial(F.relu, inplace=False)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class DACblock_with_inception(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class DACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out



class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class ChannelAtt(nn.Module):
    def __init__(self,channels):
        super(ChannelAtt, self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d((1,1))
        self.conv=nn.Conv2d(channels*2,channels,1)
        self.activate=nn.Sigmoid()
    def forward(self,x1,x2):
        x=torch.cat((x1,x2),dim=1)
        x=self.GAP(x)
        x=self.activate(self.conv(x))
        x=torch.mul(x1,x)
        x=x+x2
        # x=torch.cat((x,x2),dim=1)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CE_Net_(nn.Module):
    def __init__(self, num_classes=2, num_channels=3):
        super(CE_Net_, self).__init__()

        filters = [64,128,256,512]
        resnet = models.resnet34(pretrained=True)
        self.vars_bn = nn.ParameterList()

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # self.encoder = nn.Sequential(self.firstconv,self.firstbn,self.firstrelu,self.firstmaxpool,self.encoder1,
        #                              self.encoder2,self.encoder3,self.encoder4)
        self.dblock = DACblock(filters[-1])
        self.spp = SPPblock(filters[-1])

        self.decoder4 = DecoderBlock(filters[-1]+4, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3,2,1,1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        for i in self.modules():
            if isinstance(i,nn.BatchNorm2d):
                running_mean = nn.Parameter(torch.zeros(i.num_features), requires_grad=False)
                running_var = nn.Parameter(torch.ones(i.num_features), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

    def forward(self, x):
        # Encoder
        # xs=[]
        x = self.firstconv(x)
        # xs.append(x)
        x = self.firstbn(x)
        # xs.append(x)
        x = self.firstrelu(x)
        # xs.append(x)
        x = self.firstmaxpool(x)
        # xs.append(x)
        e1 = self.encoder1(x)
        # xs.append(e1)
        e2 = self.encoder2(e1)
        # xs.append(e2)
        e3 = self.encoder3(e2)
        # xs.append(e3)
        e4 = self.encoder4(e3)
        # xs.append(e4)

        # Center
        e4 = self.dblock(e4)
        # xs.append(e4)
        e4 = self.spp(e4)
        # xs.append(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        # xs.append(d4)
        d3 = self.decoder3(d4) + e2
        # xs.append(d3)
        d2 = self.decoder2(d3) + e1
        # xs.append(d2)
        d1 = self.decoder1(d2)
        # xs.append(d1)

        out = self.finaldeconv1(d1)
        # xs.append(out)
        out = self.finalrelu1(out)
        # xs.append(out)
        out = self.finalconv2(out)
        # xs.append(out)
        out = self.finalrelu2(out)
        # xs.append(out)
        out = self.finalconv3(out)
        # xs.append(out)
        return F.softmax(out,1),e4



    def forward_ce_net(self, x,vars=None,o=None):

        idx=96
        bn_idx = 0

        bn_training=True
        w = vars[idx]
        x = F.conv2d(x, w, stride=(2,2), padding=(3,3))
        idx += 1

        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
        x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
        idx += 2
        bn_idx += 2

        x=F.relu(x,inplace=True)
        x=F.max_pool2d(x,kernel_size=3, stride=2,padding=1)

        #BasicBlock1
        for i in list(self.encoder1.modules())[1:]:
            if isinstance(i,torchvision.models.resnet.BasicBlock):
                x,idx,bn_idx=BasicBlock(x,i,vars,self.vars_bn,idx,bn_idx)
        e1=x
        for i in list(self.encoder2.modules())[1:]:
            if isinstance(i,torchvision.models.resnet.BasicBlock):
                x,idx,bn_idx=BasicBlock(x,i,vars,self.vars_bn,idx,bn_idx)
        e2=x
        for i in list(self.encoder3.modules())[1:]:
            if isinstance(i,torchvision.models.resnet.BasicBlock):
                x,idx,bn_idx=BasicBlock(x,i,vars,self.vars_bn,idx,bn_idx)
        e3=x
        for i in list(self.encoder4.modules())[1:]:
            if isinstance(i,torchvision.models.resnet.BasicBlock):
                x,idx,bn_idx=BasicBlock(x,i,vars,self.vars_bn,idx,bn_idx)


        x,idx,bn_idx=DAC(list(self.modules())[115],x,vars,self.vars_bn,idx,bn_idx)

        x,idx,bn_idx=SPP(list(self.modules())[125],x,vars,self.vars_bn,idx,bn_idx)
        e4=x
        out,idx,bn_idx=decoder(self.decoder4,e4,vars,self.vars_bn,idx,bn_idx)
        out+=e3
        out, idx, bn_idx = decoder(self.decoder3, out, vars, self.vars_bn, idx, bn_idx)
        out+=e2
        out, idx, bn_idx = decoder(self.decoder2, out, vars, self.vars_bn, idx, bn_idx)
        out+=e1
        out, idx, bn_idx = decoder(self.decoder1, out, vars, self.vars_bn, idx, bn_idx)

        out, idx, bn_idx=final(list(self.modules())[154:],out,vars, self.vars_bn, idx,bn_idx, o)

        out=F.softmax(out,1)
        return e4,out

def final(block,x, vars, vars_bn, idx, bn_idx, o):

    for rf,i in enumerate(block):

        if isinstance(i, nn.Conv2d):
            if i.bias == None:
                w = vars[idx]
                # print(i.stride, i.padding, w.shape)
                x = F.conv2d(x, w, stride=i.stride, padding=i.padding)
                idx += 1
            else:
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=i.stride, padding=i.padding)
                idx += 2
        elif isinstance(i, nn.ConvTranspose2d):
            w, b = vars[idx], vars[idx + 1]
            x = F.conv_transpose2d(x, w, b, stride=i.stride, padding=i.padding,output_padding=i.output_padding)
            idx += 2
        if rf!=2:
            x=F.relu(x,inplace=True)


    return x, idx, bn_idx
def DAC(block,x,vars,vars_bn,idx,bn_idx):
    xs=[]
    convs=[x for x in list(block.modules())[1:]]

    w, b = vars[idx], vars[idx + 1]
    tmp = F.conv2d(x, w, b, stride=convs[0].stride, padding=convs[0].padding,dilation=convs[0].dilation)
    xs.append(tmp)

    w, b = vars[idx+2], vars[idx + 3]
    tmp = F.conv2d(x, w, b, stride=convs[1].stride, padding=convs[1].padding,dilation=convs[1].dilation)
    w, b = vars[idx + 6], vars[idx + 7]
    tmp = F.conv2d(tmp, w, b, stride=convs[3].stride, padding=convs[3].padding,dilation=convs[3].dilation)
    xs.append(tmp)

    w, b = vars[idx + 2], vars[idx + 3]
    tmp = F.conv2d(xs[0], w, b, stride=convs[1].stride, padding=convs[1].padding,dilation=convs[1].dilation)
    w, b = vars[idx + 6], vars[idx + 7]
    tmp = F.conv2d(tmp, w, b, stride=convs[3].stride, padding=convs[3].padding,dilation=convs[3].dilation)
    xs.append(tmp)

    w, b = vars[idx + 2], vars[idx + 3]
    tmp = F.conv2d(xs[0], w, b, stride=convs[1].stride, padding=convs[1].padding,dilation=convs[1].dilation)
    w, b = vars[idx + 4], vars[idx + 5]
    tmp = F.conv2d(tmp, w, b, stride=convs[2].stride, padding=convs[2].padding,dilation=convs[2].dilation)
    w, b = vars[idx + 6], vars[idx + 7]
    tmp = F.conv2d(tmp, w, b, stride=convs[3].stride, padding=convs[3].padding,dilation=convs[3].dilation)
    xs.append(tmp)
    for i in range(len(xs)):
        xs[i]=F.relu(xs[i])
    out=x+xs[0]+xs[1]+xs[2]+xs[3]
    return out,idx+8,bn_idx


def decoder(block,x,vars,vars_bn,idx,bn_idx):

    bn_training = True

    for i in block.modules():
        if isinstance(i, nn.BatchNorm2d):
            w, b = vars[idx], vars[idx + 1]
            running_mean, running_var = vars_bn[bn_idx], vars_bn[bn_idx + 1]
            x = F.relu(F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training))
            idx += 2
            bn_idx += 2
        elif isinstance(i, nn.Conv2d):
            if i.bias == None:
                w = vars[idx]
                # print(i.stride, i.padding, w.shape)
                x = F.conv2d(x, w, stride=i.stride, padding=i.padding)
                idx += 1
            else:
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=i.stride, padding=i.padding)
                idx += 2
        elif isinstance(i,nn.ConvTranspose2d):
            if i.bias == None:
                w = vars[idx]
                # print(i.stride, i.padding, w.shape)
                x = F.conv_transpose2d(x, w, stride=i.stride, padding=i.padding,output_padding=i.output_padding)
                idx += 1
            else:
                w, b = vars[idx], vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=i.stride, padding=i.padding,output_padding=i.output_padding)
                idx += 2

    return x, idx, bn_idx

def SPP(block,x,vars,vars_bn,idx,bn_idx):
    w, b = vars[idx], vars[idx + 1]
    pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
    pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
    pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
    pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
    # F.conv2d()
    conv=partial(F.conv2d, weight=w,bias=b,stride=block.stride, padding=block.padding, dilation=block.dilation)
    # tmp = F.conv2d(xs[0], w, b, )
    in_channels, h, w = x.size(1), x.size(2), x.size(3)
    layer1 = F.upsample(conv(pool1(x)), size=(h, w), mode='bilinear')
    layer2 = F.upsample(conv(pool2(x)), size=(h, w), mode='bilinear')
    layer3 = F.upsample(conv(pool3(x)), size=(h, w), mode='bilinear')
    layer4 = F.upsample(conv(pool4(x)), size=(h, w), mode='bilinear')

    out = torch.cat([layer1, layer2, layer3, layer4, x], 1)
    idx+=2
    return out,idx,bn_idx

def BasicBlock(x,block,vars,vars_bn,idx,bn_idx):

    bn_training=True
    input=x
    for i in block.modules():
        if isinstance(i, nn.BatchNorm2d):
            w, b = vars[idx], vars[idx + 1]
            running_mean, running_var = vars_bn[bn_idx], vars_bn[bn_idx + 1]
            x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
            idx += 2
            bn_idx += 2
        elif isinstance(i, nn.Conv2d):
            if i.bias == None:
                w = vars[idx]
                # print(i.stride,i.padding,w.shape)
                x = F.conv2d(x, w, stride=i.stride, padding=i.padding)
                idx += 1
            else:
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=i.stride, padding=i.padding)
                idx += 2

        elif isinstance(i, nn.MaxPool2d):
            x = F.max_pool2d(x)
        elif isinstance(i, nn.AvgPool2d):
            x = F.avg_pool2d(x)
        elif isinstance(i, nn.ReLU):
            x = F.relu(x)
        elif isinstance(i,nn.Sequential):
            j=i[0]
            w = vars[idx]
            input = F.conv2d(input, w, stride=j.stride, padding=j.padding)
            idx += 1
            w, b = vars[idx], vars[idx + 1]
            running_mean, running_var = vars_bn[bn_idx], vars_bn[bn_idx + 1]
            input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=bn_training)
            idx += 2
            bn_idx += 2
            break
    x+=input
    return F.relu(x),idx,bn_idx
from torch.autograd import Function
class GRL(Function):
    @staticmethod
    def forward(ctx,input,alpha):
        ctx.alpha=alpha
        return input

    @staticmethod
    def backward(ctx,grad_output):
        out = grad_output.neg()*ctx.alpha
        return out,None
class Domain_Discr(nn.Module):
    def __init__(self,in_channels=516,target_domains=6):
        super(Domain_Discr, self).__init__()
        self.conv1=nn.Conv2d(in_channels,256,3,padding=1)
        self.relu1=nn.ReLU(inplace=True)
        # self.dropout1=nn.Dropout2d(0.7)
        self.bn1=nn.BatchNorm2d(256)
        self.conv2=nn.Conv2d(256,128,3,padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # self.dropout2 = nn.Dropout2d(0.7)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,64,3,padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        # self.dropout3 = nn.Dropout2d(0.7)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(64,32,3,padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        # self.dropout4 = nn.Dropout2d(0.7)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        # self.dropout5 = nn.Dropout2d(0.7)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6=nn.Conv2d(16,target_domains,3,padding=1)
        self.act=nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x,alpha):
        x=GRL.apply(x,alpha)
        # x=self.grl(x)
        x=self.relu1(self.bn1(self.conv1(x)))
        x=self.relu2(self.bn2(self.conv2(x)))
        x=self.relu3(self.bn3(self.conv3(x)))
        x=self.relu4(self.bn4(self.conv4(x)))
        x=self.relu5(self.bn5(self.conv5(x)))
        x=self.conv6(x)
        return self.act(x)
class UpBlock2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpBlock2, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                stride=1)
        # self.up_sample_1 = nn.Upsample(scale_factor=2, mode="bilinear") # TODO currently not supported in PyTorch 1.4 :(
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.up_sample_1(self.conv_1(x))
class GreenBlock(nn.Module):

    def __init__(self, in_channels, out_channels=32, norm="batch"):
        super(GreenBlock, self).__init__()
        if norm == "batch":
            norm_1 = nn.BatchNorm2d(num_features=in_channels)
            norm_2 = nn.BatchNorm2d(num_features=in_channels)
        elif norm == "group":
            norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            norm_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.layer_1 = nn.Sequential(
            norm_1,
            nn.ReLU())

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            norm_2,
            nn.ReLU())

        self.conv_3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                stride=1, padding=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        y = self.conv_3(x)
        y = y + x
        return y
class BlueBlock(nn.Module):

    def __init__(self, in_channels, out_channels=32):
        super(BlueBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                              stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)
def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
class VAE(nn.Module):
    def __init__(self, in_channels=516, in_dim=(8,8), out_dim=(1,256,256)):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modalities = out_dim[0]
        self.encoder_channels = in_channels // 4  # 129
        self.split_dim = in_channels//2
        # filters=[256,128,64]

        self.reshape_dim=(16,16)# int(out_dim >> 4)
        self.linear_in_dim = int(self.encoder_channels * (in_dim[0] / 2) * (in_dim[1] / 2))#129*4*4

        self.linear_vu_dim = self.encoder_channels * self.reshape_dim[0] * self.reshape_dim[1]#129*8*8
        channels_vup2 = int(self.in_channels / 2)  # 258
        channels_vup1 = int(channels_vup2 / 2)  # 129
        channels_vup0 = int(channels_vup1 / 2)  # 64

        conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=self.encoder_channels, stride=2, kernel_size=3,
                           padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu0 = nn.ReLU(inplace=True)
        self.VD = conv_1#nn.Sequential(group_1, relu_1, conv_1)

        self.linear_1 = nn.Linear(self.linear_in_dim, in_channels)#512-->516
        self.linear_2 = nn.Linear(in_channels, in_channels//2)
        self.linear_3 = nn.Linear(in_channels, in_channels//2)#516-->258
        # TODO VU layer here
        self.linear_vu = nn.Linear(channels_vup2, self.linear_vu_dim)  # 258-->129*8*8
        relu_vu = nn.ReLU()
        VUup_block = UpBlock2(in_channels=self.encoder_channels, out_channels=self.in_channels)
        self.VU = nn.Sequential(relu_vu, VUup_block)

        self.Vup2 = UpBlock2(in_channels, channels_vup2)
        self.Vblock2 = GreenBlock(channels_vup2)

        self.Vup1 = UpBlock2(channels_vup2, channels_vup1)
        self.Vblock1 = GreenBlock(channels_vup1)

        self.Vup0 = UpBlock2(channels_vup1, channels_vup0)
        self.Vblock0 = GreenBlock(channels_vup0)

        self.Vend = BlueBlock(channels_vup0, self.modalities)

    def forward(self, x):
        # x = self.VD(self.relu0(self.bn1(x)))
        x=self.VD(x)
        x = x.view(-1, self.linear_in_dim)
        x = self.linear_1(x)
        # mu = x[:, :self.split_dim]
        # logvar=x[:, self.split_dim:]
        mu=self.linear_2(x)
        logvar=self.linear_3(x)
        # logvar = torch.log(x[:, self.split_dim:])
        y = reparametrize(mu, logvar)
        y = self.linear_vu(y)
        y = y.view(-1, self.encoder_channels, self.reshape_dim[0], self.reshape_dim[1])#B,32,8,8
        y = self.VU(y)
        y = self.Vup2(y)
        y = self.Vblock2(y)
        y = self.Vup1(y)
        y = self.Vblock1(y)
        y = self.Vup0(y)
        y = self.Vblock0(y)
        dec = self.Vend(y)
        return dec, mu, logvar
# class VAE(nn.Module):
#     def __init__(self, in_channels=516, in_dim=(8,8), out_dim=(1,256,256)):
#         super(VAE, self).__init__()
#         self.in_channels = in_channels
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.modalities = out_dim[0]
#         self.encoder_channels = in_channels//4  # int(in_channels >> 4) 129
#         self.split_dim = int(self.in_channels / 2)# 256
#
#         # self.reshape_dim = (int(self.out_dim[1] / self.encoder_channels), int(self.out_dim[2] / self.encoder_channels),
#         #                     )#8,8
#         self.reshape_dim=(16,16)# int(out_dim >> 4)
#         self.linear_in_dim = int(self.encoder_channels * (in_dim[0] / 2) * (in_dim[1] / 2))#32*4*4
#
#         self.linear_vu_dim = self.encoder_channels * self.reshape_dim[0] * self.reshape_dim[1]#32*16*16
#
#         channels_vup2 = int(self.in_channels / 2)  # 256
#         channels_vup1 = int(channels_vup2 / 2)  # 128
#         channels_vup0 = int(channels_vup1 / 2)  # 64
#
#         # group_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
#         # relu_1 = nn.ReLU()
#         conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=self.encoder_channels, stride=2, kernel_size=3,
#                            padding=1)
#
#         self.VD = conv_1#nn.Sequential(group_1, relu_1, conv_1)
#
#         self.linear_1 = nn.Linear(self.linear_in_dim, in_channels)#256-->512
#
#         # TODO VU layer here
#         self.linear_vu = nn.Linear(channels_vup2, self.linear_vu_dim)#256-->1024
#         relu_vu = nn.ReLU()
#         VUup_block = UpBlock2(in_channels=self.encoder_channels, out_channels=self.in_channels)
#         self.VU = nn.Sequential(relu_vu, VUup_block)
#
#         self.Vup2 = UpBlock2(in_channels, channels_vup2)
#         self.Vblock2 = GreenBlock(channels_vup2)
#
#         self.Vup1 = UpBlock2(channels_vup2, channels_vup1)
#         self.Vblock1 = GreenBlock(channels_vup1)
#
#         self.Vup0 = UpBlock2(channels_vup1, channels_vup0)
#         self.Vblock0 = GreenBlock(channels_vup0)
#
#         self.Vend = BlueBlock(channels_vup0, self.modalities)
#
#     def forward(self, x):
#         x = self.VD(x)
#         x = x.view(-1, self.linear_in_dim)
#         x = self.linear_1(x)
#         mu = x[:, :self.split_dim]
#         logvar=x[:, self.split_dim:]
#         # logvar = torch.log(x[:, self.split_dim:])
#         y = reparametrize(mu, logvar)
#         y = self.linear_vu(y)
#         y = y.view(-1, self.encoder_channels, self.reshape_dim[0], self.reshape_dim[1])#B,32,8,8
#         y = self.VU(y)
#         y = self.Vup2(y)
#         y = self.Vblock2(y)
#         y = self.Vup1(y)
#         y = self.Vblock1(y)
#         y = self.Vup0(y)
#         y = self.Vblock0(y)
#         dec = self.Vend(y)
#         return dec, mu, logvar
if __name__=='__main__':
    net=CE_Net_()
    net.train()
    c=0
    for i in net.parameters():
        c+=i.numel()
        print(i.numel())
    print(c)
    input=torch.randn((2,3,256,256))
    o = net(input)
    para=list(net.parameters())
    out=net.forward_ce_net(input,para,o[1])

    print(o.shape)