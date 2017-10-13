import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
class Gen1(nn.Module):
    def __init__(self):
        super(Gen1, self).__init__()
        self.conv1= nn.Conv2d(3,64,5,stride=1,padding=2)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,128,4,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,256,4,stride=2,padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.deconv1=nn.ConvTranspose2d(256,128,3,stride=2,padding=1)
        self.bn4=nn.BatchNorm2d(128)
        self.deconv2=nn.ConvTranspose2d(128,64,3,stride=2)
        self.bn5=nn.BatchNorm2d(64)
        self.convend=nn.Conv2d(64,3,4,stride=1,padding=2)
    def forward(self,x):
        out1=F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        out2=F.leaky_relu(self.bn2(self.conv2(out1)),0.2)
        out3=F.leaky_relu(self.bn3(self.conv3(out2)),0.2)
        #print out3.size()
        out4=F.leaky_relu(self.bn4(self.deconv1(out3)),0.2)
        #print out4.size()
        out5=F.leaky_relu(self.bn5(self.deconv2(out4)),0.2)
        #print out5.size()
        out6=self.convend(out5)
        #print out6.size()
        outres=x+out6,out6
        return outres
class Gen2(nn.Module):
    def __init__(self):
        super(Gen2, self).__init__()
        self.conv1= nn.Conv2d(3,64,5,stride=1,padding=2)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,128,4,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,256,4,stride=2,padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.deconv1=nn.ConvTranspose2d(256,128,3,stride=2,padding=1)
        self.bn4=nn.BatchNorm2d(128)
        self.deconv2=nn.ConvTranspose2d(128,64,3,stride=2)
        self.bn5=nn.BatchNorm2d(64)
        self.convend=nn.Conv2d(64,3,4,stride=1,padding=2)
    def forward(self,x):
        out1=F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        out2=F.leaky_relu(self.bn2(self.conv2(out1)),0.2)
        out3=F.leaky_relu(self.bn3(self.conv3(out2)),0.2)
        #print out3.size()
        out4=F.leaky_relu(self.bn4(self.deconv1(out3)),0.2)
        #print out4.size()
        out5=F.leaky_relu(self.bn5(self.deconv2(out4)),0.2)
        #print out5.size()
        out6=self.convend(out5)
        #print out6.size()
        outres=x+out6
        return outres,out6


class Generator1(nn.Module):
    def __init__(self):
        super(Generator1,self).__init__()
        self.conv1= nn.Conv2d(3,64,5,stride=1,padding=2)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,128,4,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,256,4,stride=2,padding=1)#32
        self.bn3=nn.BatchNorm2d(256)
        self.conv4=nn.Conv2d(256,512,4,stride=2,padding=1)#16
        self.bn4=nn.BatchNorm2d(512)
        self.conv5=nn.Conv2d(512,100,4,stride=2,padding=1)#8
        self.bn5=nn.BatchNorm2d(100)
        self.deconv1=nn.ConvTranspose2d(100,512,3,stride=2,padding=1)#16
        self.bn6=nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,3,stride=2)#32
        self.bn7=nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,128,3,stride=2)#64
        self.bn8=nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128,64,3,stride=2)#128
        self.bn9=nn.BatchNorm2d(64)
        self.convend=nn.Conv2d(64,3,4,stride=1,padding=2)
    def forward(self,x):
        out1=F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        out2=F.leaky_relu(self.bn2(self.conv2(out1)),0.2)
        out3=F.leaky_relu(self.bn3(self.conv3(out2)),0.2)
        out4=F.leaky_relu(self.bn4(self.conv4(out3)),0.2)
        out5=F.leaky_relu(self.bn5(self.conv5(out4)),0.2)
        #print out5.size()
        out6=F.leaky_relu(self.bn6(self.deconv1(out5)),0.2)
        #print out6.size()
        out7=F.leaky_relu(self.bn7(self.deconv2(out6)),0.2)
        #print out7.size()
        out8=F.leaky_relu(self.bn8(self.deconv3(out7)),0.2)
        #print out8.size()
        out9=F.leaky_relu(self.bn9(self.deconv4(out8)),0.2)
        #print out9.size()
        outend=self.convend(out9)
        #print outend.size()
        outres=outend+x
        return outres,outend


class Generator2(nn.Module):
    def __init__(self):
        super(Generator2,self).__init__()
        self.conv1= nn.Conv2d(3,64,5,stride=1,padding=2)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,128,4,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,256,4,stride=2,padding=1)#32
        self.bn3=nn.BatchNorm2d(256)
        self.conv4=nn.Conv2d(256,512,4,stride=2,padding=1)#16
        self.bn4=nn.BatchNorm2d(512)
        self.conv5=nn.Conv2d(512,100,4,stride=2,padding=1)#8
        self.bn5=nn.BatchNorm2d(100)
        self.deconv1=nn.ConvTranspose2d(100,512,3,stride=2,padding=1)#16
        self.bn6=nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,3,stride=2)#32
        self.bn7=nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,128,3,stride=2)#64
        self.bn8=nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128,64,3,stride=2)#128
        self.bn9=nn.BatchNorm2d(64)
        self.convend=nn.Conv2d(64,3,4,stride=1,padding=2)
    def forward(self,x):
        out1=F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        out2=F.leaky_relu(self.bn2(self.conv2(out1)),0.2)
        out3=F.leaky_relu(self.bn3(self.conv3(out2)),0.2)
        out4=F.leaky_relu(self.bn4(self.conv4(out3)),0.2)
        out5=F.leaky_relu(self.bn5(self.conv5(out4)),0.2)
        #print out5.size()
        out6=F.leaky_relu(self.bn6(self.deconv1(out5)),0.2)
        #print out6.size()
        out7=F.leaky_relu(self.bn7(self.deconv2(out6)),0.2)
        #print out7.size()
        out8=F.leaky_relu(self.bn8(self.deconv3(out7)),0.2)
        #print out8.size()
        out9=F.leaky_relu(self.bn9(self.deconv4(out8)),0.2)
        #print out9.size()
        outend=self.convend(out9)
        #print outend.size()
        outres=outend+x
        return outres,outend
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1= nn.Conv2d(3,64,4,stride=2,padding=1)#64
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,128,4,stride=2,padding=1)#32
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,256,4,stride=2,padding=1)#16
        self.bn3=nn.BatchNorm2d(256)
        self.conv4=nn.Conv2d(256,512,4,stride=2,padding=1)#8
        self.bn4=nn.BatchNorm2d(512)
        self.conv5=nn.Conv2d(512,1024,4,stride=2,padding=1)#4
        self.bn5=nn.BatchNorm2d(1024)
        self.convend=nn.Conv2d(1024,1024,4,stride=1,padding=0)
        self.fc=nn.Linear(1024,3)
    def forward(self,x):
        out1=F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        out2=F.leaky_relu(self.bn2(self.conv2(out1)),0.2)
        out3=F.leaky_relu(self.bn3(self.conv3(out2)),0.2)
        out4=F.leaky_relu(self.bn4(self.conv4(out3)),0.2)
        out5=F.leaky_relu(self.bn5(self.conv5(out4)),0.2)
        out6=self.convend(out5)
        #print out6.size()
        out7=out6.view(-1,1024)
        outend=self.fc(out7)
        return outend,out3

        

'''test code
i=Gen1openmouth().cuda(0)
t=Vb(torch.randn(1, 3, 128, 128).cuda(0))
print i(t)
'''

'''
h=Generator2().cuda(0)
b=Discriminator().cuda(0)
t=Vb(torch.randn(2, 3, 128, 128).cuda(0))
a,_=h(t)
d=a.cpu()
print d.data.numpy()
k,_=b(a)
print k
'''

        
        
        


