import net
import load_data as ld
import numpy as np
import torch
import pdb
from PIL import Image
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
import torch.optim as optim
import logging
import torchvision.utils  as tov
from torch import Tensor as tensor
#pdb.set_trace()
gpuid=0
lr_rate=0.0002
alpha=0.0005
beta=0.01*alpha
num_iter=500000
optim_betas=(0.9,0.999)
bs=64
labelfake=Vb(torch.from_numpy(np.full((bs),2,dtype=int))).cuda(gpuid)
labelpo=Vb(torch.from_numpy(np.full((bs),1,dtype=int))).cuda(gpuid)
labelne=Vb(torch.from_numpy(np.full((bs),0,dtype=int))).cuda(gpuid)
logging.basicConfig(filename='log/residualgan_v10.log', level=logging.INFO)
G1=net.Generator1().cuda(gpuid)
G2=net.Generator2().cuda(gpuid)
D=net.Discriminator().cuda(gpuid)
d_optimizer=optim.Adam(D.parameters(),lr=lr_rate,betas=optim_betas)
g1_optimizer=optim.Adam(G1.parameters(),lr=lr_rate,betas=optim_betas)
g2_optimizer=optim.Adam(G2.parameters(),lr=lr_rate,betas=optim_betas)
l1_crit=nn.L1Loss(size_average=False)
datalistpo=ld.getlist('../Eyeglasses_Positive.txt')
datalistne=ld.getlist('../Eyeglasses_Negative.txt')
iternow1=0
iternow2=0
for iter1 in xrange(num_iter):
    D.zero_grad()
    datapo,iternow1=ld.load_data('../img_align_celeba_crop/','../Eyeglasses_Positive.txt',datalistpo,iternow1,bs,gpuid=gpuid)
    datane,iternow2=ld.load_data('../img_align_celeba_crop/','../Eyeglasses_Negative.txt',datalistne,iternow2,bs,gpuid=gpuid)
    datapo_,_=G1(datane)
    datane_,_=G2(datapo)
    dlabelpo,_=D(datapo)
    dlabelne,_=D(datane)
    dlabelpo_,_=D(datapo_)
    dlabelne_,_=D(datane_)
    trueloss1=F.cross_entropy(dlabelpo,labelpo)
    trueloss2=F.cross_entropy(dlabelne,labelne)
    fakeloss1=F.cross_entropy(dlabelpo_,labelfake)
    fakeloss2=F.cross_entropy(dlabelne_,labelfake)
    trueloss1.backward()
    trueloss2.backward()
    fakeloss1.backward()
    fakeloss2.backward()
    if iter1%100==0:
        outinfo=str(trueloss1)+' '+str(trueloss2)+' '+str(fakeloss1)+' '+str(fakeloss2)
        logging.info(outinfo)
        print 'dloss',outinfo
    d_optimizer.step()
    
    
    datapo,iternow1=ld.load_data('../img_align_celeba_crop/','../Eyeglasses_Positive.txt',datalistpo,iternow1,bs,gpuid=gpuid)
    datane,iternow2=ld.load_data('../img_align_celeba_crop/','../Eyeglasses_Negative.txt',datalistne,iternow2,bs,gpuid=gpuid)
    G1.zero_grad()
    G2.zero_grad()
    datapo_,resu1=G1(datane)
    dlabelpo_,out3_=D(datapo_)
    dlabelne,out3=D(datane)
    datane_2,_=G2(datapo_)
    dlabelne_2,_=D(datane_2)
    ganloss1=F.cross_entropy(dlabelpo_,labelpo)
    pixloss1=torch.sum(torch.abs(resu1))/bs
    perloss1=torch.sum(torch.abs(out3-out3_))/bs
    dualloss1=F.cross_entropy(dlabelne_2,labelne)
    g1loss=ganloss1+alpha*pixloss1+beta*perloss1+dualloss1
    #g1loss=ganloss1+dualloss1+alpha*pixloss1
    if iter1%100==0:
        outinfo=str(g1loss)
        logging.info(outinfo)
        print 'g1loss',outinfo
    g1loss.backward()
    #g1_optimizer.step()

    datane_,resu2=G2(datapo)
    dlabelne_,out3_=D(datane_)
    dlabelpo,out3=D(datapo)
    datapo_2,_=G2(datane_)
    dlabelpo_2,_=D(datapo_2)
    ganloss2=F.cross_entropy(dlabelne_,labelne)
    pixloss2=torch.sum(torch.abs(resu2))/bs
    perloss2=torch.sum(torch.abs(out3-out3_))/bs
    dualloss2=F.cross_entropy(dlabelpo_2,labelpo)
    g2loss=ganloss2+alpha*pixloss2+beta*perloss2+dualloss2
    #g2loss=ganloss2+dualloss2+alpha*pixloss2    
    if iter1%100==0:
        outinfo=str(g2loss)
        logging.info(outinfo)
        print 'g2loss',outinfo
        logging.info(str(iter1))
    g2loss.backward()
    g1_optimizer.step()
    g2_optimizer.step()
    print iter1
    #logging.info(str(iter1))
    if iter1 % 200 == 0:
        saveim=datapo.cpu().data
        tov.save_image(saveim,'img/datapo'+str(iter1)+'.jpg')
        saveim=datane.cpu().data
        tov.save_image(saveim,'img/datane'+str(iter1)+'.jpg')
        saveim=datapo_.cpu().data
        tov.save_image(saveim,'img/datapo_'+str(iter1)+'.jpg')
        saveim=datane_.cpu().data
        tov.save_image(saveim,'img/datane_'+str(iter1)+'.jpg')
        save_name = 'model/{}_iter_{}.pth.tar'.format('residualgan', iter1)
        torch.save({'G1': G1.state_dict(), 'G2': G2.state_dict(),'D': D.state_dict()}, save_name)
        logging.info('save model to {}'.format(save_name))


    

    
    

 

