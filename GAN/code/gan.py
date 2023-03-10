import argparse
import os
import numpy as np 
from paddle.vision import transforms
#from paddle.vision.utils import save_image 
from paddle.io import DataLoader
#from torch.autograd import Variable
# paddle.nn.initializer.Assign(paddle.zeros(batch_size, 32, row, col))) 
# https://aistudio.baidu.com/paddle/forum/topic/show/992959

import paddle 
import paddle.nn as nn 

from tqdm import tqdm

os.makedirs("images",exist_ok=True)
def argopt():
    
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt.__dict__)
    return opt
opt=argopt()
img_shape = (opt.channels, opt.img_size, opt.img_size)
class Generator(nn.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat,out_feat,normalize=True):
            Layers=[ nn.Linear(in_feat,out_feat)]
            if normalize:
                Layers.append(nn.BatchNorm1D(out_feat,0.8))
            Layers.append(nn.LeakyReLU(0.2))
            return Layers
        self.model =nn.Sequential(
            *block(opt.latent_dim,128,normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self,z):
        img=self.model(z)
        img=paddle.reshape(img ,(img.shape[0],*img_shape))
        return img

gnet=Generator()
paddle.summary(gnet,(1,opt.latent_dim))

class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
            )
    def forward(self, img):
        img_flat = paddle.reshape(img,(img.shape[0],-1)) 
        validity = self.model(img_flat)
        return validity

dNet=Discriminator()
paddle.summary(dNet,(1,*img_shape))



# Loss function,交叉熵损失函数
adversarial_loss = nn.BCELoss()


# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()


from paddle.vision.datasets import MNIST
import paddle.vision.transforms as T
minsttransform=T.Compose(
    [
        T.Resize(opt.img_size),
        T.ToTensor(),
        T.Normalize([0.5],[0.5])
    ]
)

mnist_dataset = MNIST(
    mode="train",
    # image_path="~/data",
    # download=True,
    transform=minsttransform
)
dataloader=DataLoader(mnist_dataset,    
    batch_size=opt.batch_size,
    shuffle=True,)

# 优化器
optimizer_G = paddle.optimizer.Adam(parameters=generator.parameters(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)
optimizer_D = paddle.optimizer.Adam(parameters=discriminator.parameters(), learning_rate=opt.lr, beta1=opt.b1, beta2=opt.b2)




for epoch in tqdm(range(opt.n_epochs)):
    for i,(imgs,_) in enumerate(dataloader):
        valid = paddle.ones([imgs.shape[0],1]) #nn.Layer.create_parameter(shape=(imgs.shape[0],1))
        fake = paddle.zeros([imgs.shape[0],1])
        real_imgs = imgs #imgs.type(paddle.Tensor)
        

        #--------------------
        # 训练生成器
        #--------------------

        optimizer_G.clear_grad()

        # 数据输入采样
        z =  paddle.fluid.layers.gaussian_random(
            (imgs.shape[0], opt.latent_dim), 
            mean=0.0, std=1.0, seed=0, dtype='float32', name=None)
        # paddle.to_tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))) 神奇，使用paddle就不行啊？原理是什么啊？
                            
                                

        # 批量生成图片
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
        # 反向传播
        g_loss.backward()
        # 更新参数
        optimizer_G.step()


        #--------------------
        # 训练鉴别器
        #--------------------

        optimizer_D.clear_grad()

        # 计算损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2


        d_loss.backward()
        optimizer_D.step()
        if i%200==0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        # batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        # paddle 中 gen_imgs.data 没有data这个属性
