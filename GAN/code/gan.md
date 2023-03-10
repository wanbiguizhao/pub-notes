## 基本信息
gan.py 参考 https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
为了学习和pytorch和paddle学习使用的
### 论文复现的困惑
1. 复现的代码和论文复现的优化顺序不一致：
- 论文是先优化判别器，然后才是再优化的生成器。
-  代码先优化的是生成器的的损失，然后才是判别器。
2. 算法和代码实现不一致
生成器最好，需要优化的是log(D(G(z)))，最大，代码使用的的是，BCE，对应的就是 交叉熵损失函数最小。
adversarial_loss(discriminator(gen_imgs), valid) ，表示希望生成的数据都是真的，通过反向传播，激励g生成搞好的内容，这部分对上了
判别器最好：
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
d_loss = (real_loss + fake_loss) / 2
这部分和公式没有对应起来，尤其是最后为啥要除以2，公式上没有。



### paddle和pytorch的比较
基本的api都可以对应上，paddle脚clear_guard() ，pytorch叫zero_giard() 
- 但是pytorch中关联的Variable 和paddle对应不上
- 随机变量的生成，编程网络可以处理的格式，不太一致

pytorch->   z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
找了一下paddle对应的:paddle.fluid.layers.gaussian_random(
            (imgs.shape[0], opt.latent_dim), 
            mean=0.0, std=1.0, seed=0, dtype='float32', name=None)

real_imgs 传入到paddle就可以使用
但是pytorch就需要做这样的操作        real_imgs = Variable(imgs.type(Tensor))

