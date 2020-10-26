```
GAN的鼻祖之作是2014年NIPS一篇文章：Generative Adversarial Net（https://arxiv.org/abs/1406.2661）
```

一个目前各类GAN的一个论文整理集合：

https://deephunt.in/the-gan-zoo-79597dc8c347

一个目前各类GAN的一个代码整理集合：

https://github.com/zhangqianhui/AdversarialNetsPapers

# *GAN主要思想*

GAN网络（生成对抗网络），可以认为是一个造假机器，造出来的东西跟真的一样。

主要分为两个部分，一个部分是具有生成模型功能的样本生成器，

```
输入一个噪声/样本，然后把它包装成一个逼真的样本，也就是输出。
```

另一个部分是具有判别功能的判别器，

```
判别模型：比作一个二分类器（如同0-1分类器），来判断输入的样本是真是假。（就是输出值大于0.5还是小于0.5）
```

如图：

![image-20201015193234129](C:/Users/15429/AppData/Roaming/Typora/typora-user-images/image-20201015193234129.png)



- **我们有什么？** 
  比如上面的这个图，我们有的只是真实采集而来的人脸样本数据集，仅此而已，而且很关键的一点是我们连人脸数据集的类标签都没有，也就是我们不知道那个人脸对应的是谁。



- **我们要得到什么？**
  至于要得到什么，不同的任务得到的东西不一样，我们只说最原始的GAN目的，那就是我们想通过输入一个噪声，模拟得到一个人脸图像，这个图像可以非常逼真以至于以假乱真。



好了再来理解下GAN的两个模型要做什么。



首先**判别模型**，就是图中右半部分的网络，直观来看就是一个简单的神经网络结构，输入就是一副图像，输出就是一个概率值，用于判断真假使用（概率值大于0.5那就是真，小于0.5那就是假），真假也不过是人们定义的概率而已。



其次是**生成模型**，生成模型要做什么呢，同样也可以看成是一个神经网络模型，输入是一组随机数Z，输出是一个图像，不再是一个数值而已。从图中可以看到，会存在两个数据集，一个是真实数据集，这好说，另一个是假的数据集，那这个数据集就是有生成网络造出来的数据集。好了根据这个图我们再来理解一下GAN的目标是要干什么：



- **判别网络的目的**：就是能判别出来属于的一张图它是来自真实样本集还是假样本集。假如输入的是真样本，网络输出就接近1，输入的是假样本，网络输出接近0，那么很完美，达到了很好判别的目的。



- **生成网络的目的**：生成网络是造样本的，它的目的就是使得自己造样本的能力尽可能强，强到什么程度呢，你判别网络没法判断我是真样本还是假样本。



于是可以简单的理解GAN网络：生成（造假网络）主要负责用已有的"真样本"生成“假样本”，而这张生成的样本最好是可以以假乱真，让别人看不出来他是假的，也就是与真样本有高度的相似性。而与之对应的判别网络，就是在每一次生成网络生成新样本时，把这个“假样本”放进网络，判别它是否是假的。

当然，作为设计者，我们最后希望的是生成网络创建的“假样本”达到让判别网络无法判断的程度。



# *深入理解*



虽然思想设计的非常巧妙，但是又如何具体来实现这个网络呢？用图来解释最为直接：

![image-20201015195206251](C:/Users/15429/AppData/Roaming/Typora/typora-user-images/image-20201015195206251.png)



注意：GAN网络中的两个网络彼此并没有特别多的联系，它们是两个相对独立的网络，分别有不同的训练机制。

好了那么训练这样的两个模型的大方法就是：**单独交替迭代训练**。

什么意思？因为是2个网络，不好一起训练，所以才去交替迭代训练，我们一一来看。 

## 生成网络



```
假设现在生成网络模型已经有了（当然可能不是最好的生成网络），那么给一堆随机数组，就会得到一堆假的样本集（因为不是最终的生成模型，那么现在生成网络可能就处于劣势，导致生成的样本就不咋地，可能很容易就被判别网络判别出来了说这货是假冒的），但是先不管这个，假设我们现在有了这样的假样本集，真样本集一直都有，现在我们人为的定义真假样本集的标签，因为我们希望真样本集的输出尽可能为1，假样本集为0，很明显这里我们就已经默认真样本集所有的类标签都为1，而假样本集的所有类标签都为0. 
```

需要注意的一点是，现在我们不需要去分辨真样本集中，样本的类别不同的问题。

比如真样本中有男人的图片，有女人的图片。判别网络判别的不是生成网络生成后的图片是否为男\女，而是判断这张图片是不是造假生成的。

对于生成网络，想想我们的目的，是生成尽可能逼真的样本。那么原始的生成网络生成的样本你怎么知道它真不真呢？就是送到判别网络中，所以在训练生成网络的时候，我们需要联合判别网络一起才能达到训练的目的。什么意思？就是如果我们单单只用生成网络，那么想想我们怎么去训练？误差来源在哪里？细想一下没有，但是如果我们把刚才的判别网络串接在生成网络的后面，这样我们就知道真假了，也就有了误差了。所以对于生成网络的训练其实是对生成-判别网络串接的训练，就像图中显示的那样。好了那么现在来分析一下样本，原始的噪声数组Z我们有，也就是生成了假样本我们有，此时很关键的一点来了，我们要**把这些假样本的标签都设置为1**，也就是认为这些假样本在生成网络训练的时候是真样本。

那么为什么要这样呢？因为这样才能起到迷惑判别器的目的，也才能使得生成的假样本逐渐逼近为正样本。好了，重新顺一下思路，现在对于生成网络的训练，我们有了样本集（只有假样本集，没有真样本集），有了对应的label（全为1）。就可以开始训练生成网络了。

一类样本怎么训练网络？其实没有问题，训练网络的实质是反向传播，更新参数。也就是生成的假样本被判别网络判断为假之后，就会把这个参数反向传回生成网络，并告诉它你这样不行，判别网络能看出来，然后生成网络就获得了经验，更新造假图的参数，最终造出判别网络无法识别的假图。

## 判别网络

回过头来，我们现在有了真样本集以及它们的label（都是1）、假样本集以及它们的label（都是0），这样单就判别网络来说，此时问题就变成了一个再简单不过的**有监督的二分类问题**了，直接送到神经网络模型中训练就完事了。



# 进一步理解



文字的描述相信已经让大多数的人知道了这个过程，下面我们来看看原文中几个重要的数学公式描述，首先我们直接上原始论文中的目标公式吧：



![](C:/Users/15429/AppData/Roaming/Typora/typora-user-images/5af1b3080001290e06940048%5B1%5D.jpg)





上述这个公式说白了就是一个最大最小优化问题，其实对应的也就是上述的两个优化过程。有人说如果不看别的，能达看到这个公式就拍案叫绝的地步，那就是机器学习的顶级专家，哈哈，真是前路漫漫。同时也说明这个简单的公式意义重大。

这个公式既然是最大最小的优化，那就不是一步完成的，其实对比我们的分析过程也是这样的，这里现优化D，然后在取优化G，本质上是两个优化问题，把拆解就如同下面两个公式：

优化G：



![](C:/Users/15429/AppData/Roaming/Typora/typora-user-images/5af1b308000198d704400050%5B1%5D.jpg)



优化D：



![](C:/Users/15429/AppData/Roaming/Typora/typora-user-images/5af1b3080001832106650039%5B1%5D.jpg)



可以看到，优化D的时候，也就是判别网络，其实没有生成网络什么事，后面的G(z)这里就相当于已经得到的假样本。优化D的公式的第一项，使的真样本x输入的时候，得到的结果越大越好，可以理解，因为需要真样本的预测结果越接近于1越好嘛。对于假样本，需要优化是的其结果越小越好，也就是D(G(z))越小越好，因为它的标签为0。但是呢第一项是越大，第二项是越小，这不矛盾了，所以呢把第二项改成1-D(G(z))，这样就是越大越好，两者合起来就是越大越好。 那么同样在优化G的时候，这个时候没有真样本什么事，所以把第一项直接去掉了。这个时候只有假样本，但是我们说这个时候是希望假样本的标签是1的，所以是D(G(z))越大越好，但是呢为了统一成1-D(G(z))的形式，那么只能是最小化1-D(G(z))，本质上没有区别，只是为了形式的统一。之后这两个优化模型可以合并起来写，就变成了最开始的那个最大最小目标函数了。

所以回过头来我们来看这个最大最小目标函数，里面包含了判别模型的优化，包含了生成模型的以假乱真的优化，完美的阐释了这样一个优美的理论。





# 关于代码

先从图片开始，当理解了网络的功能之后，具体该怎么实现代码呢？

# How to Train a GAN? Tips and tricks to make GANs work

While research in Generative Adversarial Networks (GANs) continues to improve the fundamental stability of these models, we use a bunch of tricks to train them and make them stable day to day.

Here are a summary of some of the tricks.

[Here's a link to the authors of this document](https://github.com/soumith/ganhacks#authors)

## 1. Normalize the inputs

- normalize the images between -1 and 1
- Tanh as the last layer of the generator output
- 对输入的图片进行归一化处理，Tanh作为最后一层输出

## 2: A modified loss function

In GAN papers, the loss function to optimize G is `min (log 1-D)`, but in practice folks practically use `max log D`

- because the first formulation has vanishing gradients early on
- Goodfellow et. al (2014)

In practice, works well:

- Flip labels when training generator: real = fake, fake = real
- 在GAN中优化器的选择和参数设置

## 3: Use a spherical Z

- Dont sample from a Uniform distribution

- Sample from a gaussian distribution

- When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B
- Tom White's [Sampling Generative Networks](https://arxiv.org/abs/1609.04468) ref code https://github.com/dribnet/plat has more details

## 4: BatchNorm

- Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images.
- when batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation).
- 为真假构造不同的小批量，即每个小批量只需要包含所有真实图像或所有生成的图像。当batchnorm不是选项时，使用实例规范化（对于每个样本，减去平均值并除以标准差）。

## 5: Avoid Sparse Gradients: ReLU, MaxPool

- the stability of the GAN game suffers if you have sparse gradients
- LeakyReLU = good (in both G and D)
- For Downsampling, use: Average Pooling, Conv2d + stride
- For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
- 对于下采样，请使用：Average Pooling，Conv2d+stride              
- 对于向上采样，请使用：PixelShuffle、ConvTranspose2d+stride
  - PixelShuffle: https://arxiv.org/abs/1609.05158

## 6: Use Soft and Noisy Labels

- Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
  - Salimans et. al. 2016
- make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator
- 标签平滑，也就是说，如果你有两个目标标签：Real=1和Fake=0，那么对于每个传入的样本，如果它是真的，那么用一个介于0.7和1.2之间的随机数替换标签，如果它是一个伪样本，则将其替换为0.0和0.3（例如）。（虽然不懂为什么）

## 7: DCGAN / Hybrid Models

- Use DCGAN when you can. It works!
- if you cant use DCGANs and no model is stable, use a hybrid model : KL + GAN or VAE + GAN

## 8: Use stability tricks from RL

- Experience Replay
  - Keep a replay buffer of past generations and occassionally show them
  - Keep checkpoints from the past of G and D and occassionaly swap them out for a few iterations
- All stability tricks that work for deep deterministic policy gradients
- See Pfau & Vinyals (2016)
- 对学习率进行稳定，checkpoints存下部分较好的model

## 9: Use the ADAM Optimizer

- optim.Adam rules!
  - See Radford et. al. 2015
- Use SGD for discriminator and ADAM for generator
- 可以用SGD作为鉴别器，ADAM作为发生器（SGD作为训练优化器，ADAM作为测试发生器）

## 10: Track failures early

- D loss goes to 0: failure mode
- check norms of gradients: if they are over 100 things are screwing up
- when things are working, D loss has low variance and goes down over time vs having huge variance and spiking
- if loss of generator steadily decreases, then it's fooling D with garbage (says martin)

## 11: Dont balance loss via statistics (unless you have a good reason to)

- Dont try to find a (number of G / number of D) schedule to uncollapse training
- It's hard and we've all tried it.
- If you do try it, have a principled approach to it, rather than intuition

For example

```
while lossD > A:
  train D
while lossG > B:
  train G
```

## 12: If you have labels, use them

- if you have labels available, training the discriminator to also classify the samples: auxillary GANs
- 如果你的图片拥有标签，可以用来辅助鉴别器

## 13: Add noise to inputs, decay over time

- Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
  - http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
  - https://openreview.net/forum?id=Hk4_qw5xe
- adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
  - Improved GANs: OpenAI code also has it (commented out)
  - 在D输入增加一些高斯噪声，这是方便D网络造假

## 14: [notsure] Train discriminator more (sometimes)

- especially when you have noise
- hard to find a schedule of number of D iterations vs G iterations

## 15: [notsure] Batch Discrimination

- Mixed results

## 16: Discrete variables in Conditional GANs

- Use an Embedding layer
- Add as additional channels to images
- Keep embedding dimensionality low and upsample to match image channel size
- 使用嵌入层              作为附加频道添加到图像              保持嵌入维数低和高采样以匹配图像通道大小

## 17: Use Dropouts in G in both train and test phase

- Provide noise in the form of dropout (50%).
- Apply on several layers of our generator at both training and test time
- https://arxiv.org/pdf/1611.07004v1.pdf
- Dropout=0.5在G网络中。



# 关于LOSS的计算处理

 **Discriminator和Generator损失计算**

 

　　GANs和很多其他模型不同，GANs在训练时需要同时运行两个优化算法，我们需要为discriminator和generator分别定义一个优化器，一个用来来最小化discriminator的损失，另一个用来最小化generator的损失。即**loss = d_loss + g_loss**

 

　　**d_loss计算方法：**

 

　　对于辨别器discriminator，其损失等于真实图片和生成图片的损失之和，即 **d_loss = d_loss_real + d_loss_fake** , losses 均由交叉熵计算而得。



　　在计算真实数据产生的损失d_loss_real时，我们希望辨别器discriminator输出1；而在计算生成器生成的 ‘假’ 数据所产生的损失d_loss_fake时，我们希望discriminator输出0.

 

　　因此，对于真实数据，在计算其损失时，将上式中的labels全部都设为1，因为它们都是真实的。为了是增强辨别器discriminator的泛化能力，可以将labels设为0.9，而不是1.0。

 

　　对于生成器生成的‘假’数据，在计算其损失d_loss_fake时，将上式中的labels全部设为0。

 

　　**g_loss计算方法：**

 

　　最后，生成器generator的损失用 '假' 数据的logits（即d_logits_fake），但是，现在所有的labels全部设为1（即我们希望生成器generator输出1）。这样，通过训练，生成器generator试图 ‘骗过’ 辨别器discriminator。



​     优化目标则是：

​      minGmaxDLoss



# 代码分析

### 首先，定义基本的训练参数，如epoch，batch_size，learningrate等

```
parser = argparse.ArgumentParser()
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
```

### Generator 生成器类及接口

```
 def block(in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    self.model = nn.Sequential(
        *block(opt.latent_dim, 128, normalize=False),
        *block(128, 256),
        *block(256, 512),
        *block(512, 1024),
        nn.Linear(1024, int(np.prod(img_shape))),
        nn.Tanh()
    )

def forward(self, z):
    img = self.model(z)
    img = img.view(img.size(0), *img_shape)
    return img
```

block函数对输入数据进行归一化处理，并定义卷积核中图像处理的参数，一共四个卷积核。

卷积之后接一个nn.Linear,这里是把1024个输入特征全连接，输出img_shape的特征，这里也是1024

nn.Tanh()是一个激活函数，tanh函数的导数比sigmoid函数导数值更大，即梯度变化更快，也就是在训练过程中收敛速度更快。

​       forward函数是对图像进行矩阵化的数据预处理。

### Discriminator鉴别器类及接口

```
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
```

我们知道，Discriminator的工作是一个有监督的二分类工作。

输入一个图片，它将鉴别出图片真假的一个置信区间，也就是一个概率。

这个鉴别器的网络实现较为简单，它采用了逐渐缩小输入特征数量，并在输入过程中将特征乘以一些超参数，最后把输出特征数缩小到1，得到结果的过程。采用sigmoid作为激活函数。

forward的操作同生成器。



### Loss函数的初始化

```
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
```

​        这里使用了torch自带的BCELoss()，我们从上面的经验也可以看到，在GAN的网络训练中，这个损失函数是常用的。

​        初始化生成器和鉴别器。

### 导入数据

```
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
```

采用os自带的makedirs函数通过文件夹导入数据集，获得一个列表List，可以看到使用了Normalize对数据进行了预处理。

shuffle=True代表随机打乱输入数据。

### 定义优化器Optimizer

```
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
```

​       loss损失优化器用到的都是Adam优化器，学习率一样。

### 训练

```
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)  #打标签，真图为1
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)   #假图为0

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        #另外Pytorch 为什么每一轮batch需要设置optimizer.zero_grad：
        #根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑          问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))) #加入随机噪声

        # Generate a batch of images
        gen_imgs = generator(z) #生成一批假图

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)


        g_loss.backward() #反向传播
        optimizer_G.step() #生成器参数更新

        #那么为什么optimizer.step()需要放在每一个batch训练中，而不是epoch训练中，
        # 这是因为现在的mini-batch训练模式是假定每一个训练集就只有mini-batch这样大，
        # 因此实际上可以将每一次mini-batch看做是一次训练，一次训练更新一次参数空间，因而optimizer.step()放在这里。

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        #在判别器前向传播过程，输入的假数据被 detach 。这个数据和生成它的计算图“脱钩”了，即梯度传到它那个地方就停了
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


     
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
```

 在epoch大循环下，遍历已导入的数据集的各张图片。

首先，图片输入生成器之前，我们需要初始化图片标签。真图片label=1，假图片label=0。

每个batch的开头首先要执行optimizer_G.zero_grad()：

因为根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑          问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。

 z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))) 加入随机噪声，这是生成器输入的预处理。

接着生成一批假图，用真实的标签和假样本计算G_Loss.

接着反向传播Gloss，更新生成器参数。

后续接着训练判别器网络。

上述代码先更新生成器参数，再更新判别器参数。

为了更新生成器参数，用生成器的损失函数计算梯度，然后反向传播，传播图中经过了discriminator，根据链式法则，不得不顺带计算一下判别器的参数梯度，虽然在这一步不会更新判别器参数。反向传播过后，noise 到 fake image 再到 discriminator 的输出这个前向传播的计算图就被释放掉了，后面也不会再用到。

接着更新判别器参数，此时注意到，我们输入判别器的是两部分，一部分是真实数据，另一部分是生成器的输出，也就是假数据。注意观察细节，在判别器前向传播过程，输入的假数据被 detach 了：discriminator(gen_imgs.detach())，detach 的意思是，这个数据和生成它的计算图“脱钩”了，即梯度传到它那个地方就停了，不再继续往前传播（实际上也不会再往前传播了，因为 generator 的计算图在第一次反向传播过后就被释放了）。因此，判别器梯度反向传播，就到它自己身上为止。

这种模式的好处是通过先更新 generator ，使更新后的前向传播计算图可以放心被销毁，因此不用保留计算图占用内存。同时，在更新 discriminator 的时候，也不会计算冗余的 generator 的梯度。缺点是，在 discriminator 上，对 generator 的输出算了两次前向传播，第二次又产生了新的计算图（但比第一次的小）。