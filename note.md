## ubuntu查看cuda版本
cat /usr/local/cuda/version.json 

## 检查系统环境

python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

## 关于cuda的含义
https://zhuanlan.zhihu.com/p/91334380


### cuda

https://en.wikipedia.org/wiki/CUDA

 CUDA英文全称是Compute Unified Device Architecture，是显卡厂商NVIDIA推出的运算平台。 CUDA™是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。按照官方的说法是，CUDA是一个并行计算平台和编程模型，能够使得使用GPU进行通用计算变得简单和优雅。

 CUDA (or Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for general purpose processing, an approach called general-purpose computing on GPUs (GPGPU). CUDA is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of compute kernels.[1]

 ## 使用nvidia 安装cudatoolkit和conda cuda toolkit有什么区别？

conda安装的cuda toolkit 

f using anaconda to install tensorflow-gpu, yes it will install cuda and cudnn for you in same conda environment as tensorflow-gpu. All you need to install yourself is the latest nvidia-driver (so that it works with the latest CUDA level and all older CUDA levels you use.)

This has many advantages over the pip install tensorflow-gpu method:

Anaconda will always install the CUDA and CuDNN version that the TensorFlow code was compiled to use.
You can have multiple conda environments with different levels of TensorFlow, CUDA, and CuDNN and just use conda activate to switch between them.
You don't have to deal with installing CUDA and cuDNN manaually at the system wide level.
The disadvantage when compared to pip install tensorflow-gpu, is the latest version of tensorflow is added to pypi weeks before Anaconda is able to update the conda recipe and publish their builds of the latest TensorFlow version.

conda 允许安装多个版本的cuda，由conda的cuda调用系统的驱动。
 https://www.cnblogs.com/yhjoker/p/10972795.html


 ## Pytorch 确定所使用的 cuda 版本

https://www.cnblogs.com/yhjoker/p/10972795.html
　若在运行时需要使用 cuda 进行程序的编译或其他 cuda 相关的操作，Pytorch 会首先定位一个 cuda 安装目录( 来获取所需的特定版本 cuda 提供的可执行程序、库文件和头文件等文件 )。具体而言，Pytorch 首先尝试获取环境变量 CUDA_HOME/CUDA_PATH 的值作为运行时使用的 cuda 目录。若直接设置了 CUDA_HOME/CUDA_PATH 变量，则 Pytorch 使用 CUDA_HOME/CUDA_PATH 指定的路径作为运行时使用的 cuda 版本的目录。

　　若上述环境变量不存在，则 Pytorch 会检查系统是否存在固定路径 /usr/local/cuda 。默认情况下，系统并不存在对环境变量 CUDA_HOME 设置，故而 Pytorch 运行时默认检查的是 Linux 环境中固定路径 /usr/local/cuda 所指向的 cuda 目录。 /usr/local/cuda 实际上是一个软连接文件，当其存在时一般被设置为指向系统中某一个版本的 cuda 文件夹。使用一个固定路径的软链接的好处在于，当系统中存在多个安装的 cuda 版本时，只需要修改上述软连接实际指向的 cuda 目录，而不需要修改任何其他的路径接口，即可方便的通过唯一的路径使用不同版本的 cuda. 如笔者使用的服务器中，上述固定的 /usr/local/cuda 路径即指向一个较老的 cuda-8.0 版本的目录。

主要有两种方法，第一种是修改软链接 /usr/local/cuda 所指向的 cuda 安装目录( 若不存在则新建 )，第二种是通过设置环境变量 CUDA_HOME 指向所需使用的 cuda 版本的安装目录。除此之外，还建议将对应 cuda 目录中的可执行文件目录( 形如/home/test/cuda-10.1/bin )加入环境变量 PATH 中。

```
    export CUDA_HOME=/home/test/cuda-10.1/        　　　//设置全局变量 CUDA_HOME
    export PATH=$PATH:/home/test/cuda-10.1/bin/        //在 PATH 变量中加入需要使用的 cuda 版本的路径,使得系统可以使用 cuda 提供的可执行文件，包括 nvcc
```

## 获取 Pytorch 使用的 cuda 版本
```
    >>>import torch
    >>>torch.version.cuda    #输出一个 cuda 版本
```

## conda 重新安装conda。
