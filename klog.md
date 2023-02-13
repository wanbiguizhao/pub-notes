### 2023年02月13日 
### 本地安装paddle遇到问题

#### import paddle 问题
本地操作系统ubuntu22.04， cuda 11.6 
分别尝试了 paddle 2.32 和paddle 2.41 
安装成功，但是import paddle
```
ImportError: /home/liukun/anaconda3/envs/paddle_env/lib/python3.7/site-packages/paddle/fluid/core_avx.so: undefined symbol: _dl_sym, version GLIBC_PRIVATE
```
当前需要尝试安装cuda11.2对应的环境。

分析可能的原因：操作系统版本的问题？因为cuda11.6 只提供ubuntu20.04 的下载链接。


疑问：conda 安装cuda和系统自己的cuda的冲突吗？

解决方法，最终在系统执行：sudo apt install nvidia-cuda-toolkit 
猜测应该是为了在系统安装NVIDIA的cuda11.6在系统卸载了所有NVIDIA的安装包造成的。
应该是一个目录问题造成。

#### paddle.utils.run_check() 

刚解决玩一个问题，现在又报错了。
 PreconditionNotMetError: Cannot load cudnn shared library. Cannot invoke method cudnnGetVersion.
      [Hint: cudnn_dso_handle should not be null.] (at /paddle/paddle/phi/backends/dynload/cudnn.cc:60)
      [operator < fill_constant > error]

解决尝试一：
重新删除本库，重新安装。


莫非是，之前自己在~/.profile设置的cudahome的问题？
```
CUDA_HOME="/usr/local/cuda-11.8/bin"
```
修改为如下情况：

认真阅读了报错信息后：

  Suggestions:
  1. Check if the third-party dynamic library (e.g. CUDA, CUDNN) is installed correctly and its version is matched with paddlepaddle you installed.
  2. Configure third-party dynamic library environment variables as follows:
  - Linux: set LD_LIBRARY_PATH by `export LD_LIBRARY_PATH=...`


应该是路径问题，paddle运行依赖的一些软件库一般设置在/anaconda3/envs/paddle_env/lib 既可以运行。
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/liukun/anaconda3/envs/paddle_env/lib"
感受：以后还是尽量不要使用NVIDIA的提供的cuda toolkit。

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/liukun/anaconda3/envs/paddle_env/lib"
