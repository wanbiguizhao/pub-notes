
字典变属性
‘‘‘ python
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d
opt=dict2obj(opt)
’’’


## bert 要考虑的几个问题
- bert的轻量化
要训练一些特定领域的bert
### 轻量化的几个方向
- 量化
- 剪枝
- 权重共享
- 知识蒸馏

### 一些参考
https://zhuanlan.zhihu.com/p/436017910 bert家族介绍
https://zhuanlan.zhihu.com/p/349732075 面向中文的bert
https://zhuanlan.zhihu.com/p/391947160 轻量化bert的一些变种