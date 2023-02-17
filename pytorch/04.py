import torch 
x=torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
X=torch.tensor([12,45])
Y=torch.tensor([3,90])
print(X>Y)