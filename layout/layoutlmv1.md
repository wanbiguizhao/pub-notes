


## 嵌入层参数

### word嵌入
简单：把单词映射成为指定维度的向量 
### position_embeddings
这个需要看一下代码
### x_position_embeddings
x坐标嵌入
### y_position_embeddings 

### h_position_embeddings

### w_position_embeddings

### token_type_embeddings
这个还涉及词的类型吗？比如说动词，Ner的类型？
需要看一下代码。
## forward 函数
### input_ids
(batch_size,token_size[可能存在填充情况],)
```
if position_ids is None:
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=input_ids.device
    )
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

```
需要再看一下input_ids的数据格式
```
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
```

### dropout 和 LayerNorm
`TODO` 有什么作用？