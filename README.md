`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

本项目为*AGTGAN: Unpaired Image Translation for Photographic Ancient Character Generation*的代码实现

#### 创建运行环境

环境依赖见torch18.yaml, 


#### 训练模型


执行：

```
python train_agtgan.py --config_file configs/oracle.yaml
```



#### 生成增广样本

执行：

```
python gen_lmdb.py
```