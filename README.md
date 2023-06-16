`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

本项目为*AGTGAN: Unpaired Image Translation for Photographic Ancient Character Generation*的代码实现

#### 创建运行环境

环境依赖见torch18.yaml

#### 下载数据，并创建数据文件夹

将数据下载解压后按照以下路径存放在项目文件夹下：
甲骨文字模数据集：项目文件夹/data/oracle/trian/A or 项目文件夹/data/oracle/test/A
甲骨文拓片数据集：项目文件夹/data/oracle/trian/B or 项目文件夹/data/oracle/test/B


#### 训练模型


执行：

```
python train_agtgan.py --config_file configs/oracle.yaml
```



#### 生成增广样本

执行：

```
python gen_lmdb_oracle.py
```
