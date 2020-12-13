# gazeEstimate

Network/Config.py：提供基本的配置信息（Configure类），和一些常用的方法（Methods类）。 
  `TARGET_ROOT_PATH`为数据集的目录。
  `Epoch`为训练的迭代次数（40）
  `Learning_rate` 学习率（0.001）
  `Batch_size` 每一批训练的样本数目（6）
  
 Network/NetworkModule.py：定义了网络的结构（GazeNet类）

  —— 输出全部归一化(0~1)。
 
 DataProcessor.py：将你之前qt程序标记的数据集（人脸+GT），用AdaBoost检测并得到新的数据集（人脸框，双眼的框，GT）
     Pickle文件：
     例如：1.png': {'face': (245, 105, 219, 219), 'eye1': (285, 158, 54, 54), 'eye2': (369, 158, 56, 56), 'gt': (740, 608)}


train.py：训练我们的网络的脚本。
训练的时候做了shuffle，每一个epoch都打乱一次数据集
     

 
 
