# 域适应 domain adaptation 目标检测

### 参考工作
1. SSDA-YOLO 2023《Semi-supervised domain adaptive YOLO for cross-domain object detection》
    风格迁移+均值教师思想
    优化目标：有标签源域和类目标源域的检测损失 +有标签源域和类目标源域的一致性损失 + 无标签目标域的蒸馏损失 

2. MS-DAYOLO 2021《Multiscale Domain Adaptive YOLO for Cross-Domain Object Detection》
    梯度反转层（对抗思想）域适应网络-->最小化域分类损失，区别源域和目标域，检测主干网络-->最大化损失，学习域不变性特征
    优化目标：有标签源域的检测损失 + 有标签源域和无标签目标域的 域分类损失

```bash
clear-->foggy 
    train：Source -- labeled cityspace, Target -- unlabeled foggy cityspace
    test、val：labeled foggy cityspace
sunnny-->rainy
    train：Source -- labeled sunnny images, Target -- unlabeled rainy images
    test、val：extra labeled images from rainy 
```


3. Image-DA


4. cross-DA


Downloading: "https://download.pytorch.org/models/resnet101-63fe2227.pth" to /home/lenovo/.cache/torch/hub/checkpoints/resnet101-63fe2227.pth

模型权重默认下载这里！
![Alt text](image-2.png)

###  概念
###### 增量目标检测：

###### 零样本学习：即预测一个在训练中从未见过的类;
数据被分为三大类：
`Seen Classes： 用于训练模型的类别 ` 
`Unseen Classes: 未知类别，` 模型需要能够在没有任何特定训练的情况下进行分类的类别。在训练过程中没有使用这些类别的数据。
`Auxiliary Information：辅助信息.`由 Word2Vec 等技术提供的关于所有未见类别的描述、语义信息或嵌入向量。这对于解决 "零样本学习"问题十分必要，因为我们没有未知类别的标注示例。

```text
零样本学习包括训练和推理两个步骤。
在训练过程中，模型学习已标注好的数据样本集。
在推理过程中，模型利用这些知识和辅助信息对一组新的类别进行分类。

简而言之，创建零样本学习分类器就是简单地学习从未知空间（用于图像识别的视觉特征）到已知空间（文本特征或其他类型的辅助信息）的映射函数。
```

###### 少样本学习
我们拥有的样本数量太少，无法训练深度神经网络。少样本学习就是基于这非常有限的样本量进行预测的问题。
![Alt text](image-3.png)

```
从这个角度看，少样本学习的目标与传统图像分类问题有所不同:
传统分类问题的目标是让模型识别训练集中的图像，目标是学习类别之间的区分界限,然后将其泛化到测试集中;
少量学习的目标是学会学习。我们并不指望模型知道什么是老虎或大象。目标是了解物体之间的相似性。训练完成后，我们可以将两张图片作为输入展示给模型，并询问这两张图片是否是同类物体。
```
`Query :` 查询样本 是以前从未见过的属于未知类别。上图中，查询样本是一只兔子。它不属于训练时使用的类别。因此，模型不知道查询样本的类别是什么。为了解决这个问题，我们为模型提供了额外的信息。
`Support Set: ` 是一个很小的样本集，对于训练传统分类模型来说太小了。我们向它展示了支持集，即一组带有图像和相应标签的数据。

模型会将Query与Support Set中的 6 幅图像逐一进行比较。由于我们的模型经过训练，知道两个对象之间的相似性，因此它可以正确地分类出查询图像是兔子。

###### 孪生网络：Siamese Network
识别目标间相似度的网络，处理流程如下图：
![Alt text](image-4.png)
![Alt text](image-5.png)

