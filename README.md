# EfficientDL: Automatic Performance Prediction and Component Recommendation for Deep Learning Systems

Current automatic deep learning (i.e., AutoDL) frameworks rely on training feedback from actual runs, which often hinder their ability to provide quick and clear performance predictions for selecting suitable DL systems. To address this issue, we propose **EfficientDL**, an innovative deep learning board designed for automatic performance prediction and component recommendation.
![框架图](https://github.com/gaoqi647/anonymity/blob/main/demo/framework.png)
**EfficientDL** can quickly and precisely recommend twenty-seven system components and predict the performance of DL models without requiring any training feedback. The magic of no training feedback comes from our proposed comprehensive, multi-dimensional, fine-grained system component dataset, which enables us to develop a static performance prediction model and comprehensive optimized component recommendation algorithm (i.e., αβ-BO search), removing the dependency on actually running parameterized models during the traditional optimization search process. The simplicity and power of **EfficientDL** stem from its compatibility with most DL models.

We are in- ResNet50,MobileNetV3,EfficientNet-B0,MaxViT-T,Swin-B,DaViT-T was tested on 6 models, and the results of EfficientDL are as follows
| **Model**           | **Type**   | **Epoch** | **Data Augmentation**                                                                                      | **Batch size** | **Top-1 accuracy (%)**    |
|---------------------|------------|-----------|-------------------------------------------------------------------------------------------------------------|----------------|--------------------------|
| **ResNet50**         | Original   | 300       | CutMix, ColorJitter, Random resize scale, Lighting, horizontal flip                                          | 256            | 78.32                     |
|                     | Our        | 390       | horizontal flip, random resized crop, ColorJitter, mixup, cutmix                                             | 192            | **78.57**          |
| **MobilenetV3**      | Original   | 300       | AutoAugment                                                                                                 | 4096           | 75.20                     |
|                     | Our        | 450       | random resized crop, Random erase, mixup                                                                     | 256            | **75.43**         |
| **EfficientNet-B0**  | Original   | 350       | AutoAugment                                                                                                 | 2048           | 77.11                     |
|                     | Our        | 450       | AutoAugment, Random resize scale, Random resize aspect ratio, horizontal flip, Color jitter, Random erase, mixup | 512            | **77.80**          |
| **MaxViT-T**         | Original   | 300       | Center crop, RandAugment, Mixup                                                                             | 4096           | 83.62                     |
|                     | Our        | 290       | Random resize scale, Random resize aspect ratio, horizontal flip, color jitter, mixup, cutmix, Random erase  | 128            | **84.01**         |
| **Swin-B**           | Original   | 300       | AutoAugment, color jitter, cutmix, mixup, Random erase                                                      | 1024           | 83.11                     |
|                     | Our        | 400       | AutoAugment, horizontal flip, mixup, color jitter, Random erase                                              | 256            | **83.24**         |
| **DaViT-T**          | Original   | 300       | Random resize scale, horizontal flip, Color jitter, AutoAugment, Random erase, mixup, cutmix                 | 2048           | 82.12                     |
|                     | Our        | 340       | horizontal flip, vertical flip, Random resize aspect ratio, Random resize scale, Color jitter, AutoAugment, Random erase, mixup, cutmix | 128            | **82.34**           |


## Directory Structure
<ul>
<li>We store our data preprocessing code dataprocess.py in the dataset folder.</li>
<li>We store data such as encoding, decoding, and normalization in the data folder.</li> 
<li>In the demo folder, we have the code acquisition_Functions.py of the collected function, all the Bayesian optimized code, and provide a bayesian_optimisation.py of test examples.</li>
<li>We store the code for the ablation experiment in the Ablation folder.</li>
</ul>
The distribution of each image classification dataset in the dataset is shown in the following figure.

![数据分布](https://github.com/gaoqi647/anonymity/blob/main/dataset/dataset.png)

Our pre-trained model and the pre-trained model after we recommend 6 examples are in the Baidu network disk link below.
https://pan.baidu.com/s/1mftecYgKK5np1sxtBstoHw?pwd=7hnp 

You can test the results of the pretrained model with the following code
### Requirements
- Python3
- PyTorch (> 1.0)
- torchvision (> 0.2)
- NumPy

```
python test.py \
--net_type resnet \
--dataset imagenet \
--batch_size 64 \
--depth 50 \
--pretrained /set/your/model/path/model_best.pth.tar
```
