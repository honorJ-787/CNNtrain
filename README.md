`train.py`  
  Train WideResNet from scratch on CIFAR-10/100 and save a checkpoint  
  → used to obtain our own pretrained model (e.g. `wrn_cifar_best.pth`).  

  从零开始在 CIFAR-10/100 上训练 WideResNet，并保存 checkpoint，  
  → 用于生成**自训练预训练权重**（如 `wrn_cifar_best.pth`）。  

- `wideresnet.py`  
  Implementation of WideResNet-XX-k with residual **BasicBlock** and **NetworkBlock**.  
  This is the shared **supervised backbone** used by both training and transfer learning.

  WideResNet-XX-k 的网络结构实现（残差 BasicBlock + NetworkBlock）。  
  在本项目中作为**监督学习主干网络**，同时被 `train.py` 和 `transferlearning_train.py` 调用。  

- `transferlearning_train.py`  
  Fine-tune WideResNet on CIFAR with **transfer learning**.  
  It loads a pretrained checkpoint (e.g. the one produced by `train.py`),  
  uses stronger data augmentation (RandomErasing), and trains with a smaller LR.

  利用 **迁移学习** 对 WideResNet 进行微调。  
  从预训练 checkpoint（例如 `train.py` 训练得到的模型）加载权重，  
  使用更强的数据增强（RandomErasing），并采用更小的学习率进行微调。
