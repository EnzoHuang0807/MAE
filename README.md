### Implementation of [*KaiMing He el.al. Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377).

* Forked and referenced from [IcarusWizard/MAE](https://github.com/IcarusWizard/MAE)
* The codebase implements pre-training framework in the paper on CIFAR-10 using a ViT-Tiny.

### Environment
* Simply install the required packages using pip
```
pip install -r requirements.txt
```

### Run

#### MAE Pre-training

```bash
python mae_pretrain.py --train_epoch 300
```

* Set additional options for ablation studies (From Table 1.(c) and Table 1.(f))
```
    --mask_encoder
    --sampling {random,block,grid}
```

#### Train Classifier

* From Scratch
```bash
python train_classifier.py
```
* Fine-tuning from pretrained model
```bash
python train_classifier.py --pretrained_model_path vit-t-mae.pt --output_model_path vit-t-classifier-from_pretrained.pt
```
* Set additional options for ablation studies (From Table 1.(c) and Table 1.(f))
```
    --mask_ratio MASK_RATIO
    --mask_encoder
    --sampling {random,block,grid}
```

See logs by `tensorboard --logdir logs`.

### Result
|Model|Validation Acc|
|-----|--------------|
|ViT-T w/o pretrain|73.86|
|ViT-T w/  pretrain|**84.40**|
