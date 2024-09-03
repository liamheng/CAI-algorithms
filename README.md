# Multi-view Test-time Adaptation for Semantic Segmentation in Clinical Cataract Surgery
The repository is the official pytorch implementation of paper **Multi-view Test-time Adaptation for Semantic Segmentation in Clinical Cataract Surgery**. 

## Prerequsite
### Dataset
![](https://github.com/liamheng/CAI-algorithms/blob/main/figs/dataset_overview_github.png)

**The annotated CatInstSeg and CatS can be found at:**

- CatInstSeg: [](https://github.com/liamheng/CAI-algorithms/blob/main/storage/datasets/CatInstSeg)
- CatS: [](https://github.com/liamheng/CAI-algorithms/blob/main/storage/datasets/CatS)

**Dataset Preprocessing:** [Category Standardization](https://github.com/liamheng/CAI-algorithms/blob/main/Category%20Standardization.pdf)

### Package dependency

- easydict==5.1.1
- imageio==2.34.2
- matplotlib==3.8.4
- numpy==1.26.4
- pillow==10.4.0
- pytorch==2.3.1
- torchvision==0.18.1
- tqdm==4.66.4
- yaml==0.2.5

## Source Pretraining

**We provide the checkpoints of pretrain model for convenience:**

[Deeplabv3_ResNet50](https://www.dropbox.com/scl/fi/c4gd9jw8tq471o5jxu95l/deeplabv3_cadis.pth?rlkey=pqotfmc5cpb7crklbawvz0dl3&st=xgoxvc61&dl=0)


For source pretraining phase of MUTA model, run the following command:

```sh
# source_muta
bash storage/scripts/muta_source_cadis.sh # training on dataset CaDIS
bash storage/scripts/muta_source_cadis_catinstseg.sh # testing on dataset CatInstSeg
bash storage/scripts/muta_source_cadis_cats.sh # testing on dataset CatS
```

Alternatively, we provide a script to pretrain deeplabv3 as the baseline:
```sh
# source_deeplabv3
bash storage/scripts/deeplabv3_cadis.sh # training on dataset CaDIS
bash storage/scripts/deeplabv3_cadis_catinstseg.sh # testing on dataset CatInstSeg
bash storage/scripts/deeplabv3_cadis_cats.sh # testing on dataset CatS
```

## Adaptation
As mentioned in paper, we provide solutions for both domain adaptation scenarios without source.

To conduct **Test Time Adaptation(TTA)**, where each sample is adapted individually, run the following command:

```sh
# tta_muta
bash storage/scripts/muta_tta_cadis2catinstseg.sh # tta from CaDIS to CatInstSeg
bash storage/scripts/muta_tta_cadis2cats.sh # tta from CaDIS to CatS
```

For **Source Free Domain Adaptation(SFDA)**, where samples from target domain are adapted all together, run the command:

```sh
# sfda_muta
bash storage/scripts/muta_sfda_cadis2catinstseg.sh # sfda from CaDIS to CatInstSeg
bash storage/scripts/muta_sfda_cadis2cats.sh # sfda from CaDIS to CatS
```

Similar to the source pretraining procudure, we provide the deeplabv3 versions for TTA and SFDA respectively:

```sh
# tta_deeplabv3
bash storage/scripts/deeplanv3_tta_cadis2catinstseg.sh # tta from CaDIS to CatInstSeg
bash storage/scripts/deeplanv3_tta_cadis2cats.sh # tta from CaDIS to CatS

# sfda_deeplabv3
bash storage/scripts/deeplanv3_sfda_cadis2catinstseg.sh # sfda from CaDIS to CatInstSeg
bash storage/scripts/deeplanv3_sfda_cadis2cats.sh # sfda from CaDIS to CatS
```
