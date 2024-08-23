# Semantic Segmentation in Cataract Surgical Scenes Using Test Time Adaptation
The repository is the official pytorch implementation of paper **Semantic Segmentation in Cataract Surgical Scenes Using Multi-View Test Time Adaptation**. 

## Prerequsite
### Dataset
![](https://github.com/liamheng/CAI-algorithms/blob/main/figs/anno_samples.png)

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
For source pretraining phase of MUTA model, run the following command:

```sh
# source_muta
python exp.py --exp_dir projects/muta/source_pretraining --config source_muta.yaml
```

Alternatively, we provide a script to pretrain deeplabv3 as the baseline:
```sh
# source_deeplabv3
python exp.py --exp_dir projects/muta/source_pretraining --config source_deeplabv3.yaml
```

## Adaptation
As mentioned in paper, we provide solutions for both domain adaptation scenarios without source.

To conduct **Test Time Adaptation(TTA)**, where each sample is adapted individually, run the following command:

```sh
# tta_muta
python exp.py --exp_dir projects/muta/target_adaptation --config tta_muta.yaml
```

For **Source Free Domain Adaptation(SFDA)**, where samples from target domain are adapted all together, run the command:

```sh
# sfda_muta
python exp.py --exp_dir projects/muta/target_adaptation --config sfda_muta.yaml
```

Similar to the source pretraining procudure, we provide the deeplabv3 versions for TTA and SFDA respectively:

```sh
# tta_deeplanv3
python exp.py --exp_dir projects/muta/target_adaptation --config tta_deeplabv3.yaml

# sfda_deeplabv3
python exp.py --exp_dir projects/muta/target_adaptation --config sfda_deeplabv3.yaml
```
