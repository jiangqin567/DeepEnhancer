# DeepEnhancer: Temporally Consistent Focal Transformer for Comprehensive Video Enhancement

https://github.com/submition/DeepEnhancer/assets/160112990/d2c4dd8c-2f46-4cfd-96a4-210718bc3027

Create a folder ```./pretrained_models```
```
mkdir pretrained_models
```
Put the optical flow estimation model weights ```spynet_20210409-c6c1bd09.pth``` in ```./pretrained_models```.

To train a model, remember to modify the config file following the example ```config_example/config.yaml```.

> *NOTE*: 
>  Modify both "train.dataroot_gt" and "train.dataroot_lq" into the path of clean training frame since the degradation is generated on-the-fly.
>
>  Modify "val.dataroot_gt" and "val.dataroot_lq" to the path of validation video clips.
>
>  Set "texture_template" to the path where you download the scratch templates.


### Test

We provide the [pre-trained models](link：https://pan.baidu.com/s/1S5132HNEAsTGB1cQuPeuYQ 
code：dhcm)


To restore the old films, please run
```
CUDA_VISIBLE_DEVICES=0 python test_demo.py
```
The restored results could be found in ```./visual_restore_results``` folder.

