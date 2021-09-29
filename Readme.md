## UDC-Net
This is a implementation of the paper "Dual-Consistency Semi-Supervised Learning with Uncertainty Quantification for COVID-19 Lesion Segmentation from CT Images
".

```
@article{li2021dual,
  title={Dual-Consistency Semi-Supervised Learning with Uncertainty Quantification for COVID-19 Lesion Segmentation from CT Images},
  author={Li, Yanwen and Luo, Luyang and Lin, Huangjing and Chen, Hao and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2104.03225},
  year={2021}
}
```

###Lisence

###Installation
This repository is based on Pytorch 1.3.0

###Usage
####1. Clone the repository:
```angular2html
git clone https://github.com/poiuohke/UDC-Net
cd UDC-Net
```

####2. change the data path and hyper-parameters in ./configs/config.json

####3. Train the model:
```angular2html
python train.py
```
####4. Inference
```angular2html
python inference.py --config ./configs/config.json --model trained_model_path --data_path test_data_path --mask_path test-mask_path
```
