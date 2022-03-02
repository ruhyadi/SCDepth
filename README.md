# Depth Estimation

## Installation
Install using conda
```
conda create -n depth python=3.8 numpy
```
Assumming you have install CUDA. Download Pytorch for unsupported GPU (old GPU) from [Nelson Liu](https://cs.stanford.edu/~nfliu/files/pytorch/whl/torch_stable.html). In this case PyTorch 1.8.1 and Torchvision 0.9.1. Install PyTorch and Torchvision with pip:
```
pip install ./torch-1.8.1+cu101-cp38-cp38-linux_x86_64.whl
pip install ./torchvision-0.9.1+cu101-cp38-cp38-linux_x86_64.whl
```
Install requirements:
```
pip install -r requirements.txt
```

## Download Pretrained Model
See [README_MASTER.md](README_MASTER.md)
[**[kitti_scv1_model]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUNoHDuA2FKjjioD?e=fD8Ish):

|  Models  | Abs Rel | Sq Rel | Log10 | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|----------|---------|--------|-------|-------|-----------|-------|-------|-------|
| resnet18 | 0.119   | 0.878  | 0.053 | 4.987 | 0.196     | 0.859 | 0.956 | 0.981 |

 [**[nyu_scv2_model]**](https://1drv.ms/u/s!AiV6XqkxJHE2mUSxFrPz690xaxwH?e=wFOR6A):

|  Models  | Abs Rel | Sq Rel | Log10 | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|----------|---------|--------|-------|-------|-----------|-------|-------|-------|
| resnet18 | 0.142   | 0.112  | 0.061 | 0.554 | 0.186     | 0.808 | 0.951 | 0.987 |

## Inference
Run code:
```bash
python inference.py --config configs/v2/nyu.txt \
  --input_dir demo/input/ \
  --output_dir demo/output/ \
  --ckpt_path ckpts/nyu_scv2/version_10/epoch=101-val_loss=0.1580.ckpt \
  --save-vis --save-depth
```

## References

#### SC-DepthV1:
**Unsupervised Scale-consistent Depth Learning from Video (IJCV 2021)** \
*Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Zhichao Li, Le Zhang, Chunhua Shen, Ming-Ming Cheng, Ian Reid* 
[**[paper]**](https://jwbian.net/Papers/SC_Depth_IJCV_21.pdf)
```
@article{bian2021ijcv, 
  title={Unsupervised Scale-consistent Depth Learning from Video}, 
  author={Bian, Jia-Wang and Zhan, Huangying and Wang, Naiyan and Li, Zhichao and Zhang, Le and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian}, 
  journal= {International Journal of Computer Vision (IJCV)}, 
  year={2021} 
}
```
which is an extension of previous conference version:
**Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (NeurIPS 2019)** \
*Jia-Wang Bian, Zhichao Li, Naiyan Wang, Huangying Zhan, Chunhua Shen, Ming-Ming Cheng, Ian Reid* 
[**[paper]**](https://papers.nips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf)
```
@inproceedings{bian2019neurips,
  title={Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video},
  author={Bian, Jiawang and Li, Zhichao and Wang, Naiyan and Zhan, Huangying and Shen, Chunhua and Cheng, Ming-Ming and Reid, Ian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

#### SC-DepthV2:
**Auto-Rectify Network for Unsupervised Indoor Depth Estimation (TPAMI 2022)** \
*Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Tat-Jun Chin, Chunhua Shen, Ian Reid*
[**[paper]**](https://arxiv.org/abs/2006.02708v2)
```
@article{bian2021tpami, 
  title={Auto-Rectify Network for Unsupervised Indoor Depth Estimation}, 
  author={Bian, Jia-Wang and Zhan, Huangying and Wang, Naiyan and Chin, Tat-Jin and Shen, Chunhua and Reid, Ian}, 
  journal= {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}, 
  year={2021} 
}
```
