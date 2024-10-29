# GeoFormer for Homography Estimation

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-keypoint-detector-and/image-registration-on-fire&#41;]&#40;https://paperswithcode.com/sota/image-registration-on-fire?p=semi-supervised-keypoint-detector-and&#41;)

This is the official source code of our ICCV2023 paper: [Geometrized Transformer for Self-Supervised Homography Estimation](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Geometrized_Transformer_for_Self-Supervised_Homography_Estimation_ICCV_2023_paper.html).

![illustration](./image/fig-model.jpg)

## Environment
We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

``` conda
conda create -n GeoFormer python==3.8 -y
conda activate GeoFormer
git clone git@github.com:bingxinhu/GeoFormer.git
cd GeoFormer
pip install -r requirements.txt
```


```
1.执行单组配准执行下述命令，手动修改RGB,frame Events DVS,以及DVS npy的路径
python h.py  
2.执行单组自动配准参照
python homo.py /data/brain_inspired_dataset_szp/Brain_Inspired_Database/Dataset/Brain_Inspired_Datasets/Add_High/24_05_22_zip/24_05_22/output_15
直接执行output_15上传采集标注的数据包
3.执行批处理参照
. ./b.sh
4.执行单组Homography差距比较参照
python f.py
```
@inproceedings{liu2022SuperRetina,
  title={Geometrized Transformer for Self-Supervised Homography Estimation},
  author={Jiazhen Liu and Xirong Li},
  booktitle={ICCV},
  year={2023}
}
```

## Contact
If you encounter any issue when running the code, please feel free to reach us either by creating a new issue in the GitHub or by emailing

+ zhaifang (zhaifang@tsinghua.edu.cn)

