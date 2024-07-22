<h2 align="center">Dataset Growth</h2>
<p align="center"><b>ECCV 2024</b> | <a href="https://arxiv.org/abs/2405.18347">[Paper]</a> | <a href="https://github.com/NUS-HPC-AI-Lab/InfoGrowth">[Code]</a> </p>

InfoGrowth is an efficient online algorithm to deal with growing web data. It provides cleanness and diversity awareness on the dataset.

https://github.com/user-attachments/assets/5496aacc-f50e-494f-9dc0-87709cd2ad7a

## InfoGrowth and Processed Data


Image and captions are selected in json format. We provide our cleaned 400k samples in processed_data.

## Experiments

###Download Data/Model
Need to prepare CC3M dataset and BLIP encoders. TO BE UPDATED.

###Preprocessing
We introduce lmdb to accelerate data loading. It need a preprocessing as follows: UPDATING SOON

[//]: # (```shell)

[//]: # (TO BE UPDATED)

[//]: # (```)

###Pretrain
```shell
python3 -m torch.distributed.run --nnodes 2 --nproc_per_node 8 --master_port 12365 train/pretrain_gain.py --config train/configs/pretrain_cc3m_gain.yaml
```

###Eval
TO BE UPDATED


## Citation
@inproceedings{qin2024datasetgrowth,
      title={Dataset Growth}, 
      author={Ziheng Qin and Zhaopan Xu and Yukun Zhou and Zangwei Zheng and Zebang Cheng and Hao Tang and Lei Shang and Baigui Sun and Xiaojiang Peng and Radu Timofte and Hongxun Yao and Kai Wang and Yang You},
      booktitle={ECCV},
      year={2024}
}
