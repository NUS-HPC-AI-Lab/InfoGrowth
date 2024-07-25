<h2 align="center">Dataset Growth</h2>
<p align="center"><b>ECCV 2024</b> | <a href="https://arxiv.org/abs/2405.18347">[Paper]</a> | <a href="https://github.com/NUS-HPC-AI-Lab/InfoGrowth">[Code]</a> </p>

InfoGrowth is an efficient online algorithm to deal with growing web data. It provides cleanness and diversity awareness on the dataset.
For BLIP training on CC3M, it can provides a 14x acceleration with data reduction together with efficient sampling.

https://github.com/user-attachments/assets/5496aacc-f50e-494f-9dc0-87709cd2ad7a

## InfoGrowth and Processed Data
Algorithm code coming soon. 

Image and captions are selected in json format. We provide our cleaned 400k samples in processed_data.

## Experiments

### Download Data/Model

Need to prepare CC3M dataset and BLIP encoders. 

#### Download CC3M
Refer to https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md

#### Download Model Checkpoint
Will be automatically prepared by code. Be sure to have internet connection.

### Preprocessing

We introduce lmdb to accelerate data loading. It need a preprocessing as follows: 

```shell
python scripts/cc3m_lmdb_writer.py --image_root your_path
```

### Pretrain

```shell
python3 -m torch.distributed.run --nnodes 2 --nproc_per_node 8 --master_port 12365 train/pretrain_gain.py --config train/configs/pretrain_cc3m_gain.yaml
```

### Eval
To evalutate pretrained model on COCO, execute the following commands with substitution to your path.
```shell
TEST_CKPT=/path/to/test_checkpoint.pth
python3 -m torch.distributed.run --nproc_per_node=8 train/train_retrieval.py \ 
    --config ./train/configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco \
    --pretrained $TEST_CKPT \
    --evaluate
```

## Citation
```bibtex
@inproceedings{qin2024datasetgrowth,
      title={Dataset Growth}, 
      author={Ziheng Qin and Zhaopan Xu and Yukun Zhou and Zangwei Zheng and Zebang Cheng and Hao Tang and Lei Shang and Baigui Sun and Xiaojiang Peng and Radu Timofte and Hongxun Yao and Kai Wang and Yang You},
      booktitle={ECCV},
      year={2024}
}
```