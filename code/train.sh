# Pretrain on 8 GPUs with CC3M dataset
ANN_ROOT=/path/to/CC3M_Annotation # annotations json file
LMDB_ROOT=/path/to/CC3M_LMDB # LMDB dataset

python -m torch.distributed.run --nproc_per_node=8 pretrain.py --config ./configs/pretrain.yaml \ 
    --output_dir output/pretrain --lmdb_root $LMDB_ROOT --ann_file $ANN_ROOT 

# Finetune on 8 GPus with Flicker/COCO dataset
PRETRAINED_CKPT=/path/to/pretrained_checkpoint.pth
COCO_ANN=/path/to/coco/annotations/
COCO_LMDB=/path/to/coco/lmdb/
python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
    --config ./configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco --pretrained $PRETRAINED_CKPT --ann_root $COCO_ANN --lmdb_root $COCO_LMDB 

FLICKER_ANN=/path/to/flicker/annotations/
FLICKER_LMDB=/path/to/flicker/lmdb/
python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
    --config ./configs/retrieval_flicker.yaml \
    --output_dir output/retrieval_flicker --pretrained $PRETRAINED_CKPT --ann_root $FLICKER_ANN --lmdb_root $FLICKER_LMDB

# Evaluation on 8 GPUs
TEST_CKPT=/path/to/test_checkpoint.pth
python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
    --config ./configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco \
    --pretrained $TEST_CKPT
    --evaluate