dataset: 'cc3m_gain'
ann_root: '../processed_data/InfoGrowth_400k_samples.json'
lmdb_root: '/PATHTOCC3M/cc3m/lmdb_train'

# size of vit model; base or large
vit: 'base'
vit_grad_ckpt: False
vit_ckpt_layer: 0

image_size: 224
batch_size: 40

queue_size: 57600
alpha: 0.4

# optimizer
decay_method: step
weight_decay: 0.05
init_lr: 3e-4
min_lr: 1e-6
warmup_lr: 1e-6
lr_decay_rate: 0.9
max_epoch: 20
warmup_steps: 3000