CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4736 train_PBAFN_stage1.py \
--name PBAFN_stage1 \
--resize_or_crop scale_width \
--verbose --tf_log \
--lr 0.0001 \
--batchSize 16 \
--num_gpus 1 \
--label_nc 14 \
--launcher pytorch

