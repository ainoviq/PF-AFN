python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 train_RMGN_e2e.py --name RMGN_e2e   \
--PFAFN_warp_checkpoint 'checkpoints/PFAFN_stage1/PFAFN_warp_epoch_201.pth'  \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_warp_epoch_101.pth' \
--PBAFN_gen_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_gen_epoch_101.pth'  \
--lr 0.00003 --niter 100 --niter_decay 100 --resize_or_crop scale_width \
--verbose --tf_log \
--batchSize 2 \
--num_gpus 1 \
--label_nc 14 \
--predmask --multilevel \
--launcher pytorch

