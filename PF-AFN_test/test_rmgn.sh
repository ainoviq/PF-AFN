python test_rmgn.py --name demo --resize_or_crop None \
--warp_checkpoint 'checkpoints/RMGN_e2e/PFAFN_warp_epoch_201.pth' \
--gen_checkpoint 'checkpoints/RMGN_e2e/PFAFN_gen_epoch_201.pth' \
--batchSize 1 --gpu_ids 0 --predmask --multilevel
