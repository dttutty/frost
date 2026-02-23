CUDA_VISIBLE_DEVICES=2 TORCH_DISTRIBUTED_DEBUG=DETAIL \
~/miniconda3/envs/simple_py310/bin/python \
~/miniconda3/envs/simple_py310/bin/torchrun \
--nproc_per_node=2 \
--master_addr=127.0.0.1 \
--master_port=29500 \
train_dist.py \
--data <your_dataset> \
--config ~/Projects/lei_pipe/config/dist/TGN.yml
