CUDA_VISIBLE_DEVICES=2 TORCH_DISTRIBUTED_DEBUG=DETAIL \
/home/sqp17/miniconda3/envs/simple_py310/bin/python \
/home/sqp17/miniconda3/envs/simple_py310/bin/torchrun \
--nproc_per_node=2 \
--master_addr=127.0.0.1 \
--master_port=29500 \
train_dist.py \
--data <your_dataset> \
--config /home/sqp17/Projects/tgl/config/dist/TGN.yml
