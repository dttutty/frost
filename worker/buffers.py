# # worker/buffers.py
# import torch


# # -------------------- feature buffer helpers --------------------
# def alc_feat_buf_w_dedup(sample_param, bs, nfeat, efeat):
#     nbuf = None
#     ebuf = None
#     limit = int(bs * 3.3)
#     if sample_param.neighbor is not None and efeat is not None:
#         edge_space = 0
#         for i in sample_param.neighbor:
#             limit = limit * (i + 1)
#             edge_space += limit * sample_param.history
#         ebuf = torch.zeros((edge_space, efeat.shape[1]), pin_memory=True)

#     if nfeat is not None:
#         nbuf = torch.zeros(
#             (limit * sample_param.history, nfeat.shape[1]), pin_memory=True
#         )
#     return nbuf, ebuf


# def alc_feat_buf_no_dedup(sample_param, bs, nfeat, efeat):
#     pinned_nfeat_buffs, pinned_efeat_buffs = [], []
#     limit = int(bs * 3.3)
#     if sample_param.neighbor is not None:
#         for i in sample_param.neighbor:
#             limit *= i + 1
#             if efeat is not None:
#                 for _ in range(sample_param.history):
#                     pinned_efeat_buffs.insert(
#                         0, torch.zeros((limit, efeat.shape[1]), pin_memory=True)
#                     )
#     if nfeat is not None:
#         for _ in range(sample_param.history):
#             pinned_nfeat_buffs.insert(
#                 0, torch.zeros((limit, nfeat.shape[1]), pin_memory=True)
#             )
#     return pinned_nfeat_buffs, pinned_efeat_buffs
