# Adapted from lxd99/TPNet and lxd99/TGB_TPNet; see NOTICE.md and LICENSE.
"""TPNet-only arguments with the original classic/TGB defaults."""

import argparse


def get_args(argv: list[str], *, is_tgb: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FROST TPNet trainer")
    parser.add_argument(
        "--prefix", type=str, default="test", help="name of the experiment"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset to be used",
        default="tgbl-wiki" if is_tgb else "wikipedia",
        choices=["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-comment", "tgbl-flight"]
        if is_tgb
        else [
            "wikipedia",
            "reddit",
            "mooc",
            "lastfm",
            "myket",
            "enron",
            "SocialEvo",
            "uci",
            "Flights",
            "CanParl",
            "USLegis",
            "UNtrade",
            "UNvote",
            "Contacts",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument("--model_name", default="TPNet", choices=["TPNet"])
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="number of neighbors to sample for each node",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="how to sample historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        default=1e-06,
        type=float,
        help="the hyperparameter that controls the sampling preference with time interval, a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, it works when sample_neighbor_strategy == time_interval_aware",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of model layers"
    )
    parser.add_argument(
        "--time_feat_dim", type=int, default=100, help="dimension of the time embedding"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="name of optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    if not is_tgb:
        parser.add_argument(
            "--val_ratio", type=float, default=0.15, help="ratio of validation set"
        )
    if not is_tgb:
        parser.add_argument(
            "--test_ratio", type=float, default=0.15, help="ratio of test set"
        )
    parser.add_argument(
        "--num_runs", type=int, default=3 if is_tgb else 5, help="number of runs"
    )
    parser.add_argument(
        "--negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive", "new_random"],
        help="strategy for the negative edge sampling",
    )
    parser.add_argument(
        "--load_best_configs",
        action="store_true",
        default=False,
        help="whether to load the best configurations",
    )
    parser.add_argument(
        "--use_random_projection",
        action="store_true",
        help="whether use the random projection",
    )
    parser.add_argument(
        "--rp_num_layer",
        type=int,
        default=2 if is_tgb else 3,
        help="the max hop of the maintained temporal walk matrices",
    )
    parser.add_argument(
        "--rp_time_decay_weight",
        type=float,
        default=1e-06 if is_tgb else 1e-07,
        help="the time decay weight of the maintained temporal walk matrix",
    )
    parser.add_argument(
        "--rp_dim_factor",
        type=int,
        default=10,
        help="the dim factor of random projections w.r.t. the log(2*edge_num)",
    )
    parser.add_argument(
        "--encode_not_rp",
        action="store_true",
        help="whether to use pairwise features in encoder",
    )
    parser.add_argument(
        "--decode_not_rp",
        action="store_true",
        help="whether to use pairwise features in link decoder",
    )
    parser.add_argument(
        "--rp_not_scale",
        action="store_true",
        help="whether to scale and relu for inner product of random projections",
    )
    parser.add_argument(
        "--not_encode",
        action="store_true",
        help="whether to use encoder to learn node embeddings from historical interactions",
    )
    parser.add_argument(
        "--enforce_dim",
        type=int,
        default=-1,
        help="whether to specific the dimension of random projections",
    )
    parser.add_argument(
        "--rp_use_matrix",
        action="store_true",
        help="whether to explicitly maintain temporal walk matrices",
    )
    if is_tgb:
        parser.add_argument(
            "--output_dim",
            type=int,
            default=172,
            help="dimension of the output embedding",
        )
    if is_tgb:
        parser.add_argument(
            "--train_neg_num",
            type=int,
            default=1,
            help="the number of negative edge per postive edge in training",
        )
    if is_tgb:
        parser.add_argument(
            "--train_negative_sample_strategy",
            type=str,
            default="random",
            choices=["random", "historical", "inductive", "new_random"],
            help="strategy for the negative edge sampling",
        )
    if is_tgb:
        parser.add_argument(
            "--train_loss_type",
            type=str,
            default="pointwise",
            choices=["pointwise", "listwise"],
            help="the loss function type of training",
        )
    if is_tgb:
        parser.add_argument(
            "--not_embedding",
            action="store_true",
            help="whether to use the embedding model in TPNet",
        )
    args = parser.parse_args(argv)
    if args.load_best_configs:
        _best_configs(args, is_tgb=is_tgb)
    return args


def _best_configs(args: argparse.Namespace, *, is_tgb: bool) -> None:
    if not is_tgb:
        if args.dataset_name == "CanParl":
            args.sample_neighbor_strategy = "uniform"
        if args.use_random_projection:
            if args.dataset_name in ["Contacts"]:
                args.rp_time_decay_weight = 0.0001
            elif args.dataset_name in ["UNvote", "CanParl", "mooc"]:
                args.rp_time_decay_weight = 1e-05
            elif args.dataset_name in [
                "wikipedia",
                "reddit",
                "enron",
                "SocialEvo",
                "Flights",
                "USLegis",
            ]:
                args.rp_time_decay_weight = 1e-06
            elif args.dataset_name in ["lastfm", "uci", "UNtrade"]:
                args.rp_time_decay_weight = 1e-07
            else:
                raise ValueError("Not Recognized Dataset!")
    if is_tgb:
        args.rp_num_layer = 2
        if args.use_random_projection:
            if args.dataset_name == "tgbl-review":
                args.rp_time_decay_weight = 1e-07
            else:
                args.rp_time_decay_weight = 1e-06
