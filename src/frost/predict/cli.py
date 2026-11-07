"""User-facing ``frost train``, ``frost predict`` and ``frost features`` commands."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from frost.predict.config import DEFAULTS, resolve
from frost.predict.data import column, convert_features, ids, read_csv, times
from frost.predict.model import load, score, train


def _columns(parser):
    parser.add_argument("--src-col", help="Source column (auto: src, src_id, u).")
    parser.add_argument("--dst-col", help="Destination column (auto: dst, dst_id, i).")
    parser.add_argument("--time-col", help="Time column (auto: time, timestamp, ts).")


def _queries(parser):
    parser.add_argument("--src", help="Source node ID for candidate queries.")
    parser.add_argument(
        "--candidates",
        help="Comma-separated destination IDs; a trailing comma is accepted.",
    )
    parser.add_argument(
        "--test", help="CSV of query pairs, optionally with timestamps."
    )
    parser.add_argument(
        "--time",
        help="Query time; overrides test times. Omit to use each row's time or the training cutoff.",
    )
    parser.add_argument(
        "--predictions", help="Write scores to a new CSV instead of stdout."
    )
    parser.add_argument(
        "--device",
        default=DEFAULTS["device"],
        help="PyTorch device, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULTS["threads"], help="PyTorch CPU threads."
    )


def parser():
    root = argparse.ArgumentParser(
        prog="frost",
        description="Use temporal GNNs to predict interactions in your data.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser(
        "train",
        help="Train TGN on an events CSV; optionally score queries immediately.",
    )
    fit.add_argument("events", nargs="?")
    fit.add_argument("--config", help="YAML overrides for the starter configuration.")
    fit.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved settings without training.",
    )
    fit.add_argument("--model", choices=("tgn", "tgat"))
    fit.add_argument("--time-dim", type=int)
    fit.add_argument("--heads", type=int)
    fit.add_argument("--dropout", type=float)
    fit.add_argument("--output", default="frost-model.pt", help="New checkpoint path.")
    fit.add_argument("--node-features", help="Numeric NPY feature matrix.")
    fit.add_argument("--feature-ids", help="JSON list of node IDs in NPY row order.")
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--batch-size", type=int)
    fit.add_argument("--fanout", type=int)
    fit.add_argument("--hidden", type=int)
    fit.add_argument("--lr", type=float)
    fit.add_argument("--seed", type=int)
    _columns(fit)
    _queries(fit)
    fit.set_defaults(**{key: None for key in DEFAULTS})
    predict = commands.add_parser(
        "predict", help="Score candidates or a test CSV using a saved checkpoint."
    )
    predict.add_argument("checkpoint")
    predict.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    _columns(predict)
    _queries(predict)
    features = commands.add_parser(
        "features",
        help="Convert numeric/categorical node attributes to an aligned NPY.",
    )
    features.add_argument("csv")
    features.add_argument("--id-col", default="n_id")
    features.add_argument("--output", required=True)
    features.add_argument("--categorical-dim", type=int, default=64)
    return root


def _query_rows(args, checkpoint):
    if args.test:
        frame = read_csv(args.test)
        sources = ids(frame[column(frame, args.src_col, ("src", "src_id", "u"))])
        destinations = ids(frame[column(frame, args.dst_col, ("dst", "dst_id", "i"))])
        time_col = args.time_col or next(
            (c for c in ("time", "timestamp", "ts") if c in frame), None
        )
        if args.time is None and time_col:
            timestamps, _ = times(
                frame[column(frame, time_col, ())], checkpoint["time_kind"]
            )
        else:
            timestamps = np.full(len(sources), checkpoint["last_time"])
    else:
        sources = [args.src.strip()]
        destinations = [
            value.strip() for value in args.candidates.split(",") if value.strip()
        ]
        if not destinations:
            raise ValueError("Provide at least one candidate ID.")
        sources *= len(destinations)
        timestamps = np.full(len(sources), checkpoint["last_time"])
    if args.time is not None:
        value, _ = times([args.time], checkpoint["time_kind"])
        timestamps = np.full(len(sources), value[0])
    return sources, destinations, timestamps


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    try:
        if args.command == "features":
            shape = convert_features(
                args.csv, args.output, args.id_col, args.categorical_dim
            )
            print(
                f"Saved {args.output}: {shape[0]} nodes × {shape[1]} features",
                file=sys.stderr,
            )
            return
        if args.command == "train":
            settings = resolve(
                args.config, **{key: getattr(args, key) for key in DEFAULTS}
            )
            for key, value in settings.items():
                setattr(args, key, value)
            if args.show_config:
                print(yaml.safe_dump(settings, sort_keys=False), end="")
                return
            if not args.events:
                raise ValueError("Provide an events CSV or use --show-config.")
        if args.threads < 1 or args.batch_size < 1:
            raise ValueError("Threads and batch size must be positive.")
        if bool(args.src) != bool(args.candidates):
            raise ValueError("Use --src and --candidates together.")
        if args.test and args.src:
            raise ValueError("Choose --test or --src/--candidates, not both.")
        has_query = bool(args.test or args.src)
        if not has_query and (
            args.command == "predict" or args.time or args.predictions
        ):
            raise ValueError("Provide --test or --src with --candidates.")
        if args.predictions and Path(args.predictions).exists():
            raise ValueError("Predictions file already exists; choose a new path.")
        torch.set_num_threads(args.threads)
        if args.command == "train":
            train(
                args.events,
                output=args.output,
                node_features=args.node_features,
                feature_ids=args.feature_ids,
                src_col=args.src_col,
                dst_col=args.dst_col,
                time_col=args.time_col,
                epochs=args.epochs,
                batch_size=args.batch_size,
                fanout=args.fanout,
                hidden=args.hidden,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                model=args.model,
                time_dim=args.time_dim,
                heads=args.heads,
                dropout=args.dropout,
                threads=args.threads,
            )
            checkpoint_path = args.output
        else:
            checkpoint_path = args.checkpoint
        if has_query:
            checkpoint, model, history = load(checkpoint_path, args.device)
            src, dst, ts = _query_rows(args, checkpoint)
            scores = score(checkpoint, model, history, src, dst, ts, args.batch_size)
            stream = (
                open(args.predictions, "x", newline="")
                if args.predictions
                else sys.stdout
            )
            try:
                writer = csv.writer(stream)
                writer.writerow(["src", "dst", "time", "score"])
                writer.writerows(zip(src, dst, ts, scores))
            finally:
                if args.predictions:
                    stream.close()
    except (ValueError, OSError, RuntimeError, KeyError, yaml.YAMLError) as exc:
        arg_parser.error(str(exc))


if __name__ == "__main__":
    main()
