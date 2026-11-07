"""CPU regression cases captured from the pinned upstream TPNet implementations.

Regenerate explicitly with this file's --upstream command (see fixture README).
The test suite itself never imports or downloads an external checkout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from frost.tpnet import train as tpnet_train
from frost.tpnet import config, data, layers, model, sampling, training

FIXTURES = Path(__file__).parent / "fixtures/tpnet"
CASES = [
    "plain",
    "rp",
    "matrix",
    "not_scale",
    "not_encode",
    "encoder_only",
    "decoder_only",
]


def backend(is_tgb: bool, upstream: Path | None = None):
    if upstream is None:
        return SimpleNamespace(
            TPNet=model.TPNet,
            RP=model.RandomProjectionModule,
            Predictor=layers.LinkPredictor_v1,
            Data=data.Data,
            neighbor=sampling.get_neighbor_sampler,
            Negative=sampling.TGBNegativeEdgeSampler
            if is_tgb
            else sampling.NegativeEdgeSampler,
            seed=training.set_random_seed,
        )
    sys.path.insert(0, str(upstream.resolve()))
    sys.path.insert(1, str(upstream.resolve().parent / "TGB"))
    models = importlib.import_module("models.TPNet")
    heads = importlib.import_module("models.modules")
    utils = importlib.import_module("utils.utils")
    loader = importlib.import_module("utils.DataLoader")
    return SimpleNamespace(
        TPNet=models.TPNet,
        RP=models.RandomProjectionModule,
        Predictor=heads.LinkPredictor_v1,
        Data=loader.Data,
        neighbor=utils.get_neighbor_sampler,
        Negative=utils.NegativeEdgeSampler,
        seed=utils.set_random_seed,
    )


def synthetic_data(b):
    return b.Data(
        src_node_ids=np.array([1, 1, 2, 3, 1, 2, 4, 1], dtype=np.int64),
        dst_node_ids=np.array([2, 3, 3, 1, 1, 4, 1, 2], dtype=np.int64),
        node_interact_times=np.array([1, 1, 2, 3, 3, 4, 4, 5], dtype=np.float64),
        edge_ids=np.arange(1, 9),
        labels=np.ones(8),
    )


def tensor_summary(value):
    if value is None:
        return None
    value = value.detach().double().flatten()
    return [
        value.sum().item(),
        value.square().sum().item(),
        *value[:3].tolist(),
        *value[-3:].tolist(),
    ]


def capture_training(b, *, is_tgb: bool, case: str, full: bool = False):
    b.seed(43)
    events = synthetic_data(b)
    sampler = b.neighbor(events, sample_neighbor_strategy="recent", seed=17)
    rp = (
        None
        if case == "plain"
        else b.RP(
            node_num=5,
            edge_num=9,
            dim_factor=2,
            num_layer=2,
            time_decay_weight=0.1,
            device="cpu",
            use_matrix=case == "matrix",
            beginning_time=np.float64(1),
            not_scale=case == "not_scale",
            enforce_dim=3,
        )
    )
    args = dict(
        node_raw_features=np.arange(20).reshape(5, 4) / 20,
        edge_raw_features=np.arange(27).reshape(9, 3) / 27,
        neighbor_sampler=sampler,
        time_feat_dim=2,
        dropout=0.1,
        random_projections=None if case == "decoder_only" else rp,
        num_layers=1,
        num_neighbors=2,
        device="cpu",
    )
    width = 6 if is_tgb else 4
    if is_tgb:
        args.update(output_dim=width, not_embedding=case == "not_embedding")
    backbone = b.TPNet(**args)
    predictor = b.Predictor(
        input_dim1=width,
        input_dim2=width,
        hidden_dim=width,
        output_dim=1,
        random_projections=None if case == "encoder_only" else rp,
        not_encode=case == "not_encode",
    )
    net = tpnet_train.TPNetLinkModel(backbone, predictor)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    result = []
    for epoch in range(2):
        net.train()
        if rp is not None:
            rp.reset_random_projections()
        for start in (0, 4):
            sl = slice(start, start + 4)
            src, dst, ts = (
                events.src_node_ids[sl],
                events.dst_node_ids[sl],
                events.node_interact_times[sl],
            )
            optimizer.zero_grad(set_to_none=True)
            positive, negative = net(src, dst, ts, np.roll(dst, 1))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                positive, torch.ones_like(positive)
            )
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                negative, torch.zeros_like(negative)
            )
            loss.backward()
            grads = {
                name: (
                    p.grad.detach().tolist()
                    if full and p.grad is not None
                    else tensor_summary(p.grad)
                )
                for name, p in net.named_parameters()
                if p.requires_grad
            }
            optimizer.step()
            if rp is not None:
                with torch.no_grad():
                    rp.update(src, dst, ts)
            result.append(
                dict(
                    epoch=epoch,
                    positive=positive.detach().tolist(),
                    negative=negative.detach().tolist(),
                    loss=loss.item(),
                    gradients=grads,
                    parameters={
                        name: p.detach().tolist() if full else tensor_summary(p)
                        for name, p in net.named_parameters()
                        if p.requires_grad
                    },
                    rp=[]
                    if rp is None
                    else [p.detach().tolist() for p in rp.random_projections],
                )
            )
    if rp is not None:
        saved = rp.backup_random_projections()
        rp.reset_random_projections()
        rp.reload_random_projections(saved)
        result.append(
            dict(
                restored=[p.detach().tolist() for p in rp.random_projections],
                time=rp.now_time.item(),
            )
        )
        if is_tgb:
            rp.reset_random_projections(reset_zero=False)
            result.append(
                dict(
                    reset_without_resampling=[
                        p.detach().tolist() for p in rp.random_projections
                    ]
                )
            )
    return result


def capture_sampling(b, *, is_tgb: bool):
    events = synthetic_data(b)
    result = {}
    for strategy in ("recent", "uniform", "time_interval_aware"):
        sampler = b.neighbor(
            events, sample_neighbor_strategy=strategy, time_scaling_factor=0.1, seed=17
        )
        result[strategy] = [
            [
                v.tolist()
                for v in sampler.get_historical_neighbors(
                    node_ids=np.array([1, 2, 3, 4]),
                    node_interact_times=np.array([1, 3, 4, 5]),
                    num_neighbors=3,
                )
            ]
            for _ in range(2)
        ]
    for strategy in (
        ["random", "historical", "inductive", "new_random"]
        if is_tgb
        else ["random", "historical", "inductive"]
    ):
        for seed in (17, None):
            if seed is None and strategy in {"historical", "inductive"}:
                continue
            b.seed(43)
            sampler = b.Negative(
                events.src_node_ids,
                events.dst_node_ids,
                interact_times=events.node_interact_times,
                last_observed_time=2,
                negative_sample_strategy=strategy,
                seed=seed,
            )
            result[f"negative_{strategy}_{seed}"] = [
                list(
                    map(
                        np.ndarray.tolist,
                        sampler.sample(
                            size=4,
                            batch_src_node_ids=events.src_node_ids[4:],
                            batch_dst_node_ids=events.dst_node_ids[4:],
                            current_batch_start_time=3,
                            current_batch_end_time=5,
                        ),
                    )
                )
                for _ in range(2)
            ]
    return result


def assert_close(actual, expected, *, state_stats=False):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_close(
                actual[key],
                expected[key],
                state_stats=state_stats or key in {"gradients", "parameters"},
            )
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_close(a, e, state_stats=state_stats)
    elif isinstance(expected, (float, int)):
        # FP32 reductions and Adam amplify CPU-kernel rounding in gradients
        # and parameter summaries. Keep predictions/losses/RP state tighter.
        rel, absolute = (1e-4, 2e-5) if state_stats else (2e-5, 2e-6)
        assert actual == pytest.approx(expected, rel=rel, abs=absolute)
    else:
        assert actual == expected


def test_snapshot_tolerances_keep_outputs_strict():
    assert_close({"gradients": [0.100015]}, {"gradients": [0.1]})
    with pytest.raises(AssertionError):
        assert_close({"logits": [0.100015]}, {"logits": [0.1]})
    with pytest.raises(AssertionError):
        assert_close({"parameters": [0.101]}, {"parameters": [0.1]})


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("is_tgb", [False, True], ids=["classic", "tgb"])
@pytest.mark.parametrize("case", CASES + ["not_embedding"])
def test_training_matches_pinned_upstream(is_tgb, case):
    if not is_tgb and case == "not_embedding":
        pytest.skip("not_embedding exists only in the TGB variant")
    name = "tgb" if is_tgb else "classic"
    expected = json.loads((FIXTURES / f"{name}.json").read_text())
    assert_close(
        capture_training(backend(is_tgb), is_tgb=is_tgb, case=case), expected[case]
    )


@pytest.mark.parametrize("is_tgb", [False, True], ids=["classic", "tgb"])
def test_sampling_matches_pinned_upstream(is_tgb):
    name = "tgb" if is_tgb else "classic"
    expected = json.loads((FIXTURES / f"{name}.json").read_text())
    assert capture_sampling(backend(is_tgb), is_tgb=is_tgb) == expected["sampling"]


@pytest.mark.parametrize(
    "dataset,is_tgb,layers_count,decay,strategy",
    [
        ("wikipedia", False, 3, 1e-6, "recent"),
        ("CanParl", False, 3, 1e-5, "uniform"),
        ("Contacts", False, 3, 1e-4, "recent"),
        ("lastfm", False, 3, 1e-7, "recent"),
        ("tgbl-wiki", True, 2, 1e-6, "recent"),
        ("tgbl-review", True, 2, 1e-7, "recent"),
    ],
)
def test_best_configs_preserve_variant_defaults(
    dataset, is_tgb, layers_count, decay, strategy
):
    args = config.get_args(
        ["--dataset_name", dataset, "--load_best_configs", "--use_random_projection"],
        is_tgb=is_tgb,
    )
    assert args.model_name == "TPNet"
    assert args.rp_num_layer == layers_count
    assert args.rp_time_decay_weight == decay
    assert args.sample_neighbor_strategy == strategy


def write_classic_data(root, dataset="wikipedia"):
    directory = root / dataset
    directory.mkdir(parents=True)
    frame = pd.DataFrame(
        dict(
            u=np.tile(np.arange(1, 11), 6),
            i=np.tile(np.arange(11, 21), 6),
            ts=np.repeat(np.arange(30), 2),
            idx=np.arange(1, 61),
            label=np.ones(60),
        )
    )
    frame.to_csv(directory / f"ml_{dataset}.csv", index=False)
    np.save(directory / f"ml_{dataset}.npy", np.arange(122).reshape(61, 2))
    np.save(directory / f"ml_{dataset}_node.npy", np.arange(42).reshape(21, 2))


@pytest.mark.parametrize(
    "dataset,convert_time", [("wikipedia", False), ("Contacts", True)]
)
def test_classic_data_root_preserves_padding_and_time_conversion(
    tmp_path, dataset, convert_time
):
    write_classic_data(tmp_path, dataset)
    before_cwd = Path.cwd()
    original_sample = random.sample
    nodes, edges, full, train, val, test, *_ = data.get_link_prediction_data(
        dataset,
        0.15,
        0.15,
        logging.getLogger(__name__),
        convert_time=convert_time,
        processed_data_root=tmp_path,
    )
    assert Path.cwd() == before_cwd
    assert random.sample is original_sample
    assert nodes.shape == (21, 172)
    assert edges.shape == (61, 172)
    assert not nodes[:, 2:].any() and not edges[:, 2:].any()
    assert full.edge_ids.tolist() == list(range(1, 61))
    assert full.node_interact_times[-1] == (29 * 300 if convert_time else 29)
    assert train.node_interact_times.max() < val.node_interact_times.min()
    assert val.node_interact_times.max() < test.node_interact_times.min()
    again = data.get_link_prediction_data(
        dataset,
        0.15,
        0.15,
        logging.getLogger(__name__),
        convert_time=convert_time,
        processed_data_root=tmp_path,
    )
    assert np.array_equal(train.edge_ids, again[3].edge_ids)


def format_fixture(value, level=0):
    """Keep numeric vectors on one line when regenerating reviewable fixtures."""
    if isinstance(value, dict):
        rows = [
            "  " * (level + 1)
            + json.dumps(key)
            + ": "
            + format_fixture(item, level + 1)
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(rows) + "\n" + "  " * level + "}"
    if isinstance(value, list) and value and isinstance(value[0], dict):
        rows = ["  " * (level + 1) + format_fixture(item, level + 1) for item in value]
        return "[\n" + ",\n".join(rows) + "\n" + "  " * level + "]"
    return json.dumps(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--tgb", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    torch.set_num_threads(1)
    b = backend(opts.tgb, opts.upstream)
    result = {
        case: capture_training(b, is_tgb=opts.tgb, case=case, full=opts.full)
        for case in CASES + (["not_embedding"] if opts.tgb else [])
    }
    result["sampling"] = capture_sampling(b, is_tgb=opts.tgb)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(format_fixture(result) + "\n")
