from frost_bench.metrics import (
    parse_dyglib_summary,
    parse_frost_epoch_summaries,
    parse_frost_epoch_wall_summaries,
    parse_frost_rank_throughput,
    parse_pipetgl_epoch_summaries,
    parse_pipetgl_rank_throughput,
    parse_pipetgl_run_summary,
    normalized_progress,
    parse_rank_payloads,
    parse_tgl_rank_throughput,
    parse_tgl_epoch_summaries,
    select_bottleneck_rank,
    sum_memory_curves_by_rank,
    summarize_distribution,
)


def test_parsers_normalize_backend_throughput() -> None:
    frost = parse_frost_rank_throughput(
        "TRAIN_THROUGHPUT_PER_GPU rank=0 events_per_sec_per_gpu=12.5 measured_batches=8"
    )
    tgl = parse_tgl_rank_throughput(
        'APPENDIX_TRAIN_E2E_SUMMARY {"rank": 0, "events_per_sec_per_gpu": 11.0, "interval_count": 7}',
        n_gpu=1,
    )
    dyglib = parse_dyglib_summary(
        'APPENDIX_TRAIN_E2E_SUMMARY {"events_per_sec_per_gpu": 10.0, "measured_batches": 6}'
    )

    assert select_bottleneck_rank(frost) == (0, 12.5, 8)
    assert select_bottleneck_rank(tgl) == (0, 11.0, 7)
    assert dyglib == {"events_per_sec_per_gpu": 10.0, "measured_batches": 6}


def test_pipetgl_parsers_read_throughput_and_convergence() -> None:
    log = "\n".join(
        [
            'APPENDIX_TRAIN_E2E_SUMMARY {"system":"pipetgl","rank":0,"events_per_sec_per_gpu":21.5,"measured_batches":4}',
            'APPENDIX_TRAIN_E2E_SUMMARY {"system":"pipetgl","rank":1,"events_per_sec_per_gpu":20.0,"measured_batches":4}',
            'PIPETGL_EPOCH_SUMMARY {"epoch":1,"train_loss":0.7,"val_ap":0.8,"val_auc":0.75,"test_ap":0.79,"test_auc":0.74,"events":100,"cumulative_events":100,"train_time_s":2.0,"cumulative_train_time_s":2.0,"epoch_wall_time_s":3.0}',
            'PIPETGL_RUN_SUMMARY {"system":"pipetgl","best_epoch":1,"best_val_ap":0.8,"test_ap":0.79,"test_auc":0.74}',
        ]
    )
    assert select_bottleneck_rank(parse_pipetgl_rank_throughput(log)) == (
        1,
        20.0,
        4,
    )
    curve = parse_pipetgl_epoch_summaries(log)
    assert curve[0]["cumulative_events"] == 100
    assert curve[0]["test_ap"] == 0.79
    assert curve[0]["test_auc"] == 0.74
    assert curve[0]["epoch_wall_time_s"] == 3.0
    assert parse_pipetgl_run_summary(log)["test_ap"] == 0.79


def test_frost_epoch_parser_accepts_new_high_marker() -> None:
    rows = parse_frost_epoch_summaries(
        "[Epoch 0] Train Loss: 0.7000 | Val AP(New high): 0.8000 | "
        "Val AUC: 0.7500 | Test AUC: 0.7400 | Test AP: 0.7900"
    )
    assert rows == [
        {
            "system": "frost",
            "epoch": 0,
            "train_loss": 0.7,
            "evaluation_metric": "ap",
            "val_ap": 0.8,
            "val_auc": 0.75,
            "test_auc": 0.74,
            "test_ap": 0.79,
        }
    ]


def test_frost_epoch_wall_parser() -> None:
    assert parse_frost_epoch_wall_summaries(
        'APPENDIX_TRAIN_EPOCH_WALL {"epoch":2,"wall_time_s":3.5}'
    ) == {2: 3.5}


def test_frost_and_pipetgl_epoch_parsers_accept_tgb_mrr() -> None:
    frost = parse_frost_epoch_summaries(
        "[Epoch 0] Train Loss: 0.7000 | Val MRR(New high): 0.4000 | "
        "Test MRR: 0.3900"
    )
    pipetgl = parse_pipetgl_epoch_summaries(
        'PIPETGL_EPOCH_SUMMARY {"epoch":1,"train_loss":0.7,'
        '"val_mrr":0.4,"test_mrr":0.39,"events":100,"cumulative_events":100,'
        '"train_time_s":2.0,"cumulative_train_time_s":2.0}'
    )

    assert frost[0]["val_mrr"] == 0.4
    assert frost[0]["test_mrr"] == 0.39
    assert pipetgl[0]["evaluation_metric"] == "mrr"
    assert pipetgl[0]["val_mrr"] == 0.4
    assert pipetgl[0]["test_mrr"] == 0.39


def test_tgl_epoch_parser_accepts_mrr() -> None:
    rows = parse_tgl_epoch_summaries(
        'TGL_EPOCH_SUMMARY {"system":"tgl","epoch":1,'
        '"train_loss":0.7,"evaluation_metric":"mrr",'
        '"val_mrr":0.4,"test_mrr":0.39,"epoch_wall_time_s":3.0}'
    )

    assert rows == [
        {
            "system": "tgl",
            "epoch": 1,
            "train_loss": 0.7,
            "evaluation_metric": "mrr",
            "epoch_wall_time_s": 3.0,
            "val_mrr": 0.4,
            "test_mrr": 0.39,
        }
    ]


def test_common_curve_helpers() -> None:
    payloads = parse_rank_payloads(
        'CURVE {"rank": 1, "reserved_mib_curve": [2, 4], "allocated_mib_curve": [1, 3]}\n'
        'CURVE {"rank": 0, "reserved_mib_curve": [3, 5], "allocated_mib_curve": [2, 4]}',
        "CURVE",
    )
    assert sum_memory_curves_by_rank(payloads) == ([5.0, 9.0], [3.0, 7.0])
    assert summarize_distribution([3.0, 1.0, 2.0]) == {
        "mean": 2.0,
        "p50": 2.0,
        "p90": 2.0,
        "max": 3.0,
    }
    assert normalized_progress([1, 2, 3]) == [0.0, 50.0, 100.0]
