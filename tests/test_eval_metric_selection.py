from pathlib import Path

from omegaconf import OmegaConf

from frost.datatypes.result import Result
from frost.runtime.host._engine import Host


def _result(*, val_ap: float, val_mrr: float) -> Result:
    return Result(
        train_loss=1.0,
        val_auc=0.0,
        val_ap=val_ap,
        test_auc=0.0,
        test_ap=0.0,
        val_mrr=val_mrr,
    )


def test_default_eval_metric_remains_ap() -> None:
    config = OmegaConf.load(Path(__file__).parents[1] / "configs" / "config.yaml")

    assert config.eval_metric == "ap"


def test_host_selects_the_configured_validation_metric() -> None:
    host = Host.__new__(Host)
    result = _result(val_ap=0.8, val_mrr=0.2)

    host.eval_metric = "ap"
    assert host._selection_score(result) == 0.8

    host.eval_metric = "mrr"
    assert host._selection_score(result) == 0.2
