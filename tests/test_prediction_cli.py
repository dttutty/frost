"""End-to-end user CSV training and causal candidate scoring contracts."""

import csv
import json

import numpy as np
import pytest
import torch

from frost.predict.cli import main
from frost.predict.data import convert_features, events, load_features, times
from frost.predict.model import History, load, score, train


@pytest.fixture
def trained(tmp_path):
    torch.set_num_threads(2)
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "src,dst,time\n"
        + "".join(
            f"{'alice,book' if i % 2 == 0 else 'bob,game'},{i + 1}\n" for i in range(32)
        )
    )
    output = tmp_path / "model.pt"
    checkpoint = train(
        csv_path, output=output, epochs=8, lr=0.01, hidden=16, fanout=3, batch_size=16
    )
    return output, checkpoint


def test_train_learns_and_checkpoint_roundtrips(trained):
    output, saved = trained
    checkpoint, model, history = load(output)
    assert saved["training"]["loss"][-1] < saved["training"]["loss"][0]
    src, dst, ts = (
        ["alice", "alice", "bob", "bob"],
        ["book", "game", "game", "book"],
        np.full(4, 32.0),
    )
    scores = score(checkpoint, model, history, src, dst, ts)
    assert all(0 <= value <= 1 for value in scores)
    assert scores[0] > scores[1] and scores[2] > scores[3]
    np.testing.assert_allclose(
        scores, score(*load(output), src, dst, ts, batch_size=1), atol=1e-6
    )
    assert checkpoint["node_ids"] == ["alice", "bob", "book", "game"]


def test_prediction_cli_default_time_and_query_csv(trained, tmp_path, capsys):
    output, _ = trained
    main(["predict", str(output), "--src", "alice", "--candidates", "book,game,"])
    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert [row["dst"] for row in rows] == ["book", "game"]
    assert [float(row["time"]) for row in rows] == [32, 32]
    query = tmp_path / "test.csv"
    query.write_text("src,dst,time\nbob,book,35\nalice,game,33\n")
    predictions = tmp_path / "predictions.csv"
    main(
        [
            "predict",
            str(output),
            "--test",
            str(query),
            "--predictions",
            str(predictions),
        ]
    )
    with predictions.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["time"]) for row in rows] == [35, 33]
    assert [row["src"] for row in rows] == ["bob", "alice"]
    main(["predict", str(output), "--test", str(query), "--time", "40"])
    assert all(
        float(row["time"]) == 40
        for row in csv.DictReader(capsys.readouterr().out.splitlines())
    )
    query.write_text("src,dst\nbob,book\n")
    main(["predict", str(output), "--test", str(query)])
    assert (
        float(next(csv.DictReader(capsys.readouterr().out.splitlines()))["time"]) == 32
    )
    assert load(output)[0]["edges"].shape == (
        32,
        2,
    )  # Query CSV never becomes training history.


def test_query_rejects_unknown_nodes_and_past_times(trained):
    loaded = load(trained[0])
    with pytest.raises(ValueError, match="cutoff"):
        score(*loaded, ["alice"], ["book"], np.array([31.0]))
    with pytest.raises(ValueError, match="Unknown node"):
        score(*loaded, ["unknown"], ["book"], np.array([32.0]))
    with pytest.raises(ValueError, match="cutoff"):
        score(*loaded, ["alice"], ["book"], np.array([np.nan]))


def test_training_history_excludes_equal_and_future_events():
    history = History(
        np.array([[0, 1], [0, 2], [0, 3], [0, 4]]), np.array([1.0, 2.0, 2.0, 3.0]), 5
    )
    nodes, _, ts = history.sample([0], [2.0], 4)
    assert nodes.tolist() == [[-1, -1, -1, 1]]
    assert ts[0, -1] == 1
    assert history.sample([0], [2.0], 4, inclusive=True)[0].tolist() == [[-1, 1, 2, 3]]
    assert history.sample([0], [3.0], 2, inclusive=True)[0].tolist() == [[3, 4]]


def test_timestamp_free_csv_one_command(tmp_path, capsys):
    source = tmp_path / "events.csv"
    source.write_text("src_id,dst_id\n001,121\n002,123\n001,121\n002,123\n")
    output = tmp_path / "order.pt"
    main(
        [
            "train",
            str(source),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--hidden",
            "8",
            "--src",
            "001",
            "--candidates",
            "121,123,",
        ]
    )
    checkpoint = load(output)[0]
    assert checkpoint["time_kind"] == "order" and checkpoint["last_time"] == 4
    assert checkpoint["node_ids"][0] == "001"
    assert len(list(csv.DictReader(capsys.readouterr().out.splitlines()))) == 2
    with pytest.raises(ValueError, match="already exists"):
        train(source, output=output)


def test_datetime_and_aliases(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "u,i,timestamp\na,x,2026-01-02T00:00:00Z\nb,y,2026-01-01T00:00:00Z\n"
    )
    src, dst, ts, kind = events(path)
    assert src == ["a", "b"] and dst == ["x", "y"] and kind == "datetime"
    assert ts[0] - ts[1] == 86400
    output = tmp_path / "dates.pt"
    train(path, output=output, epochs=1, hidden=8)
    assert load(output)[0]["timestamps"].tolist() == sorted(ts.tolist())
    value, _ = times(["2026-01-03T00:00:00Z"], kind)
    assert value[0] > ts.max()
    with pytest.raises(ValueError):
        times(["10"], kind)


@pytest.mark.parametrize("values", [["1", "bad"], ["NaN"], ["inf"], [""]])
def test_invalid_time(values):
    with pytest.raises(ValueError):
        times(values)


def test_feature_conversion_and_alignment(tmp_path):
    path = tmp_path / "nodes.csv"
    path.write_text("n_id,age,kind\n001,10,A\n002,20,B\n121,,A\n123,30,B\n")
    output = tmp_path / "nodes.npy"
    assert convert_features(path, output, categorical_dim=8) == (4, 9)
    features = load_features(output, ["123", "001"])
    original = np.load(output)
    np.testing.assert_array_equal(features, original[[3, 0]])
    assert np.isfinite(original).all() and original[2, 0] == 0
    assert (
        json.loads((tmp_path / "nodes.npy.schema.json").read_text())["numeric"]["age"][
            "mean"
        ]
        == 20
    )
    events_csv = tmp_path / "events.csv"
    events_csv.write_text("src,dst\n001,121\n002,123\n")
    checkpoint = train(
        events_csv,
        output=tmp_path / "features.pt",
        node_features=output,
        epochs=1,
        hidden=8,
    )
    np.testing.assert_array_equal(
        checkpoint["state_dict"]["features"].numpy(), original
    )
    with pytest.raises(ValueError, match="already exists"):
        convert_features(path, output)
    with pytest.raises(ValueError, match="Missing node features"):
        load_features(output, ["missing"])


def test_npy_row_ids_and_validation(tmp_path):
    path = tmp_path / "matrix.npy"
    np.save(path, np.arange(12).reshape(4, 3))
    np.testing.assert_array_equal(
        load_features(path, ["3", "0"]), np.array([[9, 10, 11], [0, 1, 2]])
    )
    for nodes in (["foo"], ["01"], ["-1"], ["5"]):
        with pytest.raises(ValueError):
            load_features(path, nodes)
    np.save(path, np.array([[np.nan]]))
    with pytest.raises(ValueError, match="finite"):
        load_features(path, ["0"])


def test_invalid_inputs_fail_before_training(tmp_path):
    source = tmp_path / "events.csv"
    source.write_text("src,dst,time\na,b,1\na,c,1\n")
    with pytest.raises(ValueError, match="No negative"):
        train(source, output=tmp_path / "bad.pt")
    assert not (tmp_path / "bad.pt").exists()
    with pytest.raises(SystemExit) as error:
        main(["train", str(source), "--src", "a"])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        main(["predict", "missing.pt"])
    with pytest.raises(ValueError, match="Missing column"):
        events(source, time_col="missing")


def test_tgn_memory_is_trained_replayed_and_read_only(trained):
    from frost.predict.model import TGNPredictor, temporal_batches

    output, saved = trained
    assert saved["model"] == "tgn" and saved["format_version"] == 2
    checkpoint, model, history = load(output)
    assert isinstance(model, TGNPredictor)
    torch.manual_seed(saved["training"]["seed"])
    initial = TGNPredictor(
        len(saved["node_ids"]), saved["state_dict"]["features"], **saved["config"]
    )
    assert not torch.equal(initial.updater.rnn.weight_ih, model.updater.rnn.weight_ih)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    first = score(
        checkpoint,
        model,
        history,
        ["alice", "bob"],
        ["book", "game"],
        np.array([32.0, 40.0]),
    )
    second = score(
        checkpoint,
        model,
        history,
        ["bob", "alice"],
        ["game", "book"],
        np.array([40.0, 32.0]),
    )
    np.testing.assert_allclose(first, second[::-1])
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)
    model.reset_memory(float(checkpoint["timestamps"][0]))
    assert not model.has_mail.any() and not model.memory.any()
    for start, end in temporal_batches(
        checkpoint["timestamps"].numpy(), checkpoint["training"]["batch_size"]
    ):
        model.observe(
            checkpoint["edges"][start:end].numpy(),
            checkpoint["timestamps"][start:end].numpy(),
        )
    for key in ("memory", "memory_time", "mail", "mail_time", "has_mail"):
        torch.testing.assert_close(model.state_dict()[key], before[key], rtol=0, atol=0)


def test_tgn_equal_times_never_observe_current_or_future_batch(tmp_path, monkeypatch):
    from frost.predict.model import TGNPredictor

    source = tmp_path / "events.csv"
    source.write_text("src,dst,time\na,x,1\nb,y,1\na,x,1\nb,y,2\na,x,3\nb,y,3\n")
    original = TGNPredictor.forward
    seen = []

    def checked(self, history, src, dst, timestamps, inclusive=False):
        if self.has_mail.any():
            assert self.mail_time[self.has_mail].max().item() < min(timestamps)
        seen.append(min(timestamps))
        return original(self, history, src, dst, timestamps, inclusive)

    monkeypatch.setattr(TGNPredictor, "forward", checked)
    train(
        source,
        output=tmp_path / "model.pt",
        epochs=2,
        batch_size=2,
        hidden=8,
        time_dim=4,
    )
    # Equal-time group of 3 is computed in two bounded chunks against frozen state.
    assert seen == [1.0, 1.0, 2.0, 3.0] * 2


def test_tgat_selection_and_legacy_checkpoint(tmp_path):
    source = tmp_path / "events.csv"
    source.write_text("src,dst,time\na,x,1\nb,y,2\na,x,3\nb,y,4\n")
    path = tmp_path / "tgat.pt"
    saved = train(
        source,
        output=path,
        model="tgat",
        epochs=1,
        hidden=8,
        time_dim=16,
        heads=2,
        dropout=0.1,
    )
    expected = score(*load(path), ["a"], ["x"], np.array([4.0]))
    saved["format_version"] = 1
    saved["config"] = {key: saved["config"][key] for key in ("fanout", "hidden")}
    saved.pop("resolved_config")
    legacy = tmp_path / "legacy.pt"
    torch.save(saved, legacy)
    np.testing.assert_allclose(
        score(*load(legacy), ["a"], ["x"], np.array([4.0])), expected
    )


def test_default_config_and_override_precedence(tmp_path, capsys):
    import yaml
    from frost.predict.config import DEFAULTS

    main(["train", "--show-config"])
    config = yaml.safe_load(capsys.readouterr().out)
    assert config == DEFAULTS
    assert (
        config["model"] == "tgn" and config["fanout"] == 10 and config["hidden"] == 128
    )
    custom = tmp_path / "settings.yaml"
    custom.write_text("model: tgat\nepochs: 3\nhidden: 16\nthreads: 2\n")
    main(["train", "--config", str(custom), "--epochs", "5", "--show-config"])
    resolved = yaml.safe_load(capsys.readouterr().out)
    assert (
        resolved["model"] == "tgat"
        and resolved["epochs"] == 5
        and resolved["hidden"] == 16
    )
    source = tmp_path / "events.csv"
    source.write_text("src,dst\na,x\nb,y\n")
    output = tmp_path / "custom.pt"
    main(
        [
            "train",
            str(source),
            "--config",
            str(custom),
            "--epochs",
            "1",
            "--output",
            str(output),
        ]
    )
    saved = load(output)[0]["resolved_config"]
    assert saved["epochs"] == 1 and saved["threads"] == 2 and saved["model"] == "tgat"


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "unknown: 1",
        "epochs: 0",
        "hidden: 7",
        "dropout: 1",
        "seed: -1",
        "model: bad",
        "epochs: true",
        "lr: .nan",
    ],
)
def test_bad_config_fails_before_training(tmp_path, contents):
    config = tmp_path / "bad.yaml"
    config.write_text(contents)
    with pytest.raises(SystemExit) as exc:
        main(["train", "--config", str(config), "--show-config"])
    assert exc.value.code == 2


def test_tgn_last_message_uses_matching_peer_and_preserves_other_nodes():
    from frost.predict.model import TGNPredictor

    model = TGNPredictor(4, torch.empty(4, 0), hidden=2, time_dim=2)
    with torch.no_grad():
        model.memory.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )
    model.observe(np.array([[0, 1], [2, 0]]), np.array([1.0, 2.0]))
    torch.testing.assert_close(
        model.mail[:3],
        torch.tensor(
            [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 1.0, 2.0], [5.0, 6.0, 1.0, 2.0]]
        ),
    )
    assert model.mail_time.tolist() == [2.0, 1.0, 2.0, 0.0]
    assert model.has_mail.tolist() == [True, True, True, False]
    torch.testing.assert_close(model.memory[3], torch.tensor([7.0, 8.0]))
