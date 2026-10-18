from __future__ import annotations

from frost.data.paths import static_features_present_on_disk


def test_static_feature_probe_detects_node_or_edge_files(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "LASTFM"
    dataset.mkdir()
    monkeypatch.setenv("FROST_DATA_ROOT", str(tmp_path))

    assert not static_features_present_on_disk("LASTFM")
    (dataset / "node_features.npy").touch()
    assert static_features_present_on_disk("LASTFM")
    assert not static_features_present_on_disk(
        "LASTFM", ignore_node_features=True
    )

    (dataset / "edge_features.pt").touch()
    assert static_features_present_on_disk(
        "LASTFM", ignore_node_features=True
    )
    assert not static_features_present_on_disk(
        "LASTFM",
        ignore_node_features=True,
        ignore_edge_features=True,
    )


def test_static_feature_probe_knows_raw_tgbl_wiki_features() -> None:
    assert static_features_present_on_disk("TGBL_WIKI")
    assert not static_features_present_on_disk(
        "TGBL_WIKI", ignore_edge_features=True
    )


def test_static_feature_probe_knows_all_raw_tgb_features() -> None:
    for dataset in ("TGBL_COIN", "TGBL_COMMENT", "TGBL_REVIEW"):
        assert static_features_present_on_disk(dataset)
        assert not static_features_present_on_disk(
            dataset, ignore_edge_features=True
        )
    assert static_features_present_on_disk(
        "TGBL_FLIGHT", ignore_edge_features=True
    )
    assert not static_features_present_on_disk(
        "TGBL_FLIGHT", ignore_node_features=True, ignore_edge_features=True
    )
