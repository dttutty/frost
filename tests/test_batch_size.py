from pathlib import Path

import pytest

from frost.utils.batch_size import get_max_batch_size


def test_get_max_batch_size_from_num_edges():
    assert get_max_batch_size(74478, n_gpu=1) == 7447
    assert get_max_batch_size(74478, n_gpu=2) == 3723


def test_get_max_batch_size_from_dataset_name_case_insensitive(tmp_path: Path):
    dataset_dir = tmp_path / "lastfm"
    dataset_dir.mkdir()
    (dataset_dir / "edges.csv").write_text(
        "eid,src,dst,time,default_split\n"
        "0,1,2,3,0\n"
        "1,2,3,4,0\n"
        "2,3,4,5,1\n"
        "3,4,5,6,2\n",
        encoding="utf-8",
    )

    assert get_max_batch_size("LASTFM", split=(2, 1, 1), n_gpu=1, data_dir=tmp_path) == 1


def test_get_max_batch_size_with_custom_split():
    assert get_max_batch_size(100, split=(7, 2, 1), n_gpu=2) == 5


@pytest.mark.parametrize(
    ("split", "error"),
    [
        ((8,), ValueError),
        ((8, 0, 1), ValueError),
        ((8, True, 1), TypeError),
    ],
)
def test_get_max_batch_size_rejects_invalid_split(split: tuple[int, ...], error: type[Exception]):
    with pytest.raises(error):
        get_max_batch_size(100, split=split)


def test_get_max_batch_size_rejects_invalid_n_gpu():
    with pytest.raises(ValueError):
        get_max_batch_size(100, n_gpu=0)
