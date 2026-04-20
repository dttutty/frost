from pathlib import Path

import pytest

from frost.utils.batch_size import get_max_macro_batch_size


def test_get_max_macro_batch_size_from_num_edges():
    # 74478 edges, split (7,1.5,1.5) -> min_eval = floor(74478 * 1.5 / 10) = 11171
    assert get_max_macro_batch_size(74478) == 11171


def test_get_max_macro_batch_size_from_dataset_name_case_insensitive(tmp_path: Path):
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

    # 4 edges, split (2,1,1) -> min_eval = 4 * 1 // 4 = 1
    assert get_max_macro_batch_size("LASTFM", split=(2, 1, 1), data_dir=tmp_path) == 1


def test_get_max_macro_batch_size_with_custom_split():
    # 100 edges, split (7,1.5,1.5) -> min_eval = floor(100 * 1.5 / 10) = 15
    assert get_max_macro_batch_size(100, split=(7, 1.5, 1.5)) == 15


@pytest.mark.parametrize(
    ("split", "error"),
    [
        ((8,), ValueError),
        ((8, 0, 1), ValueError),
        ((8, "1", 1), TypeError),
        ((8, True, 1), TypeError),
    ],
)
def test_get_max_macro_batch_size_rejects_invalid_split(
    split: tuple[object, ...], error: type[Exception]
):
    with pytest.raises(error):
        get_max_macro_batch_size(100, split=split)
