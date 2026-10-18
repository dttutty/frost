import torch
import pytest

from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.runtime.worker.p2p_syncer.utils._interaction_ops import (
    generate_interaction_mails,
)


def test_generate_interaction_mails_falls_back_to_other_memory():
    root_mem = MemEntry(
        nid=torch.tensor([1, 2]),
        data=torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        ts=torch.tensor([5.0, 5.0]),
    )

    mail = generate_interaction_mails(root_mem)

    assert torch.equal(
        mail.data,
        torch.tensor([[1.0, 10.0, 2.0, 20.0], [2.0, 20.0, 1.0, 10.0]]),
    )


def test_generate_interaction_mails_rejects_misaligned_memory_rows():
    root_mem = MemEntry(
        nid=torch.tensor([1, 2]),
        data=torch.tensor([[1.0, 10.0]]),
        ts=torch.tensor([5.0, 5.0]),
    )

    with pytest.raises(ValueError, match="nid/data rows must match"):
        generate_interaction_mails(root_mem)


def test_generate_interaction_mails_rejects_misaligned_edge_features():
    root_mem = MemEntry(
        nid=torch.tensor([1, 2]),
        data=torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        ts=torch.tensor([5.0, 5.0]),
    )

    with pytest.raises(ValueError, match="edge feature rows must match"):
        generate_interaction_mails(root_mem, torch.empty((0, 1)))
