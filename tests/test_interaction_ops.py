import pytest
import torch

from frost.datatypes.mem_mail_box.mem import MemEntry
from frost.runtime.worker.p2p_syncer.utils._interaction_ops import (
    generate_interaction_mails,
)


def test_generate_interaction_mails_uses_peer_mem_for_other_half():
    root_mem = MemEntry(
        nid=torch.tensor([1, 2, 3, 4]),
        data=torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        ),
        ts=torch.tensor([5.0, 6.0, 5.0, 6.0]),
    )
    peer_mem = torch.tensor(
        [
            [100.0, 1000.0],
            [200.0, 2000.0],
            [300.0, 3000.0],
            [400.0, 4000.0],
            [999.0, 9999.0],
        ]
    )
    edge_feats = torch.tensor([[0.5], [0.6]])

    mail = generate_interaction_mails(root_mem, edge_feats, peer_mem)

    assert torch.equal(mail.nid, root_mem.nid)
    assert torch.equal(mail.ts, root_mem.ts)
    assert torch.equal(
        mail.data,
        torch.tensor(
            [
                [1.0, 10.0, 300.0, 3000.0, 0.5],
                [2.0, 20.0, 400.0, 4000.0, 0.6],
                [3.0, 30.0, 100.0, 1000.0, 0.5],
                [4.0, 40.0, 200.0, 2000.0, 0.6],
            ]
        ),
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


def test_generate_interaction_mails_rejects_peer_dim_mismatch():
    root_mem = MemEntry(
        nid=torch.tensor([1, 2]),
        data=torch.randn(2, 4),
        ts=torch.tensor([5.0, 5.0]),
    )

    with pytest.raises(ValueError, match="feature dim"):
        generate_interaction_mails(root_mem, peer_mem=torch.randn(2, 3))
