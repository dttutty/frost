import torch

from frost.datatypes.graph import HierGraph, NeighborOnHop
from frost.runtime.worker.mailbox_router import NeighborhoodMailboxRouter


def test_neighborhood_mailbox_router_targets_valid_positive_neighbors() -> None:
    graph = HierGraph(
        root_nid=torch.tensor([1, 2, 3]),
        root_ts=torch.tensor([10.0, 20.0, 20.0]),
        neighbors=[
            NeighborOnHop(
                nid=torch.tensor([4, -1, 5, 6]),
                eid=torch.tensor([0, -1, 1, 2]),
                ts=torch.tensor([9.0, 0.0, 19.0, 18.0]),
            )
        ],
        fanout=[2],
        n_neg=1,
    )
    payload = torch.tensor([[1.0, 10.0], [2.0, 20.0]])

    mail = NeighborhoodMailboxRouter().deliver(graph, payload)

    assert mail.nid.tolist() == [4, 5, 6]
    assert mail.ts.tolist() == [10.0, 20.0, 20.0]
    assert mail.data.tolist() == [[1.0, 10.0], [2.0, 20.0], [2.0, 20.0]]
