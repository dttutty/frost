from schemas.datatypes.misc import RootDyX, Shared
from schemas.datatypes.plan import TensorDyPlan


class RootXDyAssembler:
    """
    Ths is used for filling mem, mailbox related fields of RootDyX for identity GNNs.
    """

    def __init__(self, shared: Shared):
        self.shared = shared

    def assemble(
        self, plan: TensorDyPlan, root_nid, root_x_DyX: RootDyX | None
    ) -> RootDyX:
        fetch = plan.fetch
        root_x_part = RootDyX(
            nid=fetch,
            mem=self.shared.mem[plan.fetch],
            mem_ts=self.shared.mem_ts[plan.fetch],
            box=self.shared.box[plan.fetch],
            box_ts=self.shared.box_ts[plan.fetch],
        )
        if root_x_DyX is not None:
            root_x = root_x_part.merge_with_another_rootx(root_x_DyX)
        else:
            root_x = root_x_part
        root_x = root_x.unzip_by_root_nid(root_nid)
        # cannot instant caching because mem will got updated
        return root_x
