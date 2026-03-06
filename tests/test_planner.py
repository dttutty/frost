import threading
import unittest
from queue import Queue as Q  # 这是给多线程用的
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
from host.meta_batcher import MetaBundle

# 假设您的 planner.py 在当前路径或可导入路径下
from host.planner import Planner, find_all_gpu_distances_list
from schemas.datatypes.misc import Meta
from schemas.datatypes.plan import DynamicPlan, StaticPlan


class TestGpuTopology(unittest.TestCase):
    """测试 GPU 拓扑距离计算函数的正确性"""

    def test_flat_topology(self) -> None:
        # 情况 1: 扁平列表 [0, 1, 2]
        # 所有 GPU 都在同一层级，两两之间距离应相等（除了自身）
        topo = [0, 1, 2]
        dist_mat = find_all_gpu_distances_list(topo)

        expected_size = (3, 3)
        self.assertEqual(dist_mat.shape, expected_size)

        # Max Depth = 1. Current Depth = 0. Distance = 1.
        # 对角线为 0，其余为 1
        expected = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        np.testing.assert_array_equal(dist_mat, expected)

    def test_nested_topology(self) -> None:
        # 情况 2: 嵌套列表 [[0, 1], [2, 3]] (例如两台机器，每台2卡)
        # 组内距离 (0-1) 应小于组间距离 (0-2)
        topo = [[0, 1], [2, 3]]
        dist_mat = find_all_gpu_distances_list(topo)

        # Max Depth = 2.
        # Root (depth 0) splits [0,1] and [2,3]. Distance = 2 - 0 = 2.
        # Group [0,1] (depth 1) splits 0 and 1. Distance = 2 - 1 = 1.

        expected = np.array([[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 1], [2, 2, 1, 0]])
        np.testing.assert_array_equal(dist_mat, expected)

    def test_unbalanced_topology(self) -> None:
        # 情况 3: 不平衡拓扑 [[0, 1], 2]
        topo = [[0, 1], 2]
        dist_mat = find_all_gpu_distances_list(topo)

        # Max Depth = 2 (因为 [[...]] 这一支深).
        # Root (depth 0) splits [0,1] and 2. Distance = 2.
        # Group [0,1] (depth 1) splits 0 and 1. Distance = 1.

        expected = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]])
        np.testing.assert_array_equal(dist_mat, expected)


class TeStaticPlanner(unittest.TestCase):
    """测试 Planner 线程的逻辑流"""

    def setUp(self) -> None:
        # 1. 准备基础依赖
        self.stop_ev = threading.Event()

        # 2. 准备队列

        self.meta_in_q: Q[Meta | None | Literal["end_of_epoch"]] = Q(maxsize=2)
        self.in_q: Q[MetaBundle | None | Literal["end_of_epoch"]] = Q(maxsize=1)
        self.out_q: Q[MetaBundle | None | Literal["end_of_epoch"]] = Q(maxsize=1)

        self.mem_plan_out_q: Q[list[DynamicPlan] | None] | None = None
        self.mail_plan_out_q: Q[list[DynamicPlan] | None] | None = None
        self.nfeat_plan_out_q: Q[list[StaticPlan] | None] | None = None
        self.planner: Planner | None = None
        self.mem_out_q = Q(maxsize=1)
        self.mail_out_q = Q(maxsize=1)
        self.nfeat_out_q = Q(maxsize=1)

        # 3. Mock 掉内部的子 Planner，避免启动真实线程和依赖复杂逻辑
        self.dy_patcher = patch("host.planner.DynamicPlanner")
        self.st_patcher = patch("host.planner.StaticPlanner")
        self.MockDynamicPlanner = self.dy_patcher.start()
        self.MockStaticPlanner = self.st_patcher.start()

        # 4. 简单的 2 卡拓扑
        self.topo = [0, 1]

    def tearDown(self) -> None:
        self.dy_patcher.stop()
        self.st_patcher.stop()

    def test_init(self) -> None:
        """测试初始化是否正确创建了子 Planner"""
        planner = Planner(
            gpu_topo=self.topo,
            in_q=self.in_q,
            out_q=self.out_q,
            mem_plan_out_q=self.mem_out_q,
            mail_plan_out_q=self.mail_out_q,
            nfeat_plan_out_q=self.nfeat_out_q,
            stop_ev=self.stop_ev,
        )

        # 验证是否根据队列的存在初始化了对应的子 Planner
        self.assertTrue(self.MockDynamicPlanner.called)
        self.assertTrue(self.MockStaticPlanner.called)
        self.assertEqual(planner.n_gpus, 2)

    def test_logic_shutdown(self) -> None:
        """测试收到 None 时的关闭逻辑"""
        planner = Planner(
            gpu_topo=self.topo,
            in_q=self.in_q,
            out_q=self.out_q,
            mem_plan_out_q=self.mem_out_q,
            mail_plan_out_q=self.mail_out_q,
            nfeat_plan_out_q=self.nfeat_out_q,
            stop_ev=self.stop_ev,
        )

        # 在独立线程中运行 logic，因为它包含 while 循环
        t = threading.Thread(target=planner.logic)
        t.start()

        # 发送停止信号
        self.in_q.put(None)

        # 等待线程结束
        t.join(timeout=2.0)
        self.assertFalse(
            t.is_alive(), "Planner thread should exit after receiving None"
        )

        # 验证 None 是否传播到了所有下游队列
        self.assertIsNone(self.out_q.get())
        self.assertIsNone(planner.mem_plan_in_q.get())
        self.assertIsNone(planner.mail_plan_in_q.get())
        self.assertIsNone(planner.nfeat_plan_in_q.get())

    def test_logic_end_of_epoch(self) -> None:
        """测试 end_of_epoch 信号的传播"""
        planner = Planner(
            gpu_topo=self.topo,
            in_q=self.in_q,
            out_q=self.out_q,
            mem_plan_out_q=self.mem_out_q,
            mail_plan_out_q=self.mail_out_q,
            nfeat_plan_out_q=self.nfeat_out_q,
            stop_ev=self.stop_ev,
        )

        t = threading.Thread(target=planner.logic)
        t.start()

        self.in_q.put("end_of_epoch")

        # 验证传播
        self.assertEqual(self.out_q.get(timeout=1.0), "end_of_epoch")
        self.assertEqual(planner.mem_plan_in_q.get(timeout=1.0), "end_of_epoch")

        # 清理并停止
        self.in_q.put(None)
        t.join()

    def test_logic_normal_batch(self) -> None:
        """测试正常数据包的处理与分发"""
        planner = Planner(
            gpu_topo=self.topo,
            in_q=self.in_q,
            out_q=self.out_q,
            mem_plan_out_q=self.mem_out_q,
            mail_plan_out_q=self.mail_out_q,
            nfeat_plan_out_q=self.nfeat_out_q,
            stop_ev=self.stop_ev,
        )

        t = threading.Thread(target=planner.logic)
        t.start()

        # --- 构造模拟数据 ---
        # 模拟 TensorSet (这里用 MagicMock 即可，因为 logic 只做传递)
        mock_tensor_set = MagicMock()

        # 模拟 StateIO (BatchNodeStateAccess)
        state_io = MagicMock()
        state_io.mem_needed = mock_tensor_set
        state_io.mem_updated = mock_tensor_set
        state_io.mailbox_needed = mock_tensor_set
        state_io.mailbox_updated = mock_tensor_set
        state_io.nfeat_needed = mock_tensor_set

        # 模拟 Meta 对象
        meta = MagicMock()
        meta.state = state_io

        # 输入必须是列表，且长度等于 n_gpus (这里是2)
        metas = [meta, meta]

        # 发送数据
        self.in_q.put(metas)

        # --- 验证分发逻辑 ---

        # 1. 验证 Mem Planner 收到的数据: ([needs...], [updates...])
        mem_data = planner.mem_plan_in_q.get(timeout=1.0)
        self.assertIsInstance(mem_data, tuple)
        self.assertEqual(len(mem_data[0]), 2)  # needs list
        self.assertEqual(len(mem_data[1]), 2)  # updates list

        # 2. 验证 Mail Planner 收到的数据
        mail_data = planner.mail_plan_in_q.get(timeout=1.0)
        self.assertIsInstance(mail_data, tuple)

        # 3. 验证 NFeat Planner 收到的数据: [needs...]
        nfeat_data = planner.nfeat_plan_in_q.get(timeout=1.0)
        self.assertIsInstance(nfeat_data, list)
        self.assertEqual(len(nfeat_data), 2)

        # 4. 验证 Out Queue (Metas 应该透传出去)
        out_metas = self.out_q.get(timeout=1.0)
        self.assertEqual(out_metas, metas)

        # 停止
        self.in_q.put(None)
        t.join()


if __name__ == "__main__":
    unittest.main()
