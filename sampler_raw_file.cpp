#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef int NodeIDType;
typedef int EdgeIDType;
typedef float TimeStampType;

struct ThreadSamplingBuffers {
    std::vector<NodeIDType> row_data;
    std::vector<NodeIDType> col_data;
    std::vector<EdgeIDType> eid_data;
    std::vector<TimeStampType> ts_data;
    std::vector<TimeStampType> dts_data;
    std::vector<NodeIDType> nodes_data;
    int single_thead_src_num; // 对应你原来的 _out_node[tid]

    // 构造函数可以负责初始化，比如预留容量
    explicit ThreadSamplingBuffers(int capacity = 0) : single_thead_src_num(0) {
        if (capacity > 0) {
            row_data.reserve(capacity);
            col_data.reserve(capacity);
            eid_data.reserve(capacity);
            ts_data.reserve(capacity);
            dts_data.reserve(capacity);
            nodes_data.reserve(capacity);
        }
    }

    // 内联函数，用于向当前线程的缓冲区中添加一个采样到的邻居信息
    inline void add_dst(
        const EdgeIDType& picked,
        const TimeStampType& src_ts,
        const std::vector<NodeIDType>& indices,
        const std::vector<EdgeIDType>& eid,
        const std::vector<TimeStampType>& ts,
        const bool prop_time
    )
    {
        NodeIDType local_src_id = single_thead_src_num; // 当前线程的源节点在当前层输出节点中的相对ID
        NodeIDType local_dst_id = nodes_data.size(); // 当前线程的源节点在当前层输出节点中的相对ID
        row_data.push_back(local_src_id);        // 源节点在当前层输出节点中的相对ID
        col_data.push_back(local_dst_id);// 目标节点在下一层输入节点中的相对ID (即当前已采样的邻居数)
        EdgeIDType global_eid = eid[picked]; // 采样到的边的原始ID
        eid_data.push_back(global_eid);        // 边的原始ID
        if (prop_time)                  // 如果 prop_time 为 true 是否将父节点的时间戳传播给子节点采样的边
            ts_data.push_back(src_ts);     // 使用源节点的时间戳作为这条采样边的时间戳
        else
            ts_data.push_back(ts[picked]);      // 否则使用边的原始时间戳
        dts_data.push_back(src_ts - ts[picked]);// 计算时间差
        nodes_data.push_back(indices[picked]);  // 添加邻居节点的原始ID

    }
};

class TemporalGraphBlock
{
public:
    // COO (Coordinate Format) 格式存储的图边信息
    std::vector<NodeIDType> row;    // 边的源节点列表 (在当前块内的相对ID)
    std::vector<NodeIDType> col;    // 边的目标节点列表 (在当前块内的相对ID)
    std::vector<EdgeIDType> eid;    // 边的原始ID列表
    std::vector<TimeStampType> ts;  // 边的时间戳列表 (或传播过来的时间戳)
    std::vector<TimeStampType> dts; // 边的时间差列表 (源节点时间戳 - 边原始时间戳)
    std::vector<NodeIDType> nodes;  // 块中包含的节点ID列表 (原始图中的ID)
    NodeIDType dim_in, dim_out;
    double ptr_time = 0;            // 更新时间指针所用时间
    double search_time = 0;         // 搜索邻居范围所用时间
    double sample_time = 0;         // 采样邻居所用时间
    double tot_time = 0;            // 该块总采样时间
    double coo_time = 0;            // 合并COO格式数据所用时间

    TemporalGraphBlock() {}

    TemporalGraphBlock(
        std::vector<NodeIDType>& _row,
        std::vector<NodeIDType>& _col,
        std::vector<EdgeIDType>& _eid,
        std::vector<TimeStampType>& _ts,
        std::vector<TimeStampType>& _dts,
        std::vector<NodeIDType>& _nodes,
        NodeIDType _dim_in,
        NodeIDType _dim_out
    ) :
        row(_row), col(_col), eid(_eid), ts(_ts), dts(_dts),
        nodes(_nodes), dim_in(_dim_in), dim_out(_dim_out) {
    }
};

class ParallelSampler
{
public:
    std::vector<EdgeIDType> indptr;  // 节点的邻接边在 indices, eid, ts 中的起始索引指针, python传入
    std::vector<EdgeIDType> indices; // 邻接节点ID列表 (所有节点的邻居顺序存储), python传入
    std::vector<EdgeIDType> eid;     // 对应 indices 中边的原始ID列表, python传入
    std::vector<TimeStampType> ts;   // 对应 indices 中边的时间戳列表, python传入

    NodeIDType num_nodes;           // 图中的总节点数, 计算得出
    EdgeIDType num_edges;           // 图中的总边数, 计算得出

    // 并行化参数
    int num_thread_per_worker;      // 每个 worker 使用的线程数 (通常 worker 指一个CPU核心或一组核心), python传入
    int num_workers;                // worker 的数量, python传入
    int num_threads;                // 总线程数 (num_thread_per_worker * num_workers), 计算得出

    // 采样参数
    int num_layers;                 // 采样的层数 (GNN的层数), python传入
    std::vector<int> num_neighbors; // 每层采样的邻居数量, python传入
    bool recent;                    // 是否总是采样最近的邻居 (如果为 true，则不进行随机采样), python传入
    bool prop_time;                 // 是否将父节点的时间戳传播给子节点采样的边, python传入
    int num_history;                // 历史窗口的数量 (用于 DySAT 或类似模型，将时间划分为多个窗口), python传入

    TimeStampType window_duration;  // 每个历史窗口的持续时间, python传入


    // 用于优化时间范围搜索的指针
    // ts_ptr[h][n] 表示节点 n 在历史窗口 h 内的第一条边的索引
    std::vector<std::vector<std::vector<EdgeIDType>::size_type>> ts_ptr;
    std::vector<omp_lock_t> ts_ptr_lock; // 用于保护 ts_ptr 更新的 OpenMP 锁

    std::vector<TemporalGraphBlock> tgbs; // 存储多层采样结果的 TemporalGraphBlock 列表, 传回python

    // 构造函数
    ParallelSampler(
        std::vector<EdgeIDType>& _indptr,
        std::vector<EdgeIDType>& _indices,
        std::vector<EdgeIDType>& _eid,
        std::vector<TimeStampType>& _ts,
        int _num_thread_per_worker,
        int _num_workers,
        int _num_layers,
        std::vector<int>& _num_neighbors,
        bool _recent,
        bool _prop_time,
        int _num_history,
        TimeStampType _window_duration
    ) :
        indptr(_indptr), indices(_indices), eid(_eid), ts(_ts), prop_time(_prop_time),
        num_thread_per_worker(_num_thread_per_worker), num_workers(_num_workers),
        num_layers(_num_layers), num_neighbors(_num_neighbors), recent(_recent),
        num_history(_num_history), window_duration(_window_duration)
    {
        omp_set_num_threads(num_thread_per_worker * num_workers); // 设置 OpenMP 线程数
        num_threads = num_thread_per_worker * num_workers;
        num_nodes = indptr.size() - 1; // 节点数通过 indptr 长度推断
        num_edges = indices.size();    // 边数通过 indices 长度推断


        // 初始化 ts_ptr_lock
        ts_ptr_lock.resize(indptr.size() - 1);
        for (auto& lock : ts_ptr_lock) {
            omp_init_lock(&lock); // 初始化每个节点的锁
        }

        // 初始化 ts_ptr
        // ts_ptr[0] 对应窗口 (t - T, t]
        // ts_ptr[1] 对应窗口 (t - 2T, t - T]
        // ...
        // ts_ptr[num_history-1] 对应窗口 (t - num_history*T, t - (num_history-1)*T]
        // ts_ptr[num_history] 对应 t (用于记录 <= t 的最后一条边)
        ts_ptr.resize(num_history + 1);
        for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
        {
            it->resize(indptr.size() - 1); // 为每个节点分配空间
            // 使用 OpenMP 并行初始化 ts_ptr，每个节点的初始指针指向其邻接列表的开始
#pragma omp parallel for
            for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                (*it)[itt - indptr.begin()] = *itt;
        }
    }

    // 重置 ts_ptr 的状态，将其恢复到初始状态
    void reset()
    {
        for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
        {
            it->resize(indptr.size() - 1);
#pragma omp parallel for
            for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                (*it)[itt - indptr.begin()] = *itt;
        }
    }


    // 更新时间指针 ts_ptr[slc][n]
    // 对于给定的根节点 root_nodes 和它们的时间戳 root_ts，
    // 以及一个时间偏移 offset，更新 ts_ptr[slc][n] 指向节点 n 的邻接边中
    // 时间戳 <= root_ts[i] + offset 的最后一条边。
    // 这个函数用于加速后续的邻居搜索。
    void update_ts_ptr(
        int slc,
        std::vector<NodeIDType>& root_nodes,
        std::vector<TimeStampType>& root_ts,
        float neg_offset
    )
    {
        // 使用 OpenMP 并行处理不同的根节点
        // schedule(static, chunk_size) 静态调度，chunk_size 大致均分任务
        float root_node_size_float = static_cast<float>(root_nodes.size());
        int chunk_size = int(ceil(root_node_size_float / num_threads));
#pragma omp parallel for schedule(static, chunk_size)
        for (std::vector<NodeIDType>::size_type i = 0; i < root_nodes.size(); i++)
        {
            NodeIDType n = root_nodes[i]; // 当前处理的根节点 e.g. root_nodes = [0, 3, 5]
            omp_set_lock(&(ts_ptr_lock[n])); // 获取该节点的锁，防止并发修改 ts_ptr[slc][n]
            auto& node_specific_ptr = ts_ptr[slc][n];
            for (std::vector<EdgeIDType>::size_type j = node_specific_ptr; j < indptr[n + 1]; j++)
            {
                // e.g. n = 0: ts_ptr[slc][0] -> indptr[1]
                // std::cout << "comparing " << ts[j] << " with " << root_ts[i] << std::endl;
                if (ts[j] > (root_ts[i] + neg_offset - 1e-7f)) // 如果当前边的时间戳已经超过了目标时间上限
                {
                    if (j != node_specific_ptr) // 如果指针移动了
                        node_specific_ptr = j - 1; // 更新指针到前一条满足条件的边
                    break; // 找到上界，停止搜索
                }
                if (j == indptr[n + 1] - 1) // 如果搜索到了节点 n 的最后一条边
                {
                    node_specific_ptr = j; // 更新指针到最后这条边
                }
            }
            omp_unset_lock(&(ts_ptr_lock[n])); // 释放锁
        }
    }



    // 内联函数，用于将各个线程采样到的 COO 格式的边信息合并到最终的 TemporalGraphBlock 中
    inline void combine_coo(
        TemporalGraphBlock& tgb,
        std::vector<ThreadSamplingBuffers>& all_threads_data
    ) // _out_nodes[tid] 是第 tid 个线程处理的源节点数量
    {


        std::vector<EdgeIDType> cum_row, cum_col; // 用于计算各个线程结果合并时的偏移量
        cum_row.push_back(0); // 源节点ID的累积偏移
        cum_col.push_back(0); // 目标节点ID和边相关属性的累积偏移


        // 计算每个线程的偏移量
        for (const ThreadSamplingBuffers& buffers_for_one_thread : all_threads_data) {
            // NodeIDType all_thread_level_local_src_id = cum_row.back(); // 上一个线程的源节点ID偏移
            // NodeIDType all_thread_level_local_dst_id = cum_col.back(); // 上一个线程的目标节点ID偏移
            cum_row.push_back(cum_row.back() + buffers_for_one_thread.single_thead_src_num); // 第 tid 个线程处理了 _out_nodes[tid] 个源节点
            NodeIDType single_thread_level_max_local_dst_id = buffers_for_one_thread.col_data.size(); // 当前线程处理的源节点数量,100*neighs
            cum_col.push_back(cum_col.back() + single_thread_level_max_local_dst_id); // 第 tid 个线程采样了 _col[tid]->size() 条边
        }


        int num_root_nodes = tgb.nodes.size(); // 当前层已有的节点数 (上一层采样到的节点)
        // 为合并后的结果预分配空间
        NodeIDType all_thread_n_edges = cum_col.back(); // = edge num
        tgb.row.resize(all_thread_n_edges);
        tgb.col.resize(all_thread_n_edges);
        tgb.eid.resize(all_thread_n_edges);
        // 时间戳和节点列表需要额外加上 num_root_nodes，因为它们包含了上一层的节点
        tgb.ts.resize(all_thread_n_edges + num_root_nodes); //这一层所有全部节点+num_root_nodes
        tgb.dts.resize(all_thread_n_edges + num_root_nodes);
        tgb.nodes.resize(all_thread_n_edges + num_root_nodes);

        // 使用 OpenMP 并行合并各个线程的结果
        // schedule(static, 1) 每个线程处理一个 tid 的合并任务
#pragma omp parallel for schedule(static, 1)
        // for (int tid = 0; tid < num_threads; tid++)
        for (size_t tid = 0; tid < all_threads_data.size(); ++tid)
        {
            ThreadSamplingBuffers& buffers_for_one_thread = all_threads_data[tid];
            // 调整当前线程的 row 和 col 数组中的相对ID，使其变为全局ID
            // v + cum_row[tid] : 将线程内相对源节点ID加上偏移，得到在合并后块内的源节点ID
            auto rowItBegin = buffers_for_one_thread.row_data.begin();
            auto rowItEnd = buffers_for_one_thread.row_data.end();
            auto colItBegin = buffers_for_one_thread.col_data.begin();
            auto colItEnd = buffers_for_one_thread.col_data.end();

            std::transform(rowItBegin, rowItEnd, rowItBegin, [&](auto& v) { return v + cum_row[tid]; });
            // v + cum_col[tid] + num_root_nodes : 将线程内相对目标节点ID加上偏移和上一层节点数，得到在合并后块内的目标节点ID
            std::transform(colItBegin, colItEnd, colItBegin, [&](auto& v) { return v + cum_col[tid] + num_root_nodes; });

            // 将调整后的数据拷贝到 _ret 的相应位置
            std::copy(rowItBegin, rowItEnd, tgb.row.begin() + cum_col[tid]);
            std::copy(colItBegin, colItEnd, tgb.col.begin() + cum_col[tid]);
            std::copy(buffers_for_one_thread.eid_data.begin(), buffers_for_one_thread.eid_data.end(), tgb.eid.begin() + cum_col[tid]);
            // ts, dts, nodes 的拷贝位置需要加上 num_root_nodes
            std::copy(buffers_for_one_thread.nodes_data.begin(), buffers_for_one_thread.nodes_data.end(), tgb.nodes.begin() + cum_col[tid] + num_root_nodes);
            std::copy(buffers_for_one_thread.ts_data.begin(), buffers_for_one_thread.ts_data.end(), tgb.ts.begin() + cum_col[tid] + num_root_nodes);
            std::copy(buffers_for_one_thread.dts_data.begin(), buffers_for_one_thread.dts_data.end(), tgb.dts.begin() + cum_col[tid] + num_root_nodes);//前面一部分没有东西

        }
        tgb.dim_in = tgb.nodes.size();    // 更新输入节点总数
        tgb.dim_out = cum_row.back();      // 更新输出节点总数 (即当前层采样操作的源节点总数)
    }

    // 对一层进行邻居采样
    // root_nodes, root_ts: 当前层进行采样的源节点及其时间戳
    // neighs: 需要采样的邻居数量
    // use_ptr: 是否使用预计算的 ts_ptr 来加速搜索
    // from_root: 当前采样是否是从整个采样过程的初始根节点开始 (i.e., 第0层采样)
    void sample_layer(
        std::vector<NodeIDType>& root_nodes,
        std::vector<TimeStampType>& root_ts,
        int neighs,
        bool use_ptr,
        bool from_root
    )
    {

        double t_s = omp_get_wtime(); // 记录开始时间
        std::vector<NodeIDType>* sample_base_nodes;    // 指向实际使用的源节点列表
        std::vector<TimeStampType>* sample_base_ts;    // 指向实际使用的源节点时间戳列表

        if (from_root) // 如果是从初始根节点开始采样
        {
            sample_base_nodes = &root_nodes; // 直接使用传入的 root_nodes
            sample_base_ts = &root_ts;
        }

        double t_ptr_s = omp_get_wtime();
        if (use_ptr)
            update_ts_ptr(num_history, *sample_base_nodes, *sample_base_ts, 0);
        // ts_ptr size : num_history + 1
        tgbs[0].ptr_time += omp_get_wtime() - t_ptr_s;
        for (int i = 0; i < num_history; i++)
        {
            if (!from_root)
            {
                int sample_base_layer = static_cast<int>(tgbs.size()) - i - num_history - 1;
                // if num_history = 3, ret.size = 3(impossible), 6, layer=2
                // impossible: if ret.size = 3, i = 0, 1, 2 -> sample_base_layer = 3 - (0,1,2) - 3 - 1 = -1, -2, -3
                // if ret.size = 6, i = 0, 1, 2 -> sample_base_layer = 6 - (0,1,2) - 3 - 1 = 2, 1, 0
                // if ret.size = 9, i = 0, 1, 2 -> sample_base_layer = 9 - (0,1,2) - 3 - 1 = 5, 4, 3
                sample_base_nodes = &(tgbs[sample_base_layer].nodes);
                sample_base_ts = &(tgbs[sample_base_layer].ts);
            }
            TimeStampType offset = -i * window_duration;
            t_ptr_s = omp_get_wtime();
            if ((use_ptr) && (std::abs(window_duration) > 1e-7f))
                update_ts_ptr(num_history - 1 - i, *sample_base_nodes, *sample_base_ts, offset - window_duration);
            // if num_history = 2, num_history - 1 - i = 2 - 1 - i = 1 - i: 1, 0
            tgbs[0].ptr_time += omp_get_wtime() - t_ptr_s;
            int num_sample_base_nodes = (*sample_base_nodes).size();
            int nodes_per_chunk = int((num_sample_base_nodes + num_threads - 1) / num_threads);
            // int reserve_capacity = int(ceil((*sample_base_nodes).size() / num_threads)) * neighs; // bug found by leizhao
            int reserve_capacity = nodes_per_chunk * neighs; // 每个线程预留的容量
            std::vector<ThreadSamplingBuffers> per_thread_buffers(num_threads, ThreadSamplingBuffers(reserve_capacity));

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                ThreadSamplingBuffers& my_buffers = per_thread_buffers[tid]; // 获取当前线程的缓冲区集合引用

                unsigned int loc_seed = tid;
#pragma omp for schedule(static, nodes_per_chunk)
                for (std::vector<NodeIDType>::size_type j = 0; j < (*sample_base_nodes).size(); j++)
                {
                    NodeIDType n = (*sample_base_nodes)[j];
                    // if (tid == 16)
                    //     std::cout << _out_node[tid] << " " <<j << " " << n << std::endl;
                    TimeStampType src_ts = (*sample_base_ts)[j];
                    EdgeIDType s_search, e_search;
                    if (use_ptr)
                    {
                        s_search = ts_ptr[num_history - 1 - i][n];
                        e_search = ts_ptr[num_history - i][n];
                    }
                    else
                    {
                        // search for start and end pointer
                        double t_search_s = omp_get_wtime();
                        if (num_history == 1)
                        {
                            // TGAT style: sample from all past edges up to src_ts
                            s_search = indptr[n];
                            auto e_it = std::upper_bound(ts.begin() + indptr[n], ts.begin() + indptr[n + 1], src_ts);
                            e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                        }
                        else
                        {
                            // DySAT stylesample from a specific time window
                            auto s_it = std::upper_bound(ts.begin() + indptr[n], ts.begin() + indptr[n + 1], src_ts + offset - window_duration);
                            s_search = std::max(int(s_it - ts.begin()) - 1, indptr[n]);
                            auto e_it = std::upper_bound(ts.begin() + indptr[n], ts.begin() + indptr[n + 1], src_ts + offset);
                            e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                        }
                        if (tid == 0)
                            tgbs[0].search_time += omp_get_wtime() - t_search_s;
                    }
                    // std::cout << n << " " << s_search << " " << e_search << std::endl;
                    double t_sample_s = omp_get_wtime();
                    if ((recent) || (e_search - s_search < neighs))
                    {
                        // no sampling, pick recent neighbors
                        for (EdgeIDType picked = e_search; picked > std::max(s_search, e_search - neighs); picked--)
                        {
                            if (ts[picked] < src_ts + offset - 1e-7f)
                            {
                                my_buffers.add_dst(picked, src_ts, indices, eid, ts, prop_time); // this can be slow
                            }
                        }
                    }
                    else
                    {
                        // random sampling within ptr
                        for (int _i = 0; _i < neighs; _i++)
                        {
                            EdgeIDType picked = s_search + rand_r(&loc_seed) % (e_search - s_search + 1);
                            if (ts[picked] < src_ts + offset - 1e-7f)
                            {
                                my_buffers.add_dst(picked, src_ts, indices, eid, ts, prop_time); // this can be slow
                            }
                        }
                    }
                    my_buffers.single_thead_src_num += 1;
                    if (tid == 0)
                        tgbs[0].sample_time += omp_get_wtime() - t_sample_s;
                }
            }
            double t_coo_s = omp_get_wtime();
            // if ret.size() = 3, i = 0, 1, 2, index = 3-1-i = 2, 1, 0
            // if ret.size() = 6, i = 0, 1, 2, index = 6-1-i = 5, 4, 3
            // if ret.size() = 9, i = 0, 1, 2, index = 9-1-i = 8, 7, 6

            TemporalGraphBlock& element_reference = tgbs[tgbs.size() - 1 - i];
            element_reference.nodes.insert(element_reference.nodes.end(), sample_base_nodes->begin(), sample_base_nodes->end());
            element_reference.ts.insert(element_reference.ts.end(), sample_base_ts->begin(), sample_base_ts->end());
            element_reference.dts.resize(sample_base_nodes->size());
            combine_coo(element_reference, per_thread_buffers);
            tgbs[0].coo_time += omp_get_wtime() - t_coo_s;
        }
        tgbs[0].tot_time += omp_get_wtime() - t_s;
    }



    void sample(std::vector<NodeIDType>& root_nodes, std::vector<TimeStampType>& root_ts)
    {
        // a weird bug, dgl library seems to modify the total number of threads
        omp_set_num_threads(num_threads);
        tgbs.resize(0);
        for (int i = 0; i < num_layers; i++)
        {
            tgbs.resize(tgbs.size() + num_history); // if num_history = 3, ret.size = 3, 6
            bool use_ptr = i == 0 || (prop_time && num_history == 1) || recent;
            bool from_root = i == 0;
            sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, from_root);
        }
    }
};





template<typename T>
inline py::array vec2npy(const std::vector<T>& vec)
{
    // need to let python garbage collector handle C++ vector memory 
    // see https://github.com/pybind/pybind11/issues/1042
    auto v = new std::vector<T>(vec);
    auto capsule = py::capsule(v, [](void* v)
        { delete reinterpret_cast<std::vector<T>*>(v); });
    return py::array(v->size(), v->data(), capsule);
    // return py::array(vec.size(), vec.data());
}

PYBIND11_MODULE(sampler_core, m)
{
    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(
            py::init<
            std::vector<NodeIDType> &,
            std::vector<NodeIDType> &,
            std::vector<EdgeIDType> &,
            std::vector<TimeStampType> &,
            std::vector<TimeStampType> &,
            std::vector<NodeIDType> &,
            NodeIDType,
            NodeIDType
            >()
        )
        .def("row", [](const TemporalGraphBlock& tgb) { return vec2npy(tgb.row); })
        .def("col", [](const TemporalGraphBlock& tgb) { return vec2npy(tgb.col); })
        .def("eid", [](const TemporalGraphBlock& tgb) { return vec2npy(tgb.eid); })
        .def("ts", [](const TemporalGraphBlock& tgb) { return vec2npy(tgb.ts); })
        .def("dts", [](const TemporalGraphBlock& tgb) { return vec2npy(tgb.dts); })
        .def("nodes", [](const TemporalGraphBlock& tgb) { return vec2npy(tgb.nodes); })
        .def("dim_in", [](const TemporalGraphBlock& tgb) { return tgb.dim_in; })
        .def("dim_out", [](const TemporalGraphBlock& tgb) { return tgb.dim_out; })
        .def("tot_time", [](const TemporalGraphBlock& tgb) { return tgb.tot_time; })
        .def("ptr_time", [](const TemporalGraphBlock& tgb) { return tgb.ptr_time; })
        .def("search_time", [](const TemporalGraphBlock& tgb) { return tgb.search_time; })
        .def("sample_time", [](const TemporalGraphBlock& tgb) { return tgb.sample_time; })
        .def("coo_time", [](const TemporalGraphBlock& tgb) { return tgb.coo_time; });
    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(
            py::init<
            std::vector<EdgeIDType> &,
            std::vector<EdgeIDType> &,
            std::vector<EdgeIDType> &,
            std::vector<TimeStampType> &,
            int,
            int,
            int,
            std::vector<int> &,
            bool,
            bool,
            int,
            TimeStampType
            >()
        )
        .def("sample", &ParallelSampler::sample)
        .def("reset", &ParallelSampler::reset)
        .def("get_ret", [](const ParallelSampler& ps) { return ps.tgbs; });
}
