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

    std::vector<TemporalGraphBlock> ret; // 存储多层采样结果的 TemporalGraphBlock 列表, 传回python

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
    void update_ts_ptr(int slc, std::vector<NodeIDType>& root_nodes,
        std::vector<TimeStampType>& root_ts, float offset)
    {
        // 使用 OpenMP 并行处理不同的根节点
        // schedule(static, chunk_size) 静态调度，chunk_size 大致均分任务
#pragma omp parallel for schedule(static, int(ceil(static_cast<float>(root_nodes.size()) / num_threads)))
        for (std::vector<NodeIDType>::size_type i = 0; i < root_nodes.size(); i++)
        {
            NodeIDType n = root_nodes[i]; // 当前处理的根节点 e.g. root_nodes = [0, 3, 5]
            omp_set_lock(&(ts_ptr_lock[n])); // 获取该节点的锁，防止并发修改 ts_ptr[slc][n]
            for (std::vector<EdgeIDType>::size_type j = ts_ptr[slc][n]; j < indptr[n + 1]; j++)
            {
                // e.g. n = 0: ts_ptr[slc][0] -> indptr[1]
                // std::cout << "comparing " << ts[j] << " with " << root_ts[i] << std::endl;
                if (ts[j] > (root_ts[i] + offset - 1e-7f)) // 如果当前边的时间戳已经超过了目标时间上限
                {
                    if (j != ts_ptr[slc][n]) // 如果指针移动了
                        ts_ptr[slc][n] = j - 1; // 更新指针到前一条满足条件的边
                    break; // 找到上界，停止搜索
                }
                if (j == indptr[n + 1] - 1) // 如果搜索到了节点 n 的最后一条边
                {
                    ts_ptr[slc][n] = j; // 更新指针到最后这条边
                }
            }
            omp_unset_lock(&(ts_ptr_lock[n])); // 释放锁
        }
    }

    // 内联函数，用于向当前线程的缓冲区中添加一个采样到的邻居信息
    inline void add_neighbor(
        std::vector<NodeIDType>* _row,
        std::vector<NodeIDType>* _col,
        std::vector<EdgeIDType>* _eid,
        std::vector<TimeStampType>* _ts,
        std::vector<TimeStampType>* _dts,
        std::vector<NodeIDType>* _nodes,
        EdgeIDType& picked,
        TimeStampType& src_ts,
        int& _out_node
    )
    {
        _row->push_back(_out_node);        // 源节点在当前层输出节点中的相对ID
        _col->push_back(_nodes->size());// 目标节点在下一层输入节点中的相对ID (即当前已采样的邻居数)
        _eid->push_back(eid[picked]);        // 边的原始ID
        if (prop_time)                  // 如果 prop_time 为 true 是否将父节点的时间戳传播给子节点采样的边
            _ts->push_back(src_ts);     // 使用源节点的时间戳作为这条采样边的时间戳
        else
            _ts->push_back(ts[picked]);      // 否则使用边的原始时间戳
        _dts->push_back(src_ts - ts[picked]);// 计算时间差
        _nodes->push_back(indices[picked]);  // 添加邻居节点的原始ID
        // _row.push_back(0);
        // _col.push_back(0);
        // _eid.push_back(0);
        // if (prop_time)
        //     _ts.push_back(src_ts);
        // else
        //     _ts.push_back(10000);
        // _nodes.push_back(100);
    }

    // 内联函数，用于将各个线程采样到的 COO 格式的边信息合并到最终的 TemporalGraphBlock 中
    inline void combine_coo(
        TemporalGraphBlock& _ret,
        std::vector<NodeIDType>** _row,
        std::vector<NodeIDType>** _col,
        std::vector<EdgeIDType>** _eid,
        std::vector<TimeStampType>** _ts,
        std::vector<TimeStampType>** _dts,
        std::vector<NodeIDType>** _nodes,
        std::vector<int>& _out_nodes) // _out_nodes[tid] 是第 tid 个线程处理的源节点数量
    {
        std::vector<EdgeIDType> cum_row, cum_col; // 用于计算各个线程结果合并时的偏移量
        cum_row.push_back(0); // 源节点ID的累积偏移
        cum_col.push_back(0); // 目标节点ID和边相关属性的累积偏移

        // 计算每个线程的偏移量
        for (int tid = 0; tid < num_threads; tid++)
        {
            // std::cout<<tid<<" here "<<_out_nodes[tid]<<std::endl;  
            cum_row.push_back(cum_row.back() + _out_nodes[tid]); // 第 tid 个线程处理了 _out_nodes[tid] 个源节点
            cum_col.push_back(cum_col.back() + _col[tid]->size()); // 第 tid 个线程采样了 _col[tid]->size() 条边
        }
        int num_root_nodes = _ret.nodes.size(); // 当前层已有的节点数 (上一层采样到的节点)
        // 为合并后的结果预分配空间
        _ret.row.resize(cum_col.back());
        _ret.col.resize(cum_col.back());
        _ret.eid.resize(cum_col.back());
        // 时间戳和节点列表需要额外加上 num_root_nodes，因为它们包含了上一层的节点
        _ret.ts.resize(cum_col.back() + num_root_nodes);
        _ret.dts.resize(cum_col.back() + num_root_nodes);
        _ret.nodes.resize(cum_col.back() + num_root_nodes);

        // 使用 OpenMP 并行合并各个线程的结果
        // schedule(static, 1) 每个线程处理一个 tid 的合并任务
#pragma omp parallel for schedule(static, 1)
        for (int tid = 0; tid < num_threads; tid++)
        {
            // 调整当前线程的 row 和 col 数组中的相对ID，使其变为全局ID
            // v + cum_row[tid] : 将线程内相对源节点ID加上偏移，得到在合并后块内的源节点ID
            auto rowItBegin = _row[tid]->begin();
            auto rowItEnd = _row[tid]->end();
            auto colItBegin = _col[tid]->begin();
            auto colItEnd = _col[tid]->end();
            std::transform(rowItBegin, rowItEnd, rowItBegin,
                [&](auto& v) { return v + cum_row[tid]; });
            // v + cum_col[tid] + num_root_nodes : 将线程内相对目标节点ID加上偏移和上一层节点数，得到在合并后块内的目标节点ID
            std::transform(colItBegin, colItEnd, colItBegin,
                [&](auto& v) { return v + cum_col[tid] + num_root_nodes; });

            // 将调整后的数据拷贝到 _ret 的相应位置
            std::copy(rowItBegin, rowItEnd, _ret.row.begin() + cum_col[tid]);
            std::copy(colItBegin, colItEnd, _ret.col.begin() + cum_col[tid]);
            std::copy(_eid[tid]->begin(), _eid[tid]->end(), _ret.eid.begin() + cum_col[tid]);
            // ts, dts, nodes 的拷贝位置需要加上 num_root_nodes
            std::copy(_ts[tid]->begin(), _ts[tid]->end(), _ret.ts.begin() + cum_col[tid] + num_root_nodes);
            std::copy(_dts[tid]->begin(), _dts[tid]->end(), _ret.dts.begin() + cum_col[tid] + num_root_nodes);
            std::copy(_nodes[tid]->begin(), _nodes[tid]->end(), _ret.nodes.begin() + cum_col[tid] + num_root_nodes);

            // 释放线程的临时缓冲区内存
            delete _row[tid];
            delete _col[tid];
            delete _eid[tid];
            delete _ts[tid];
            delete _dts[tid];
            delete _nodes[tid];
        }
        _ret.dim_in = _ret.nodes.size();    // 更新输入节点总数
        _ret.dim_out = cum_row.back();      // 更新输出节点总数 (即当前层采样操作的源节点总数)
    }

    // 对一层进行邻居采样
    // _root_nodes, _root_ts: 当前层进行采样的源节点及其时间戳
    // neighs: 需要采样的邻居数量
    // use_ptr: 是否使用预计算的 ts_ptr 来加速搜索
    // from_root: 当前采样是否是从整个采样过程的初始根节点开始 (i.e., 第0层采样)
    void sample_layer(
        std::vector<NodeIDType>& _root_nodes,
        std::vector<TimeStampType>& _root_ts,
        int neighs,
        bool use_ptr,
        bool from_root
    )
    {

        double t_s = omp_get_wtime(); // 记录开始时间
        std::vector<NodeIDType>* root_nodes;    // 指向实际使用的源节点列表
        std::vector<TimeStampType>* root_ts;    // 指向实际使用的源节点时间戳列表

        if (from_root) // 如果是从初始根节点开始采样
        {
            root_nodes = &_root_nodes; // 直接使用传入的 _root_nodes
            root_ts = &_root_ts;
        }
        // else (非 from_root) 的情况在下面的循环中处理

        double t_ptr_s = omp_get_wtime();
        if (use_ptr)
            update_ts_ptr(num_history, *root_nodes, *root_ts, 0);
        ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
        for (int i = 0; i < num_history; i++)
        {
            if (!from_root)
            {
                int root_layer = static_cast<int>(ret.size()) - i - num_history - 1;
                // if num_history = 2, ret_size = 5, root_layer = 5 - 0 - 2 - 1 = 2, 5 - 1 - 2 - 1 = 1
                root_nodes = &(ret[root_layer].nodes);
                root_ts = &(ret[root_layer].ts);
            }
            TimeStampType offset = -i * window_duration;
            t_ptr_s = omp_get_wtime();
            if ((use_ptr) && (std::abs(window_duration) > 1e-7f))
                update_ts_ptr(num_history - 1 - i, *root_nodes, *root_ts, offset - window_duration);
            // if num_history = 2, num_history - 1 - i = 2 - 1 - i = 1 - i: 1, 0
            ret[0].ptr_time += omp_get_wtime() - t_ptr_s;
            // 为每个线程创建临时缓冲区，用于存储采样结果
            std::vector<NodeIDType>* _row[num_threads];
            std::vector<NodeIDType>* _col[num_threads];
            std::vector<EdgeIDType>* _eid[num_threads];
            std::vector<TimeStampType>* _ts[num_threads];
            std::vector<TimeStampType>* _dts[num_threads];
            std::vector<NodeIDType>* _nodes[num_threads];
            std::vector<int> _out_node(num_threads, 0);
            int reserve_capacity = int(ceil((*root_nodes).size() / num_threads)) * neighs;




#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                unsigned int loc_seed = tid;
                _row[tid] = new std::vector<NodeIDType>;
                _col[tid] = new std::vector<NodeIDType>;
                _eid[tid] = new std::vector<EdgeIDType>;
                _ts[tid] = new std::vector<TimeStampType>;
                _dts[tid] = new std::vector<TimeStampType>;
                _nodes[tid] = new std::vector<NodeIDType>;
                _row[tid]->reserve(reserve_capacity);
                _col[tid]->reserve(reserve_capacity);
                _eid[tid]->reserve(reserve_capacity);
                _ts[tid]->reserve(reserve_capacity);
                _dts[tid]->reserve(reserve_capacity);
                _nodes[tid]->reserve(reserve_capacity);
                // #pragma omp critical
                //                     std::cout<<tid<<" sampling: "<<root_nodes->size()<<" "<<int(ceil((*root_nodes).size() / num_threads))<<std::endl;
#pragma omp for schedule(static, int(ceil(static_cast<float>((*root_nodes).size()) / num_threads)))
                for (std::vector<NodeIDType>::size_type j = 0; j < (*root_nodes).size(); j++)
                {
                    NodeIDType n = (*root_nodes)[j];
                    // if (tid == 16)
                    //     std::cout << _out_node[tid] << " " <<j << " " << n << std::endl;
                    TimeStampType node_ts = (*root_ts)[j];
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
                            // TGAT style: sample from all past edges up to node_ts
                            s_search = indptr[n];
                            auto e_it = std::upper_bound(ts.begin() + indptr[n],
                                ts.begin() + indptr[n + 1], node_ts);
                            e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                        }
                        else
                        {
                            // DySAT stylesample from a specific time window
                            auto s_it = std::upper_bound(ts.begin() + indptr[n],
                                ts.begin() + indptr[n + 1],
                                node_ts + offset - window_duration);
                            s_search = std::max(int(s_it - ts.begin()) - 1, indptr[n]);
                            auto e_it = std::upper_bound(ts.begin() + indptr[n],
                                ts.begin() + indptr[n + 1], node_ts + offset);
                            e_search = std::max(int(e_it - ts.begin()) - 1, s_search);
                        }
                        if (tid == 0)
                            ret[0].search_time += omp_get_wtime() - t_search_s;
                    }
                    // std::cout << n << " " << s_search << " " << e_search << std::endl;
                    double t_sample_s = omp_get_wtime();
                    if ((recent) || (e_search - s_search < neighs))
                    {
                        // no sampling, pick recent neighbors
                        for (EdgeIDType picked = e_search; picked > std::max(s_search, e_search - neighs); picked--)
                        {
                            if (ts[picked] < node_ts + offset - 1e-7f)
                            {
                                add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid],
                                    _dts[tid], _nodes[tid], picked, node_ts, _out_node[tid]);
                            }
                        }
                    }
                    // std::vector<int> _out_node(num_threads, 0);
                    // add_neighbor logic
                    // _row->push_back(row_id);        // 源节点在当前层输出节点中的相对ID
                    // _col->push_back(_nodes->size());// 目标节点在下一层输入节点中的相对ID (即当前已采样的邻居数)
                    // _eid->push_back(eid[picked]);        // 边的原始ID
                    // if (prop_time)                  // 如果 prop_time 为 true 是否将父节点的时间戳传播给子节点采样的边
                    //     _ts->push_back(src_ts);     // 使用源节点的时间戳作为这条采样边的时间戳
                    // else
                    //     _ts->push_back(ts[picked]);      // 否则使用边的原始时间戳
                    // _dts->push_back(src_ts - ts[picked]);// 计算时间差
                    // _nodes->push_back(indices[picked]);  // 添加邻居节点的原始ID
                    else
                    {
                        // random sampling within ptr
                        for (int _i = 0; _i < neighs; _i++)
                        {
                            EdgeIDType picked = s_search + rand_r(&loc_seed) % (e_search - s_search + 1);
                            if (ts[picked] < node_ts + offset - 1e-7f)
                            {
                                add_neighbor(_row[tid], _col[tid], _eid[tid], _ts[tid],
                                    _dts[tid], _nodes[tid], picked, node_ts, _out_node[tid]);
                            }
                        }
                    }
                    _out_node[tid] += 1;
                    if (tid == 0)
                        ret[0].sample_time += omp_get_wtime() - t_sample_s;
                }
            }
            double t_coo_s = omp_get_wtime();
            ret[ret.size() - 1 - i].ts.insert(ret[ret.size() - 1 - i].ts.end(),
                root_ts->begin(), root_ts->end());
            ret[ret.size() - 1 - i].nodes.insert(ret[ret.size() - 1 - i].nodes.end(),
                root_nodes->begin(), root_nodes->end());
            ret[ret.size() - 1 - i].dts.resize(root_nodes->size());
            combine_coo(ret[ret.size() - 1 - i], _row, _col, _eid, _ts, _dts, _nodes, _out_node);
            ret[0].coo_time += omp_get_wtime() - t_coo_s;
        }
        ret[0].tot_time += omp_get_wtime() - t_s;
    }

    // python example
    // def sampler_wrapper(no_neg, root_nodes, time_sample, sampler: ParallelSampler|None, ts):
    //     if no_neg:
    //         pos_root_end = root_nodes.shape[0] * 2 // 3
    //         sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
    //     else:
    //         sampler.sample(root_nodes, ts)
    //     ret = sampler.get_ret()
    //     if time_sample is not None:
    //         time_sample += ret[0].sample_time()
    //     return ret, time_sample

    void sample(std::vector<NodeIDType>& root_nodes, std::vector<TimeStampType>& root_ts)
    {
        // a weird bug, dgl library seems to modify the total number of threads
        omp_set_num_threads(num_threads);
        ret.resize(0);
        bool use_ptr = false;
        // 如果(first_layer) || ((prop_time) && num_history == 1) || (recent) 不符合, use_ptr_true
        for (int i = 0; i < num_layers; i++)
        {
            ret.resize(ret.size() + num_history);
            if (i == 0 || ((prop_time) && num_history == 1) || recent)
            {
                use_ptr = true;
            }
            else
                use_ptr = false;

            if (i == 0)
                sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, true);
            else
                sample_layer(root_nodes, root_ts, num_neighbors[i], use_ptr, false);
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
        .def("get_ret", [](const ParallelSampler& ps) { return ps.ret; });
}
