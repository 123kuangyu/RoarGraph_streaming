// index bindings of bipartite_index for python
// python3 setup.py bdist_wheel
#include <omp.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include "../thirdparty/DiskANN/include/omp_utils.h"
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <filesystem>
#include <unistd.h>

#include "index_bipartite.h"
#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"

namespace py = pybind11;

template <typename T>

class IndexRoarGraph {
private:
    std::unique_ptr<efanna2e::IndexBipartite> index_;
    uint32_t dim;
    uint32_t total_num;
    uint32_t base_num, base_dim, sq_num, sq_dim;
    float* base_data_ = nullptr;
    float* sq_data_ = nullptr;

public:
    IndexRoarGraph(uint32_t dimension, uint32_t total_points, efanna2e::Metric metric,
         const std::string& base_file, const std::string& sq_file) {
        dim = dimension;
        total_num = total_points;
        index_ = std::make_unique<efanna2e::IndexBipartite>(dim, total_num, metric, nullptr);
        load_base_data(base_file, sq_file);
    }
    
    void load_base_data(const std::string& base_file, const std::string& sq_file) {
        efanna2e::load_meta<float>(base_file.c_str(), base_num, base_dim);
        efanna2e::load_meta<float>(sq_file.c_str(), sq_num, sq_dim);
        if(base_dim != dim) {
            throw std::runtime_error("Data dimensions do not match");
        }

        efanna2e::load_data<float>(base_file.c_str(), base_num, base_dim, base_data_);
        efanna2e::load_data<float>(sq_file.c_str(), sq_num, sq_dim, sq_data_); //训练查询数据
        index_->LoadSearchNeededData(base_file.c_str(), sq_file.c_str()); 
        index_->SetBaseData(base_data_);

    }

    void build_roar_graph(uint32_t initial_num, const py::dict& params) {

        efanna2e::Parameters build_params;
        build_params.Set<uint32_t>("M_sq", params["M_sq"].cast<uint32_t>());
        build_params.Set<uint32_t>("M_pjbp", params["M_pjbp"].cast<uint32_t>());
        build_params.Set<uint32_t>("L_pjpq", params["L_pjpq"].cast<uint32_t>());
        build_params.Set<uint32_t>("num_threads", params["num_threads"].cast<uint32_t>());
        
        omp_set_num_threads(params["num_threads"].cast<uint32_t>());
        index_->BuildRoarGraph(sq_num, sq_data_, initial_num, base_data_, build_params);
        // 保存初始图
        // std::string init_index_path = "t2i_10M_roar10.index";
        // index_->SaveProjectionGraph(init_index_path.c_str());

    }
    

    void insert_into_roar_graph(size_t start_idx, size_t end_idx, const py::dict& params) {
        // 获取插入数据的缓冲区
        size_t num_insert = end_idx - start_idx;
        // 设置插入参数
        efanna2e::Parameters insert_params;
        insert_params.Set<uint32_t>("M_sq", params["M_sq"].cast<uint32_t>());
        insert_params.Set<uint32_t>("M_pjbp", params["M_pjbp"].cast<uint32_t>());
        insert_params.Set<uint32_t>("L_pjpq", params["L_pjpq"].cast<uint32_t>());
        insert_params.Set<uint32_t>("num_threads", params["num_threads"].cast<uint32_t>());
        insert_params.Set<uint32_t>("L_pq", params["L_pq"].cast<uint32_t>());
   
        index_->InitVisitedListPool(params["num_threads"].cast<uint32_t>());
        // float* aligned_data_sq = efanna2e::data_align(sq_data_, sq_num, sq_dim);
        float* aligned_data_sq = sq_data_;
        index_->SetBaseData(base_data_);
        
        //当前批次的id
        std::vector<size_t> ids(num_insert);
        for(size_t i = 0; i < num_insert; i++) {
            ids[i] = start_idx + i;
        }   
        // 执行插入操作
        float* aligned_data = efanna2e::data_align(base_data_, base_num, base_dim);
         
        index_->InsertIntoRoarGraph(
            aligned_data + start_idx * dim,
            ids.data(),
            num_insert,
            insert_params
        );
        std::cout<<"cpp端插入完成！"<<std::endl;

        //计算插入后的投影图的入口节点
        index_->InsertCalculateProjectionep(end_idx);

        // std::string checkpoint_path = "roargraph_checkpoint_"+std::to_string(start_idx)+"_"+std::to_string(end_idx)+".index";
        // index_->SaveInsertProjectionGraph(checkpoint_path.c_str(), end_idx);
    }
    
    
    py::array_t<uint32_t> search_roar_graph(py::array_t<float> query_data, uint32_t k, const py::dict& params) {
        auto query_buf = query_data.request();
        size_t num_queries = query_buf.shape[0];
        float* query_ptr = static_cast<float*>(query_buf.ptr);
        
        // 设置搜索参数
        efanna2e::Parameters search_params;
        search_params.Set<uint32_t>("L_pq", params["L_pq"].cast<uint32_t>());
        search_params.Set<uint32_t>("num_threads", params["num_threads"].cast<uint32_t>());
        omp_set_num_threads(params["num_threads"].cast<uint32_t>());
        
        // 初始化访问列表池
        index_->InitVisitedListPool(params["num_threads"].cast<uint32_t>());

        // 为查询向量分配内存并复制数据
        float* query_data_copy = new float[num_queries * dim];
        memcpy(query_data_copy, query_ptr, num_queries * dim * sizeof(float));
        
        // 对齐数据
        unsigned aligned_dim = dim;
        float* queries_ptr_align = efanna2e::data_align(query_data_copy, num_queries, aligned_dim);
        // 此时 query_data_copy 已经被 data_align 释放（data_align已经实现）

        // 创建结果缓冲区
        std::vector<uint32_t> res(num_queries * k, 0);
        std::vector<std::vector<float>> res_dists(num_queries, std::vector<float>(k, 0.0));

        // 处理所有查询
        std::cout << "Processing " << num_queries << " queries..." << std::endl;
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < num_queries; i++) {
            auto ret_val = index_->SearchRoarGraph(
                queries_ptr_align + i * dim,
                k,
                i,
                search_params,
                res.data() + i * k,
                res_dists[i]
            );

        }

        // 创建返回的numpy数组
        std::vector<ssize_t> shape = {static_cast<ssize_t>(num_queries), static_cast<ssize_t>(k)};
        py::array_t<uint32_t> result(shape);
        auto buf = result.request();
        
        // 复制结果到numpy数组
        memcpy(buf.ptr, res.data(), sizeof(uint32_t) * num_queries * k);

        // 释放对齐的查询数据
        free(queries_ptr_align);  // 只释放一次，因为原始数据已经在data_align中被释放
        
        std::cout << "搜索完成，返回结果数组" << std::endl;
        return result;
        
    }


// py::array_t<uint32_t> search_roar_graph(py::array_t<float> query_data, uint32_t k, const py::dict& params) {
//     if (query_data.dtype() != py::dtype::of<float>()) {
//         throw std::runtime_error("query_data must be an array of floats");
//     }   
//     auto query_buf = query_data.request();
//     size_t num_queries = query_buf.shape[0];
//     float* query_ptr = static_cast<float*>(query_buf.ptr);
    
//     // 设置搜索参数
//     efanna2e::Parameters search_params;
//     search_params.Set<uint32_t>("L_pq", params["L_pq"].cast<uint32_t>());
//     search_params.Set<uint32_t>("num_threads", params["num_threads"].cast<uint32_t>());
//     omp_set_num_threads(params["num_threads"].cast<uint32_t>());
    
//     // 初始化访问列表池
//     index_->InitVisitedListPool(params["num_threads"].cast<uint32_t>());

//     try {
//         // 为所有查询向量创建对齐的副本
//         std::vector<float> aligned_queries(num_queries * dim);
//         memcpy(aligned_queries.data(), query_ptr, num_queries * dim * sizeof(float));
//         unsigned aligned_dim = dim;
//         float* queries_ptr_align = efanna2e::data_align(aligned_queries.data(), num_queries, aligned_dim);

//         // 创建返回结果数组
//         py::array_t<uint32_t> result({num_queries, k});
//         auto buf = result.mutable_unchecked<2>();  // 使用 mutable_unchecked 获取可写的缓冲区

//         // 为每个查询创建临时缓冲区
//         std::vector<unsigned> nn_indices(k);
//         std::vector<float> res_dists(k);

//         // 处理每个查询
//         for (size_t i = 0; i < num_queries; i++) {
//             size_t qid = i;
            
//             // 执行搜索
//             index_->SearchRoarGraph(
//                 queries_ptr_align + i * dim,  // 当前查询向量的指针
//                 k,
//                 qid,
//                 search_params,
//                 nn_indices.data(),
//                 res_dists
//             );

//             // 将结果复制到返回数组
//             for (size_t j = 0; j < k; j++) {
//                 buf(i, j) = nn_indices[j];
//             }

//             if (i == 0) {  // 只打印第一个查询的结果用于调试
//                 std::cout << "Search completed for query 0. First result: " << nn_indices[0] << std::endl;
//                 std::cout << "First distance: " << res_dists[0] << std::endl;
//             }
//         }

//         // 释放对齐的内存
//         free(queries_ptr_align);
        
//         return result;
//     } catch (const std::exception& e) {
//         std::cerr << "Error during search: " << e.what() << std::endl;
//         throw;  // 重新抛出异常，而不是返回不完整的结果
//     }
// }


    void load_learn_base_knn(const std::string& filename) {
        index_->LoadLearnBaseKNN(filename.c_str());
    }

    void load_projection_graph(const std::string& filename) {
        index_->LoadProjectionGraph(filename.c_str());
    }

    void load_bipartite_graph(const std::string& filename) {
        index_->Load(filename.c_str());
    }

    void save_projection_graph(const std::string& filename, size_t num_vectors) {
        index_->SaveInsertProjectionGraph(filename.c_str(), num_vectors);
    }

    void set_base_data(py::array_t<float> base_data) {
        auto buf = base_data.request();
        index_->SetBaseData(static_cast<float*>(buf.ptr));
    }

    void init_visited_list_pool(uint32_t num_threads) {
        index_->InitVisitedListPool(num_threads);
    }

    void load_search_needed_data(const std::string& base_file, const std::string& query_file) {
        index_->LoadSearchNeededData(base_file.c_str(), query_file.c_str());
    }

    ~IndexRoarGraph() {}
};

PYBIND11_MODULE(roargraph, m) {
    py::enum_<efanna2e::Metric>(m, "Metric")
        .value("L2", efanna2e::Metric::L2)
        .value("IP", efanna2e::Metric::INNER_PRODUCT)
        .value("COSINE", efanna2e::Metric::COSINE)
        .export_values();

    py::class_<IndexRoarGraph<float>>(m, "IndexRoarGraph")
        .def(py::init<uint32_t, uint32_t, efanna2e::Metric,  const std::string&, const std::string&>(),
             py::arg("dimension"),
             py::arg("total_points"),
             py::arg("metric_type"),
             py::arg("base_file"), 
             py::arg("sq_file")) 
        .def("build_roar_graph", &IndexRoarGraph<float>::build_roar_graph,
             py::arg("initial_num"),
             py::arg("params"))
        .def("insert_into_roar_graph", &IndexRoarGraph<float>::insert_into_roar_graph,
             py::arg("start_idx"),
             py::arg("end_idx"),
             py::arg("params"))
        .def("search_roar_graph", &IndexRoarGraph<float>::search_roar_graph, 
            py::arg("query_data"),
            py::arg("k"), 
            py::arg("params"))
        .def("load_bipartite_graph", &IndexRoarGraph<float>::load_bipartite_graph, py::arg("filename"))
        .def("load_learn_base_knn", &IndexRoarGraph<float>::load_learn_base_knn)
        .def("load_projection_graph", &IndexRoarGraph<float>::load_projection_graph, py::arg("filename"))
        .def("save_projection_graph", &IndexRoarGraph<float>::save_projection_graph, py::arg("filename"), py::arg("num_vectors"))
        .def("init_visited_list_pool", &IndexRoarGraph<float>::init_visited_list_pool);

    m.doc() = "Python bindings for RoarGraph Index implementation";
}