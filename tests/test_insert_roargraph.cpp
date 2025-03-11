#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_bipartite.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    // 1. 参数定义
    std::string base_data_file;           // 原始基础数据
    std::string roar_graph_file="";       // 原始RoarGraph文件
    std::string output_graph_file;        // 更新后的图文件
    std::string data_type;
    std::string dist;
    uint32_t num_threads;
    float initial_ratio;
    
    std::string query_file;        // 查询文件
    std::string gt_file;           // 真值文件
    uint32_t batch_size = 10000;    // 每批插入数量
    std::string learn_base_nn_file;  // 学习到的基础近邻关系文件
    std::string sampled_query_data_file;  // 采样查询数据文件

    // 添加新的参数
    bool batch_mode = false;  // 默认使用一次性插入模式

    // 2. 参数解析
    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), 
                          "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), 
                          "distance function <l2/ip/cosine>");
        desc.add_options()("base_data_path", 
                          po::value<std::string>(&base_data_file)->required(),
                          "Base data file containing all vectors");
        desc.add_options()("roar_graph_path", 
                          po::value<std::string>(&roar_graph_file)->default_value(""),
                          "Original RoarGraph file");
        desc.add_options()("output_graph_path", 
                          po::value<std::string>(&output_graph_file)->required(),
                          "Path to save updated graph");
        desc.add_options()("num_threads,T", 
                          po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                          "Number of threads");
        // desc.add_options()("sampled_query_data_path", po::value<std::string>(&sampled_query_data_file)->required(),
        //                    "Sampled query file in bin format");                   
        desc.add_options()("batch_size", 
                          po::value<uint32_t>(&batch_size)->default_value(1000),
                          "Batch size for insertion");
        desc.add_options()("learn_base_nn_path", 
                          po::value<std::string>(&learn_base_nn_file)->required(),
                          "Path of learn-base NN file");
        desc.add_options()("sampled_query_data_path", 
                          po::value<std::string>(&sampled_query_data_file)->default_value(""),
                          "Sampled query file in bin format");
        desc.add_options()("batch_mode", 
                          po::value<bool>(&batch_mode)->default_value(false),
                          "Use batch insertion mode");
        desc.add_options()("initial_ratio", 
                          po::value<float>(&initial_ratio)->default_value(0.1),
                          "Use initial base data to construct index");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

 //  加载所有必要的数据
    uint32_t base_num, base_dim;
    uint32_t sq_num, sq_dim;
    float *base_data = nullptr;
    float *sq_data = nullptr;
    // 加载基础数据元信息
    efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    
    // 计算初始构建所需的数据量
    // float initial_ratio = batch_mode ? 0.2f : 0.6f;
    uint32_t initial_num = static_cast<uint32_t>(base_num * initial_ratio);

    // 加载全部基础数据
    efanna2e::load_data<float>(base_data_file.c_str(), base_num, base_dim, base_data);
    
    // 加载训练查询数据
    if (!sampled_query_data_file.empty()) {
        efanna2e::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);
    }
    std::cout<<"sq_num: "<<sq_num<<std::endl;
    // 4. 设置距离度量
    efanna2e::Metric dist_metric;
    if (dist == "l2") {
        dist_metric = efanna2e::L2;
    } else if (dist == "ip") {
        dist_metric = efanna2e::INNER_PRODUCT;
    } else if (dist == "cosine") {
        dist_metric = efanna2e::COSINE;
    } else {
        std::cerr << "Unknown distance type: " << dist << std::endl;
        return -1;
    }

    // 5. 计算初始构建和分批插入的数量

    size_t remaining = base_num - initial_num;
    size_t num_batches = (remaining + batch_size - 1) / batch_size;
    std::cout<<"base_num: "<<base_num<<std::endl;
    std::cout<<"sq_num: "<<sq_num<<std::endl;
    // 6. 创建索引对象
    efanna2e::IndexBipartite index(base_dim, base_num + sq_num, dist_metric, nullptr);
    float* aligned_data = efanna2e::data_align(base_data, base_num, base_dim);
    if (!aligned_data) {
        std::cerr << "Failed to align data" << std::endl;
        return -1;
    }


    // 7. 加载学习到的基础近邻关系
    std::cout << "Loading learn-base KNN data." << std::endl;
    index.LoadLearnBaseKNN(learn_base_nn_file.c_str());

    efanna2e::Parameters build_params;
    build_params.Set<uint32_t>("M_sq", 100);
    build_params.Set<uint32_t>("M_pjbp", 35);
    build_params.Set<uint32_t>("L_pjpq", 500);
    build_params.Set<uint32_t>("num_threads", num_threads);
    build_params.Set<uint32_t>("L_pq", 100);

    // roar_graph_file = "t2i_10M_Initroar0.4.index";
    // roar_graph_file = "../data/t2i-10M/t2i_10M_roar20.index";
    // 8. 检查是否存在初始图文件
    if (!roar_graph_file.empty() && std::ifstream(roar_graph_file)) {
        // 如果存在，直接加载初始图
        std::cout << "\nLoading existing RoarGraph from " << roar_graph_file << std::endl;
        index.LoadProjectionGraph(roar_graph_file.c_str());
        
    } else {
        // 如果不存在，构建新的初始图
        std::cout << "\nBuilding initial RoarGraph with " 
                  << (initial_ratio * 100) << "% vectors..." << std::endl;

        float *data_bp = nullptr;
        float *data_sq = nullptr;
        float *aligned_data_bp = nullptr;
        float *aligned_data_sq = nullptr;
        efanna2e::Parameters parameters;
        efanna2e::load_data<float>(base_data_file.c_str(), base_num, base_dim, data_bp);
        // efanna2e::load_data<float>(sampled_query_data_file.c_str(), sq_num, sq_dim, data_sq);
        aligned_data_bp = data_bp;

        index.BuildRoarGraph(sq_num, sq_data, initial_num, aligned_data_bp, build_params); //(1M, null, 9.8M, base, params)
        std::cout << "已经建立好初始图" << std::endl;
        
        index.SaveProjectionGraph(roar_graph_file.c_str());
        std::cout << "Save initial index to " << roar_graph_file << std::endl;
    }

    index.SetBaseData(base_data);
    // 9. 插入剩余数据
    if (batch_mode) {
        // 分批插入模式 (20% -> 80%)
        std::cout << "\nStarting batch insertion for remaining " 
                  << (remaining * 100.0 / base_num) << "% vectors..." << std::endl;
        
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t start_idx = initial_num + batch * batch_size;
            size_t curr_batch_size = std::min(static_cast<size_t>(batch_size), base_num - start_idx);
            
            std::cout << "\nProcessing batch " << batch + 1 << "/" << num_batches 
                      << " (vectors " << start_idx << " to " << start_idx + curr_batch_size - 1 << ")" << std::endl;
            
            // 准备当前批次的ID
            std::vector<size_t> curr_ids(curr_batch_size);
            for(size_t i = 0; i < curr_batch_size; i++) {
                curr_ids[i] = start_idx + i;  // 直接使用连续的ID，从initial_num + batch * batch_size开始
            }
            
            // 执行批量插入
            auto insert_start = std::chrono::high_resolution_clock::now();
            index.InitVisitedListPool(num_threads);

            index.LoadSearchNeededData(base_data_file.c_str(), sampled_query_data_file.c_str()); 

            index.InsertIntoRoarGraph(
                aligned_data + start_idx * base_dim, 
                curr_ids.data(), 
                curr_batch_size, 
                build_params
            );
            auto insert_end = std::chrono::high_resolution_clock::now();
            
            // 保存中间状态
            std::string checkpoint_file = "roargraph_checkpoint_" + std::to_string(batch) + ".bin";
            index.SaveProjectionGraph(checkpoint_file.c_str());
            
            // 输出统计信息
            auto insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                insert_end - insert_start).count();
            std::cout<< std::endl;
            std::cout << "Batch " << batch + 1 << " insertion time: " << insert_time << "ms" << std::endl;
        }
    } else {
        // 一次性插入模式 (80% -> 20%)

        std::cout << "\nInserting remaining " 
                  << (remaining * 100.0 / base_num) << "% vectors..." << std::endl;
        
        // 准备剩余数据的ID
        std::vector<size_t> remaining_ids(remaining);
        for(size_t i = 0; i < remaining; i++) {
            remaining_ids[i] = initial_num + i;
        }
        
    
        std::cout << "Inserting remaining vectors..." << std::endl;
        
        auto insert_start = std::chrono::high_resolution_clock::now();
        index.InitVisitedListPool(num_threads);
        // uint32_t k=1;
        // uint32_t *res = new uint32_t[base_num * k];
        // memset(res, 0, sizeof(uint32_t) * base_num * k);
        // std::vector<std::vector<float>> res_dists(base_num, std::vector<float>(k, 0.0));
        index.LoadSearchNeededData(base_data_file.c_str(), sampled_query_data_file.c_str()); 
        
        // for (size_t i = 0; i < 3; ++i) {
        //     size_t a=initial_num+i;
        //     index.SearchRoarGraph(aligned_data + initial_num * base_dim + i * base_dim, k, a, build_params, res + i * k, res_dists[i]);
        //     std::cout << "搜索成功 TOP1 ID: " << res[i*k] << std::endl;
        // }
        index.InsertIntoRoarGraph(
            aligned_data + initial_num * base_dim,
            remaining_ids.data(),
            remaining, //0.8
            build_params
        );
 
         
        auto insert_end = std::chrono::high_resolution_clock::now();
        
        auto insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            insert_end - insert_start).count();
        std::cout << "The final insertion time: " << insert_time << "ms" << std::endl;
   
    
        //  保存最终图
        index.SaveProjectionGraph(output_graph_file.c_str());
        std::cout << "Save index to " << output_graph_file << std::endl;
        return 0;
 }
}