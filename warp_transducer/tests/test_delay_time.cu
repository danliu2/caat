#include <cmath>
#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>

#include <chrono>

#include <iostream>

#include <rnnt.h>

#include "test.h"

template<typename T>
void vector_to_gpu(T*& gpu_space, std::vector<T>& vec, cudaStream_t& stream) {
    cudaMalloc(&gpu_space, vec.size() * sizeof(T));
    cudaMemcpyAsync(gpu_space, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void vector_to_gpu(T*& gpu_space, const T* cpu_space, int len, cudaStream_t& stream) {
    cudaMalloc(&gpu_space, len * sizeof(T));
    cudaMemcpyAsync(gpu_space, cpu_space, len * sizeof(T), cudaMemcpyHostToDevice, stream);
}
float* gen_delay_value(int b, int T, cudaStream_t& stream){
    float* buffer= new float[b*T];
    for(int i = 0; i < b; i++){
        for(int j=0;j<T;j ++){
            buffer[i*T +j] = float(j)/T;
        }
    }
    float* dev_buff= nullptr;
    cudaMalloc(&dev_buff, sizeof(float)*b*T);
    cudaMemcpyAsync(dev_buff, buffer, sizeof(float)*b*T, cudaMemcpyHostToDevice, stream);
    return dev_buff;
}

bool run_test(int B, int T, int L, int A, int num_threads) {
    std::mt19937 gen(2);

    auto start = std::chrono::high_resolution_clock::now();
    int len = B * T * (L + 1) * A;
    float * acts = genActs(len);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "genActs elapsed time: " << elapsed.count() * 1000 << " ms\n";

    std::vector<std::vector<int>> labels;
    std::vector<int> sizes;

    for (int mb = 0; mb < B; ++mb) {
        labels.push_back(genLabels(A, L));
        sizes.push_back(T);
    }

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(B*3);

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L + 1;
    options.blank_label = 0;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = num_threads;

    float* acts_gpu;
    vector_to_gpu<float>(acts_gpu, acts, len, stream);
    // cudaMalloc(&acts_gpu, len * sizeof(float));
    // cudaMemcpyAsync(acts_gpu, acts, len * sizeof(float), cudaMemcpyHostToDevice, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, len * sizeof(float));
    int* label_gpu;
    vector_to_gpu(label_gpu, flat_labels, stream);
    // cudaMalloc(&label_gpu, flat_labels.size() * sizeof(int))
    // cudaMemcpyAsync(label_gpu, flat_labels.data(), flat_labels.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    // cudaMalloc(&label_length_gpu, label_lengths.size() * sizeof(int));
    // cudaMemcpyAsync(label_length_gpu, label_lengths.data(), label_lengths.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, sizes, stream);
    // cudaMalloc(&input_length_gpu, sizes.size() * sizeof(int));
    // cudaMemcpyAsync(input_length_gpu, sizes.data(), sizes.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    float* delay_values=gen_delay_value(B,T, stream);

    size_t gpu_alloc_bytes;
    throw_on_error(get_delay_workspace_size(T, L+1, B,
                                     true,
                                     &gpu_alloc_bytes),
                    "Error: get_workspace_size in run_test");

    std::vector<float> time;
    // for (int i = 0; i < 10; ++i) {
    for (int i = 0; i < 1; ++i) {
        void* rnnt_gpu_workspace;
        cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

        start = std::chrono::high_resolution_clock::now();
        throw_on_error(compute_rnnt_delay_loss(acts_gpu, grads_gpu,
                                        label_gpu, label_length_gpu,
                                        input_length_gpu,delay_values,
                                        A, B,
                                        costs.data(),
                                        rnnt_gpu_workspace,1.0,1.0,
                                        options),
                        "Error: compute_rnnt_loss (0) in run_test");
        end = std::chrono::high_resolution_clock::now();
        int U=L+1;
        int grad_size= B*T*U;
        float* grads_cpu= new float[grad_size];
        cudaMemcpy(grads_cpu, grads_gpu, sizeof(float) * grad_size, cudaMemcpyDeviceToHost);
        float max_=-10, min_=10;
        int badnum=0;
        for(int i =0; i < grad_size; i++){
            if(grads_cpu[i] > max_){
                max_=grads_cpu[i];
            } 
            if(grads_cpu[i] < min_){
                min_=grads_cpu[i];
            } 
            if(grads_cpu[i] >50 || grads_cpu[i]< -50) {
                int u = i % U;
                int bt = (i - u) / U;
                int t = bt % T;
                int mb = (bt - t) / T;
                printf("b=%d, t=%d,u=%d, bad grad =%f\n", mb,t, u, grads_cpu[i]);
                badnum++;
            }
        }
        printf("max=%.4f,min=%.4f, badnum=%d\n", max_, min_, badnum);
        cudaFree(rnnt_gpu_workspace);
        elapsed = end - start;
        time.push_back(elapsed.count() * 1000);
        std::cout << "compute_rnnt_delay_loss elapsed time: " << elapsed.count() * 1000 << " ms\n";
    }

    cudaFree(grads_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);

    float sum = 0;
    //for (int i = 0; i < 10; ++i) {
    for (int i = 0; i < 1; ++i) {
        sum += time[i];
    }
    sum /= time.size();

    float std = 0;
    for (int i = 0; i < 10; ++i) {
        std += (time[i] - sum) * (time[i] - sum);
    }
    std /= time.size();

    std::cout << "average 10 time cost: " << sum << " ms variance: " << std << std::endl;

    float cost = std::accumulate(costs.begin(), costs.end(), 0.);

    free(acts);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Arguments: <Batch size> <Time step> <Label length> <Alphabet size>\n";
        return 1;
    }

    int B = atoi(argv[1]);
    int T = atoi(argv[2]);
    int L = atoi(argv[3]);
    int A = atoi(argv[4]);
    std::cout << "Arguments: " \
                << "\nBatch size: " << B \
                << "\nTime step: " << T \
                << "\nLabel length: " << L \
                << "\nAlphabet size: " << A \
                << std::endl;
    
    int num_threads = 1;
    if (argc >= 6) {
        num_threads = atoi(argv[5]);
        std::cout << "Num threads: " << num_threads << std::endl;
    }

    run_test(B, T, L, A, num_threads);
}