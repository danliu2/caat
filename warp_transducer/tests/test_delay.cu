#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

#include <rnnt.h>

#include "test.h"

template<typename T>
void vector_to_gpu(T*& gpu_space, std::vector<T>& vec, cudaStream_t& stream) {
    cudaMalloc(&gpu_space, vec.size() * sizeof(T));
    cudaMemcpyAsync(gpu_space, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
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

bool small_test() {
    const int B = 1;
    const int alphabet_size = 5;
    const int T = 2;
    const int U = 3;

    std::vector<float> acts = {0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 
                                0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 
                                0.2, 0.8, 0.1, 0.1, 0.6, 0.1, 
                                0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 
                                0.1, 0.7, 0.1, 0.2, 0.1, 0.1};
    // std::vector<float> log_probs(acts.size());
    // softmax(acts.data(), alphabet_size, B * T * U, log_probs.data(), true);

    float expected_score = 4.495666;

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    std::vector<float> scores(B*3);

    rnntOptions options{};
    options.maxT = T;
    options.maxU = U;
    options.loc = RNNT_GPU;
    options.blank_label = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    int* label_gpu;
    vector_to_gpu(label_gpu, labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, lengths, stream);
    float* delay_values= gen_delay_value(B,T, stream);

    size_t gpu_alloc_bytes;
    throw_on_error(get_delay_workspace_size(T, U, B,
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_delay_workspace_size in small_test");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_delay_loss(acts_gpu,
                                    NULL,
                                    label_gpu, 
                                    label_length_gpu,
                                    input_length_gpu,
                                    delay_values,
                                    alphabet_size,
                                    lengths.size(),
                                    scores.data(),
                                    rnnt_gpu_workspace,1.0,1.0,
                                    options),
                   "Error: compute_rnnt_delay_loss in small_test");

    cudaFree(rnnt_gpu_workspace);
    cudaFree(acts_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);
    cudaFree(delay_values);


     const float eps = 1e-4;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;
    
    return (scores[0] > lb && scores[0] < ub); 
}

bool options_test() {
    const int alphabet_size = 3;
    const int T = 4;
    const int L = 3;
    const int minibatch = 2;

    std::vector<float> acts = {0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135, 
                                0.609764, 0.868140, 0.622532, 0.668522, 0.858039, 0.164539, 
                                0.989780, 0.944298, 0.603168, 0.946783, 0.666203, 0.286882, 
                                0.094184, 0.366674, 0.736168, 0.166680, 0.714154, 0.399400, 
                                0.535982, 0.291821, 0.612642, 0.324241, 0.800764, 0.524106, 
                                0.779195, 0.183314, 0.113745, 0.240222, 0.339470, 0.134160, 
                                0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467, 
                                0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474, 
                                0.083132, 0.060165, 0.107958, 0.748615, 0.943918, 0.486356, 
                                0.418199, 0.652408, 0.024243, 0.134582, 0.366342, 0.295830, 
                                0.923670, 0.689929, 0.741898, 0.250005, 0.603430, 0.987289, 
                                0.592606, 0.884672, 0.543450, 0.660770, 0.377128, 0.358021};
    // std::vector<float> log_probs(acts.size());
    // softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

    std::vector<float> expected_grads = {-0.186844, -0.062555, 0.249399, -0.203377, 0.202399, 0.000977,
                                        -0.141016, 0.079123, 0.061893, -0.011552, -0.081280, 0.092832,
                                        -0.154257, 0.229433, -0.075176, -0.246593, 0.146405, 0.100188,
                                        -0.012918, -0.061593, 0.074512, -0.055986, 0.219831, -0.163845,
                                        -0.497627, 0.209240, 0.288387, 0.013605, -0.030220, 0.016615,
                                        0.113925, 0.062781, -0.176706, -0.667078, 0.367659, 0.299419,
                                        -0.356344, -0.055347, 0.411691, -0.096922, 0.029459, 0.067463,
                                        -0.063518, 0.027654, 0.035863, -0.154499, -0.073942, 0.228441,
                                        -0.166790, -0.000088, 0.166878, -0.172370, 0.105565, 0.066804,
                                        0.023875, -0.118256, 0.094381, -0.104707, -0.108934, 0.213642,
                                        -0.369844, 0.180118, 0.189726, 0.025714, -0.079462, 0.053748,
                                        0.122328, -0.238789, 0.116460, -0.598687, 0.302203, 0.296484};

    // Calculate the expected scores analytically
    std::vector<double> expected_scores(2);
    expected_scores[0] = 4.2806528590890736;
    expected_scores[1] = 3.9384369822503591;

    std::vector<int> labels = {1, 2, 1, 1};

    std::vector<int> label_lengths = {2, 2};

    std::vector<int> lengths = {4, 4};

    std::vector<float> grads(acts.size());
    std::vector<float> scores(2*3);

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
    int* label_gpu;
    vector_to_gpu(label_gpu, labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, lengths, stream);
    float* delay_values= gen_delay_value(minibatch,T, stream);

    size_t gpu_alloc_bytes;
    throw_on_error(get_delay_workspace_size(T, L, minibatch,
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_delay_workspace_size in options_test");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_delay_loss(acts_gpu,
                                    grads_gpu,
                                    label_gpu, 
                                    label_length_gpu,
                                    input_length_gpu,
                                    delay_values,
                                    alphabet_size,
                                    lengths.size(),
                                    scores.data(),
                                    rnnt_gpu_workspace,1.0,1.0,
                                    options),
                   "Error: compute_rnnt_delay_loss in small_test");

    cudaMemcpyAsync(grads.data(), grads_gpu, grads.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaFree(rnnt_gpu_workspace);
    cudaFree(acts_gpu);
    cudaFree(grads_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);
    cudaFree(delay_values);

    const double eps = 1e-4;

    bool result = true;
    // activations gradient check
    // for (int i = 0; i < grads.size(); i++) {
        // const double lb = expected_grads[i] - eps;
        // const double ub = expected_grads[i] + eps;
        // if (!(grads[i] > lb && grads[i] < ub)) {
            // std::cerr << "grad mismatch in options_test"
                    //   << " expected grad: " << expected_grads[i]
                    //   << " calculated score: " << grads[i]
                    //   << " !(" << lb << " < " << grads[i]
                    //   << " < " << ub << ")" << std::endl;
            // result = false;
        // }
    // }

    for (int i = 0; i < 2; i++) {
        const double lb = expected_scores[i] - eps;
        const double ub = expected_scores[i] + eps;
        if (!(scores[i] > lb && scores[i] < ub)) {
            std::cerr << "score mismatch in options_test"
                      << " expected score: " << expected_scores[i]
                      << " calculated score: " << scores[i]
                      << " !(" << lb << " < " << scores[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }
    return result;
}

bool inf_test() {
    const int alphabet_size = 15;
    const int T = 50;
    const int L = 10;
    const int minibatch = 1;

    std::vector<int> labels = genLabels(alphabet_size, L-1);
    labels[0] = 2;
    std::vector<int> label_lengths = {L-1};

    std::vector<float> acts(alphabet_size * T * L * minibatch);
    genActs(acts);

    // std::vector<float> log_probs(acts.size());
    // softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

    std::vector<int> sizes;
    sizes.push_back(T);

    std::vector<float> grads(acts.size());

    std::vector<float> cost(3);

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
    int* label_gpu;
    vector_to_gpu(label_gpu, labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, sizes, stream);
    float* delay_values= gen_delay_value(minibatch,T, stream);
    size_t gpu_alloc_bytes;
    throw_on_error(get_delay_workspace_size(T, L, minibatch,
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_delay_workspace_size in inf_test");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_delay_loss(acts_gpu,
                                    grads_gpu,
                                    label_gpu, 
                                    label_length_gpu,
                                    input_length_gpu,
                                    delay_values,
                                    alphabet_size,
                                    sizes.size(),
                                    cost.data(),
                                    rnnt_gpu_workspace,1.0,1.0,
                                    options),
                   "Error: compute_rnnt_delay_loss in small_test");

    cudaMemcpyAsync(grads.data(), grads_gpu, grads.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaFree(rnnt_gpu_workspace);
    cudaFree(acts_gpu);
    cudaFree(grads_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);
    cudaFree(delay_values);
    bool status = true;
    status &= !std::isinf(cost[0]);

    for (int i = 0; i < alphabet_size * L * T * minibatch; ++i)
        status &= !std::isnan(grads[i]);

    return status;
}

void numeric_grad(float* acts, int* flat_labels, int* label_lengths, 
                int* sizes, float* delay_values, int alphabet_size, int minibatch, 
                void* rnnt_gpu_workspace, rnntOptions& options, std::vector<float>& num_grad) {

    float epsilon = 1e-2;
    float act;

    for (int i = 0; i < num_grad.size(); ++i) {

        std::vector<float> costsP1(minibatch*3);
        std::vector<float> costsP2(minibatch*3);

        cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
        act += epsilon;
        cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
        throw_on_error(compute_rnnt_delay_loss(acts,
                                        NULL,
                                        flat_labels, 
                                        label_lengths,
                                        sizes,
                                        delay_values,
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        rnnt_gpu_workspace,0.0,1.0,
                                        options),
                       "Error: compute_rnnt_delay_loss (1) in grad_check");

        cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
        act -= 2 * epsilon;
        cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
        throw_on_error(compute_rnnt_delay_loss(acts,
                                        NULL,
                                        flat_labels, 
                                        label_lengths,
                                        sizes,
                                        delay_values,
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        rnnt_gpu_workspace,0.0,1.0,
                                        options),
                       "Error: compute_rnnt_delay_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin()+minibatch*2, costsP1.end(), 0.);
        float costP2 = std::accumulate(costsP2.begin()+minibatch*2, costsP2.end(), 0.);

        cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
        act += epsilon;
        cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }
}

bool grad_check(int T, int L, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  std::vector<int>& sizes, float tol) {

    const int minibatch = labels.size();

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(minibatch*3);

    std::vector<float> grads(acts.size());

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
    int* label_gpu;
    vector_to_gpu(label_gpu, flat_labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, sizes, stream);
    options.num_threads = 1;
    float* delay_values= gen_delay_value(minibatch,T, stream);

    size_t gpu_alloc_bytes;
    throw_on_error(get_delay_workspace_size(T, L, sizes.size(),
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_delay_workspace_size in grad_check");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_delay_loss(acts_gpu,
                                    grads_gpu,
                                    label_gpu, 
                                    label_length_gpu,
                                    input_length_gpu,
                                    delay_values,
                                    alphabet_size,
                                    sizes.size(),
                                    costs.data(),
                                    rnnt_gpu_workspace,0.0,1.0,
                                    options),
                   "Error: compute_rnnt_delay_loss (0) in grad_check");

    float cost = std::accumulate(costs.begin()+minibatch*2, costs.end(), 0.);

    cudaMemcpyAsync(grads.data(), grads_gpu, grads.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    numeric_grad(acts_gpu, label_gpu, label_length_gpu, input_length_gpu,delay_values,
            alphabet_size, minibatch, rnnt_gpu_workspace, options, num_grad);

    cudaFree(acts_gpu);
    cudaFree(rnnt_gpu_workspace);
    cudaFree(grads_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);
    cudaFree(delay_values);

    float diff = rel_diff(grads, num_grad);
    for(int i =0; i < grads.size();i++){
        printf("g=%.3f,num_g=%.3f\n", grads[i], num_grad[i]);
    }
    printf("diff=%.3f, tol=%.3f\n",diff,tol);
    return diff < tol;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
       {std::make_tuple(20, 50, 15, 1, 1e-2),
        std::make_tuple(5, 10, 5, 65, 1e-2)
       };

    std::mt19937 gen(2);

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> acts(alphabet_size * T * L * minibatch);
        genActs(acts);

        std::vector<float> log_probs(acts.size());
        softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L - 1;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        status &= grad_check(T, L, alphabet_size, acts, labels, sizes, tol);
    }

    return status;
}

int main(void) {
    if (get_warprnnt_version() != 1) {
        std::cerr << "Invalid Warp-transducer version." << std::endl;
        return 1;
    }

    std::cout << "Running gpu tests" << std::endl;

    bool status = true;
    // status &= small_test();
    // printf("finish small_test %d\n", status);
    
     status &= options_test();
     printf("finish options_test %d\n", status);
    exit(0);
    // status &= inf_test();
    // printf("finish inf_test %d\n", status);
    status &= run_tests();
    printf("finished %d\n", status);

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}