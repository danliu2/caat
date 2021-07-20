#pragma once

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>

#if !defined(RNNT_DISABLE_OMP) && !defined(APPLE)
#include <omp.h>
#endif

#include "reduce.h"
#include "gpu_rnnt_kernel.h"

template<typename ProbT>
class DelayTransducer {
public:
    // Noncopyable
    DelayTransducer(int minibatch, int maxT, int maxU, int alphabet_size, void* workspace, 
            int blank, int num_threads, float delay_scale,float smooth, CUstream stream) :
        minibatch_(minibatch), maxT_(maxT), maxU_(maxU), alphabet_size_(alphabet_size), 
        gpu_workspace(workspace), blank_(blank), num_threads_(num_threads), stream_(stream) ,
        delay_scale_(delay_scale), smooth_(smooth){
#if defined(RNNT_DISABLE_OMP) || defined(APPLE)
#else
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads_ = omp_get_max_threads();
        }
#endif
    };

    DelayTransducer(const DelayTransducer&) = delete;
    DelayTransducer& operator=(const DelayTransducer&) = delete;

    void log_softmax(const ProbT* const acts, ProbT* denom);

    rnntStatus_t compute_cost_and_score(const ProbT* const acts,
                                        const ProbT* delay_values,
                                        ProbT* grad,
                                        ProbT* costs,
                                        const int* const pad_labels,
                                        const int* const label_lengths,
                                        const int* const input_lengths);

    rnntStatus_t cost_and_grad(const ProbT* const acts,
                              const ProbT* delay_values,
                              ProbT* grad,
                              ProbT* costs,
                              const int* const pad_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

    rnntStatus_t score_forward(const ProbT* const acts,
                              const ProbT* delay_values,
                              ProbT* costs,
                              const int* const pad_labels,
                              const int* const label_lengths,
                              const int* const input_lengths);

private:
    int minibatch_;
    int maxT_;
    int maxU_;
    int alphabet_size_; // Number of characters plus blank
    void* gpu_workspace;
    int blank_;
    int num_threads_;
    CUstream stream_;
    float delay_scale_; // scale for delay loss
    float smooth_;
    
};



template<typename ProbT>
void
DelayTransducer<ProbT>::log_softmax(const ProbT* const acts, ProbT* denom) {

    // trans_acts + pred_acts -> log_softmax denominator
    reduce_max(acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 0, stream_);
    reduce_exp(acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 1, stream_);
}

template<typename ProbT>
rnntStatus_t
DelayTransducer<ProbT>::compute_cost_and_score(const ProbT* const acts,
                                    const ProbT* delay_values,
                                    ProbT* grads,
                                    ProbT* costs,
                                    const int* const labels,
                                    const int* const label_lengths,
                                    const int* const input_lengths) {
    /*
        costs: 3xB, for each sample , full_loss, loss_prob and loss_delay
    */
    
    bool training = (grads != nullptr);
    size_t bytes_used = 0;
    // denom
    ProbT* denom = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    // alphas & betas
    ProbT* alphas = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    ProbT* betas = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    // logllh
    ProbT* llForward = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * minibatch_;
    ProbT* llBackward = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * minibatch_;
    ProbT* alpha_delay= reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) +bytes_used);
    bytes_used +=sizeof(ProbT)*maxT_*maxU_ *minibatch_;
    ProbT* beta_delay = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) +bytes_used);
    bytes_used+=sizeof(ProbT)*maxT_*maxU_*minibatch_;
    ProbT* delay_expect_fwd = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) +bytes_used);
    bytes_used+=sizeof(ProbT)*minibatch_;
    ProbT* delay_expect_bwd = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) +bytes_used);
    bytes_used+=sizeof(ProbT)*minibatch_;

    if (training) {
        // zero grads
        cudaMemsetAsync(grads, 0, sizeof(ProbT) * minibatch_ * maxT_ * maxU_ * alphabet_size_, stream_);
    }
    // denom
#if defined(DEBUG_TIME)
     auto start = std::chrono::high_resolution_clock::now();
#endif
    log_softmax(acts, denom);
#if defined(DEBUG_TIME)
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "DEBUG: log_softmax " << elapsed.count() * 1000 << " ms\n";
    // alphas
    start = std::chrono::high_resolution_clock::now();
#endif
#if defined(USE_NAIVE_KERNEL)
    compute_alphas_kernel_naive<ProbT><<<1, minibatch_, 0, stream_>>>(acts, denom, alphas, llForward, 
        input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
#else
    compute_alphas_kernel<ProbT><<<minibatch_, maxU_, 0, stream_>>>(acts, denom, alphas, llForward, 
        input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    compute_alpha_delay_kernel<ProbT><<<minibatch_, maxU_,0, stream_>>>(acts,denom, delay_values,
        alpha_delay,delay_expect_fwd,alphas,
        input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
#endif
#if defined(DEBUG_TIME)
    cudaStreamSynchronize(stream_);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "DEBUG: compute_alphas_kernel " << elapsed.count() * 1000 << " ms\n";
#endif
#if defined(DEBUG_KERNEL)
    ProbT* cpu_alphas = new ProbT[minibatch_ * maxT_ * maxU_];
    ProbT* cpu_alpha_delay = new ProbT[minibatch_ * maxT_ * maxU_];
    int* cpu_xlen = new int[minibatch_];
    int* cpu_ylen = new int[minibatch_];
    cudaMemcpy(cpu_alphas, alphas, sizeof(ProbT) * minibatch_ * maxT_ * maxU_, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_xlen, input_lengths, sizeof(int) * minibatch_, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_ylen, label_lengths, sizeof(int) * minibatch_, cudaMemcpyDeviceToHost);
    printf("gpu alphas\n");
    for (int b = 0; b < minibatch_; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_alphas[(b*maxT_+t)*maxU_+u]);
            }
            printf("\n");
        }
        printf("\n");
    }
    cudaMemcpy(cpu_alpha_delay, alpha_delay, sizeof(ProbT) * minibatch_ * maxT_ * maxU_, cudaMemcpyDeviceToHost);
    printf("alpha delay\n");
    for (int b = 0; b < minibatch_; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_alpha_delay[(b*maxT_+t)*maxU_+u]);
            }
            printf("\n");
        }
        printf("\n");
    }
#endif
    if (training) {
        // betas
#if defined(DEBUG_TIME)
        start = std::chrono::high_resolution_clock::now();
#endif
#if defined(USE_NAIVE_KERNEL)
        compute_betas_kernel_naive<ProbT><<<1, minibatch_, 0, stream_>>>(acts, denom, betas, llBackward,
            input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
#else
        compute_betas_kernel<ProbT><<<minibatch_, maxU_, 0, stream_>>>(acts, denom, betas, llBackward,
            input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
        compute_beta_delay_kernel<ProbT><<<minibatch_, maxU_, 0, stream_>>>(acts, denom,delay_values, 
            beta_delay, delay_expect_bwd, betas,
            input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);

#endif
#if defined(DEBUG_TIME)
        cudaStreamSynchronize(stream_);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "DEBUG: compute_betas_kernel " << elapsed.count() * 1000 << " ms\n";
#endif
#if defined(DEBUG_KERNEL)
    ProbT* ll = new ProbT[minibatch_];
    cudaMemcpy(ll, llForward, sizeof(ProbT)*minibatch_, cudaMemcpyDeviceToHost);
    printf("llforward:");
    for(int b=0; b<minibatch_;b++){
        printf("%.3f ", ll[b]);
    }
    printf("\n");
    // cudaMemcpy(ll, llBackward, sizeof(ProbT)*minibatch_, cudaMemcpyDeviceToHost);
    // printf("llbackward:");
    // for(int b=0; b<minibatch_;b++){
    //     printf("%.3f ", ll[b]);
    // }
    // cudaMemcpy(ll, delay_expect_fwd, sizeof(ProbT)*minibatch_, cudaMemcpyDeviceToHost);
    // printf("delay_expect_fwd:");
    // for(int b=0; b<minibatch_;b++){
    //     printf("%.3f ", ll[b]);
    // }
    // printf("\n");
    // cudaMemcpy(ll, delay_expect_bwd, sizeof(ProbT)*minibatch_, cudaMemcpyDeviceToHost);
    // printf("delay_expect_bwd:");
    // for(int b=0; b<minibatch_;b++){
    //     printf("%.3f ", ll[b]);
    // }
    // printf("\n");
    // delete[] ll;
    ProbT* cpu_betas = new ProbT[minibatch_ * maxT_ * maxU_];
    ProbT* cpu_beta_delay = new ProbT[minibatch_ * maxT_ * maxU_];
    cudaMemcpy(cpu_betas, betas, sizeof(ProbT) * minibatch_ * maxT_ * maxU_, cudaMemcpyDeviceToHost);
    printf("gpu betas\n");
    for (int b = 0; b < minibatch_; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_betas[(b*maxT_+t)*maxU_+u]);
            }
            printf("\n");
        }
        printf("\n");
    }
    cudaMemcpy(cpu_beta_delay, beta_delay,sizeof(ProbT) * minibatch_ * maxT_ * maxU_, cudaMemcpyDeviceToHost);
    printf("beta delay\n");
    for (int b = 0; b < minibatch_; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_beta_delay[(b*maxT_+t)*maxU_+u]);
            }
            printf("\n");
        }
        printf("\n");
    }
    for(int b = 0; b<minibatch_;b++){
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        int N=T+U-1;
        for(int n=1; n<N;n++){
            ProbT loss=0;
            ProbT sum_prob=0;
            for(int u=0;(u<=n)&& (u<U);u++){
                int t=n-u;
                if (t<0 || t>=T)continue;
                ProbT prob = cpu_alphas[(b*maxT_+t)*maxU_+u]+cpu_betas[(b*maxT_+t)*maxU_+u]-ll[b];
                prob=exp(prob);
                sum_prob+=prob;
                loss+=prob*(cpu_beta_delay[(b*maxT_+t)*maxU_+u] +cpu_alpha_delay[(b*maxT_+t)*maxU_+u]);
            }
            printf("b=%d,n=%d, prob_sum=%.3f, loss_sum=%.3f\n", b, n, sum_prob,loss);
        }
    }
    /* printf("gamma\n");
    for (int b = 0; b < minibatch_; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_betas[(b*maxT_+t)*maxU_+u] + cpu_alphas[(b*maxT_+t)*maxU_+u]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("expect delay\n");
    for (int b = 0; b < minibatch_; b++) {
        int T = cpu_xlen[b];
        int U = cpu_ylen[b] + 1;
        printf("B %d, T %d, U %d\n", b, T, U);
        for (int t = 0; t < T; t++) {
            for (int u = 0; u < U; u++) {
                printf("%.2f ", cpu_beta_delay[(b*maxT_+t)*maxU_+u] + cpu_alpha_delay[(b*maxT_+t)*maxU_+u]);
            }
            printf("\n");
        }
        printf("\n");
    } */
    delete[] ll;
    delete[] cpu_beta_delay;
    delete[] cpu_alpha_delay;
    delete[] cpu_betas;
    delete[] cpu_alphas;
    delete[] cpu_xlen;
    delete[] cpu_ylen;
#endif

        // gradient
#if defined(DEBUG_TIME)
        start = std::chrono::high_resolution_clock::now();
#endif
        // TODO optimize gradient kernel
        // compute_grad_withdelay_kernel<128, ProbT><<<minibatch_ * maxT_ * maxU_, 128, 0, stream_>>>(grads, 
        //     acts, denom, alphas, betas, llForward,delay_values, alpha_delay,beta_delay, delay_expect_fwd,
        //     input_lengths, label_lengths, labels, 
        //     minibatch_, maxT_, maxU_, alphabet_size_, blank_, delay_scale_);
        compute_grad_withdelay_smooth_kernel<128, ProbT><<<minibatch_ * maxT_ * maxU_, 128, 0, stream_>>>(grads, 
            acts, denom, alphas, betas, llForward,delay_values, alpha_delay,beta_delay, delay_expect_fwd,
            input_lengths, label_lengths, labels, 
            minibatch_, maxT_, maxU_, alphabet_size_, blank_, delay_scale_, smooth_);
#if defined(DEBUG_KERNEL)
        ProbT* cpu_grad = new ProbT[minibatch_ * maxT_ * maxU_*alphabet_size_];
        cudaMemcpy(cpu_grad, grads, sizeof(ProbT) * minibatch_ * maxT_ * maxU_*alphabet_size_, cudaMemcpyDeviceToHost);
        /* int b=0, t=14,u=2;
        printf("b=%d,t=%d,u=%d\n", b,t,u);
        for(int i= 0; i < 10;i++){
            printf("%.3f ", cpu_grad[((b*maxT_+t)*maxU_+u)*alphabet_size_+i]);
        }
        printf("\n"); */
        for(int b= 0; b< minibatch_ && b<2;b++){
            for(int t=0; t< maxT_ && t< 5; t++){
                for(int u=0; u<maxU_ && u<5;u ++){
                    printf("b=%d,t=%d,u=%d\n", b,t,u);
                    for(int i= 0; i < 10;i++){
                        printf("%.3f ", cpu_grad[((b*maxT_+t)*maxU_+u)*alphabet_size_+i]);
                    }
                    printf("\n");
                }
            }
        } 
        delete[] cpu_grad;
#endif
#if defined(DEBUG_TIME)
        cudaStreamSynchronize(stream_);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "DEBUG: compute_grad_kernel " << elapsed.count() * 1000 << " ms\n";
#endif
    }
    // cost
    cudaMemcpyAsync(costs, llForward, sizeof(ProbT) * minibatch_, cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(costs+minibatch_, delay_expect_fwd, sizeof(ProbT) * minibatch_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    for (int mb = 0; mb < minibatch_; ++mb) {
        costs[mb] = -costs[mb];
        costs[minibatch_*2+mb] = costs[mb] + delay_scale_*costs[minibatch_+mb];
    }
    #if defined(DEBUG_KERNEL)
    for (int mb = 0; mb < minibatch_; ++mb) {
        printf("b=%d,cost=%f\n", mb, costs[minibatch_+mb]);
    }
    #endif
    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
rnntStatus_t
DelayTransducer<ProbT>::cost_and_grad(const ProbT* const acts,const ProbT* delay_values,
                       ProbT* grads,
                       ProbT* costs,
                       const int* const pad_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {

    if (acts == nullptr ||
        grads == nullptr || 
        costs == nullptr ||
        pad_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr)
        return RNNT_STATUS_INVALID_VALUE;

    return compute_cost_and_score(acts,delay_values, grads, costs, pad_labels, label_lengths, input_lengths);
}

template<typename ProbT>
rnntStatus_t
DelayTransducer<ProbT>::score_forward(const ProbT* const acts,const ProbT* delay_values,
                       ProbT* costs,
                       const int* const pad_labels,
                       const int* const label_lengths,
                       const int* const input_lengths) {
    
    if (acts == nullptr ||
        costs == nullptr ||
        pad_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr)
        return RNNT_STATUS_INVALID_VALUE;

    return compute_cost_and_score(acts,delay_values, nullptr, costs, pad_labels, label_lengths, input_lengths);
}
