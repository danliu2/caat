#include <cstddef>
#include <iostream>
#include <algorithm>

#include <rnnt.h>

// #include "detail/cpu_rnnt.h"
// #ifdef __CUDACC__
//     #include "detail/gpu_rnnt.h"
// #endif
#include "detail/delay_transducer.h"

extern "C" {

rnntStatus_t compute_rnnt_delay_loss(const float* const activations, //BTUV
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             const float* delay_values,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             float delay_scale,
                             float smooth,
                             rnntOptions options) {
    /*
        delay_values: BxT, delay cost for read each source frame
    */

    if (activations == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0 ||
        options.maxT <= 0 ||
        options.maxU <= 0 ||
        delay_scale <-1e8)
        return RNNT_STATUS_INVALID_VALUE;

        #ifdef __CUDACC__
                DelayTransducer<float> rnnt(minibatch, options.maxT, options.maxU, alphabet_size, workspace,
                                        options.blank_label, options.num_threads,delay_scale,smooth, options.stream);

                if (gradients != NULL)
                    return rnnt.cost_and_grad(activations,delay_values, gradients,
                                                costs,
                                                flat_labels, label_lengths,
                                                input_lengths);
                else
                    return rnnt.score_forward(activations, delay_values,costs, flat_labels,
                                                label_lengths, input_lengths);
        #else
                std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
                return RNNT_STATUS_EXECUTION_FAILED;
        #endif
}


rnntStatus_t get_delay_workspace_size(int maxT, int maxU,
                               int minibatch,
                               bool gpu,
                               size_t* size_bytes,
                               size_t dtype_size)
{
    if (minibatch <= 0 ||
        maxT <= 0 ||
        maxU <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    *size_bytes = 0;

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += dtype_size * maxT * maxU * 2;
    // softmax denominator
    per_minibatch_bytes += dtype_size * maxT * maxU;
    // forward-backward loglikelihood
    per_minibatch_bytes += dtype_size * 2;
    // alpha_delay & beta_delay
    per_minibatch_bytes += dtype_size*maxT* maxU *2;
    // fwd-bwd delay expectation
    per_minibatch_bytes += dtype_size*2;

    *size_bytes = per_minibatch_bytes * minibatch;

    return RNNT_STATUS_SUCCESS;
}

}