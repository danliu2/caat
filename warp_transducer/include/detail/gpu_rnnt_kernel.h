#pragma once

#include "rnnt_helper.h"
#include <stdio.h>
template<typename T>
inline __device__ T logp(const T* const denom, const T* const acts, const int maxT, const int maxU, const int alphabet_size, int mb, int t, int u, int v) {
    const int col = (mb * maxT + t) * maxU + u;
    return denom[col] + acts[col * alphabet_size + v];
}

template<typename Tp>
__global__ void compute_alphas_kernel(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    // launch B blocks, each block has U threads
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1); // mb label start point
    const int offset = b * maxT * maxU;
    alphas += offset;
    if (u == 0) alphas[0] = 0;

    __syncthreads();
    for (int n = 1; n < T+U-1; ++n) {
        int t = n - u;
        if (u == 0) {
            if (t > 0 && t < T) {
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t-1, 0, blank_);
            }
        } else if (u < U) {
            if (t == 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, 0, u-1, labels[u-1]);
            else if (t > 0 && t < T) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t-1, u, blank_);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u-1, labels[u-1]);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, U-1, blank_);
        llForward[b] = loglike;
    }
}


template<typename Tp>
__global__ void compute_alpha_delay_kernel(const Tp* const acts, const Tp* const denom, const Tp* delay_values,
    Tp* alpha_delay, Tp* delay_expect_fwd, 
    const Tp* alphas, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU,
    const int alphabet_size, const int blank_){
    /*
        delay_values: from BxT to BxTxU
    */
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1); // mb label start point
    const int offset = b * maxT * maxU;
    alphas += offset;
    alpha_delay+=offset;
    delay_values += offset;
    if (u == 0) alpha_delay[0] = 0;

    __syncthreads();
    for (int n = 1; n < T+U-1; ++n) {
        int t = n - u;
        if (u == 0) {
            if (t > 0 && t < T) {
                alpha_delay[t*maxU+u] = 0;
            }
        } else if (u < U) {
            if (t == 0)
                alpha_delay[u] = alpha_delay[u-1]+delay_values[t*maxU +u];
            else if (t > 0 && t < T) {
                Tp no_emit= alphas[(t-1)*maxU+u]+ logp(denom, acts, maxT,maxU, alphabet_size, b, t-1,u,blank_) -alphas[t*maxU+u];
                no_emit= exp(no_emit)*alpha_delay[(t-1)*maxU+u];
                Tp emit= alphas[t*maxU+u-1] +logp(denom, acts, maxT, maxU, alphabet_size, b, t, u-1, labels[u-1])-alphas[t*maxU+u];
                emit= exp(emit)*(alpha_delay[t*maxU+u-1]+ delay_values[t*maxU + u]);
                alpha_delay[t*maxU+u] = no_emit+emit;
                /* printf("batch=%d, t=%d,u=%d, delay=%f, %f,%f\n", b,t,u, alpha_delay[t*maxU+u], alpha_delay[(t-1)*maxU+u],
                    alpha_delay[t*maxU+u-1]); */
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        Tp delay_expect = alpha_delay[(T-1)*maxU+ U-1];
        delay_expect_fwd[b]= delay_expect;
    }
}

template<typename Tp>
__global__ void compute_alphas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1); // mb label start point
    const int offset = tid * maxT * maxU;
    alphas += offset;
    alphas[0] = 0;

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (u == 0 && t > 0)
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, 0, blank_);
            if (t == 0 && u > 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, 0, u-1, labels[u-1]);
            if (t > 0 && u > 0) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, u, blank_);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u-1, labels[u-1]);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);
    llForward[tid] = loglike;
}


template<typename Tp>
__global__ void compute_betas_kernel(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1);
    const int offset = b * maxT * maxU;
    betas += offset;
    if (u == 0)
        betas[(T-1) * maxU + U-1] = logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, U-1, blank_);

    __syncthreads();
    for (int n = T+U-2; n >= 0; --n) {
        int t = n - u;
        if (u == U-1) {
            if (t >= 0 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, U-1, blank_);
        } else if (u < U) {
            if (t == T-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, u, labels[u]);
            else if (t >= 0 && t < T-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        llBackward[b] = betas[0];
    }
}


template<typename Tp>
__global__ void compute_beta_delay_kernel(const Tp* const acts, const Tp* const denom,const Tp* delay_values,
    Tp* beta_delay, Tp* delay_expect_bwd, const Tp* betas, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_){
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1);
    const int offset = b * maxT * maxU;
    betas += offset;
    beta_delay += offset;
    delay_values += offset;
    if (u == 0)
        beta_delay[(T-1)*maxU + U-1] = 0;

    __syncthreads();
    for (int n = T+U-2; n >= 0; --n) {
        int t = n - u;
        if (u == U-1) {
            if (t >= 0 && t < T-1)
                beta_delay[t*maxU+U-1] = beta_delay[(t+1)*maxU+U-1];
        } else if (u < U) {
            if (t == T-1){
                beta_delay[(T-1)*maxU +u] = beta_delay[(T-1)*maxU +u+1] +delay_values[(T-1)*maxU + u];
                //printf("t=%d,u=%d, delay=%f", t,u, beta_delay[t*maxU +u]);
            }
            else if (t >= 0 && t < T-1) {
                Tp no_emit= betas[(t+1) * maxU + u] +logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_)
                    - betas[t*maxU+u];
                no_emit= exp(no_emit)*beta_delay[(t+1)*maxU+u];
                Tp emit= betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u])
                    - betas[t*maxU+u];
                emit= exp(emit)*(beta_delay[t*maxU+u+1]+delay_values[t*maxU+u]);
                beta_delay[t * maxU + u] = no_emit+emit;
                /* printf("b=%d,t=%d,u=%d, delay=%f,  %f,%f\n",b, t,u, beta_delay[t*maxU +u], beta_delay[(t+1) * maxU + u],
                    beta_delay[t * maxU + u+1]); */
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        delay_expect_bwd[b] = beta_delay[0];
    }

}

template<typename Tp>
__global__ void compute_betas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1);
    const int offset = tid * maxT * maxU;
    betas += offset;
    betas[(T-1) * maxU + U-1] = logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);

    for (int t = T-1; t >=0; --t) {
        for (int u = U-1; u >= 0; --u) {
            if (u == U-1 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, U-1, blank_);
            if (t == T-1 && u < U-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, u, labels[u]);
            if (t < T-1 && u < U-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, blank_);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, labels[u]);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    llBackward[tid] = betas[0];
}

template<int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* grads, const Tp* const acts, const Tp* const denom, const Tp* alphas, const Tp* betas, const Tp* const logll, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // alphabet dim
    int idx = tid;
    int col = blockIdx.x; // mb, t, u

    int u = col % maxU;
    int bt = (col - u) / maxU;
    int t = bt % maxT;
    int mb = (bt - t) / maxT;

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = denom[col] + acts[col * alphabet_size + idx];
            // Tp logpk = logp(denom, acts, maxT, maxU, alphabet_size, mb, t, u, idx);
            Tp grad = exp(alphas[col] + betas[col] + logpk - logll[mb]);
            // grad to last blank transition
            if (idx == blank_ && t == T-1 && u == U-1) {
                grad -= exp(alphas[col] + logpk - logll[mb]);
            }
            if (idx == blank_ && t < T-1) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + maxU]);
            }
            if (u < U-1 && idx == labels[u]) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col+1]);
            }
            grads[col * alphabet_size + idx] = grad;

            idx += NT;
        }
    }
}


template<int NT, typename Tp>
__global__ void compute_grad_withdelay_kernel(Tp* grads, const Tp* const acts, const Tp* const denom,
    const Tp* alphas, const Tp* betas, const Tp* const logll, 
    const Tp* delay_values, const Tp* alpha_delay,const Tp* beta_delay, const Tp* delay_expect_fwd,
    const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, 
    const int maxU, const int alphabet_size, const int blank_, const float delay_scale){
    int tid = threadIdx.x; // alphabet dim
    int idx = tid;
    int col = blockIdx.x; // mb, t, u

    int u = col % maxU;
    int bt = (col - u) / maxU;
    int t = bt % maxT;
    int mb = (bt - t) / maxT;
     /* bool to_print=false;
    if(mb==1 && t==2 && u==2) {
        to_print= true;
    }  */

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = denom[col] + acts[col * alphabet_size + idx];
            Tp logpy =  0;
            if(u<U-1){
                logpy=denom[col] + acts[col * alphabet_size + labels[u]];
            }
            Tp logpb= denom[col] + acts[col*alphabet_size +blank_ ];
            // Tp logpk = logp(denom, acts, maxT, maxU, alphabet_size, mb, t, u, idx);
            Tp grad = exp(alphas[col] + betas[col] + logpk - logll[mb]);
            Tp c_t_u_1 = 0;
            Tp c_t_u_0=0;
            Tp grad2 = 0;
            if(t<T-1){
                c_t_u_0=alpha_delay[col] + beta_delay[col+maxU]-delay_expect_fwd[mb];
                grad2 -= exp(alphas[col]+ betas[col+maxU]+logpk -logll[mb] +logpb)*c_t_u_0;
            }
            if(u<U-1){
                c_t_u_1= alpha_delay[col] + delay_values[bt]+beta_delay[col+ 1] -delay_expect_fwd[mb];
                grad2-=exp(alphas[col]+ betas[col+1]+logpk-logll[mb]+logpy)*c_t_u_1;
            }
            
            
           /* if(to_print && idx <10){
               printf("adelay[col]=%e, bdelay[t+1]=%e, dexp=%e, delay_val=%e, beta[u+1]=%e\n",
                    alpha_delay[col],  beta_delay[col+maxU],delay_expect_fwd[mb], 
                    delay_values[bt], beta_delay[col+ 1]);
                printf("s1=%e,s2=%e,s3=%e,s4=%e\n",alphas[col]+ betas[col+1]+logpk-logll[mb]+logpy,
                    -exp(alphas[col]+ betas[col+1]+logpk-logll[mb]+logpy)*c_t_u_1,
                   alphas[col]+ betas[col+maxU]+logpk -logll[mb] +logpb,
                    exp(alphas[col]+ betas[col+maxU]+logpk -logll[mb] +logpb)*c_t_u_0);
                printf("alpha=%e,beta=%e, lpk=%e,lpb=%e, logll=%e\n",alphas[col] ,betas[col],logpk,logpb,logll[mb]) ;
                printf("%d,grad=%.3f,grad2=%.3f, c1=%.3f,c0=%.3f\n",idx,grad, grad2, c_t_u_1,c_t_u_0);
            }   */
            // grad to last blank transition
            if (idx == blank_ && t == T-1 && u == U-1) {
                grad -= exp(alphas[col] + logpk - logll[mb]);
            }
            if (idx == blank_ && t < T-1) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + maxU]);
                grad2+= exp(alphas[col] + betas[col+maxU]+logpb-logll[mb])*c_t_u_0;
            }
            if (u < U-1 && idx == labels[u]) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col+1]);
                 grad2+= exp(alphas[col] + betas[col+1]+logpy-logll[mb])*c_t_u_1;
            }
            grads[col * alphabet_size + idx] = grad + delay_scale*grad2;
            /* if(to_print && idx <10){
                printf("last%d: gall=%f,grad=%.3f,grad2=%.3f, c1=%.3f,c0=%.3f\n",idx, grads[col * alphabet_size + idx],
                    grad, grad2, c_t_u_1,c_t_u_0);
            } */
            idx += NT;
        }
    }

}


template<int NT, typename Tp>
__global__ void compute_grad_withdelay_smooth_kernel(Tp* grads, const Tp* const acts, const Tp* const denom,
    const Tp* alphas, const Tp* betas, const Tp* const logll, 
    const Tp* delay_values, const Tp* alpha_delay,const Tp* beta_delay, const Tp* delay_expect_fwd,
    const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, 
    const int maxU, const int alphabet_size, const int blank_, const float delay_scale, const float smooth){
    int tid = threadIdx.x; // alphabet dim
    int idx = tid;
    int col = blockIdx.x; // mb, t, u

    int u = col % maxU;
    int bt = (col - u) / maxU;
    int t = bt % maxT;
    int mb = (bt - t) / maxT;
     /* bool to_print=false;
    if(mb==1 && t==2 && u==2) {
        to_print= true;
    }  */

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = denom[col] + acts[col * alphabet_size + idx];
            Tp logpy =  0;
            if(u<U-1){
                logpy=denom[col] + acts[col * alphabet_size + labels[u]];
            }
            Tp logpb= denom[col] + acts[col*alphabet_size +blank_ ];
            // Tp logpk = logp(denom, acts, maxT, maxU, alphabet_size, mb, t, u, idx);
            Tp grad = exp((alphas[col] + betas[col]- logll[mb])*smooth + logpk );
            Tp c_t_u_1 = 0;
            Tp c_t_u_0=0;
            Tp grad2 = 0;
            if(t<T-1){
                c_t_u_0=alpha_delay[col] + beta_delay[col+maxU]-delay_expect_fwd[mb];
                grad2 -= exp(alphas[col]+ betas[col+maxU]+logpk -logll[mb] +logpb)*c_t_u_0;
            }
            if(u<U-1){
                c_t_u_1= alpha_delay[col] + delay_values[bt]+beta_delay[col+ 1] -delay_expect_fwd[mb];
                grad2-=exp(alphas[col]+ betas[col+1]+logpk-logll[mb]+logpy)*c_t_u_1;
            }

            // grad to last blank transition
            if (idx == blank_ && t == T-1 && u == U-1) {
                grad -= exp(smooth*(alphas[col] - logll[mb]+ logpk) );
            }
            if (idx == blank_ && t < T-1) {
                grad -= exp(smooth*(alphas[col] - logll[mb] + betas[col + maxU]+ logpk) );
                grad2+= exp(alphas[col] + betas[col+maxU]+logpb-logll[mb])*c_t_u_0;
            }
            if (u < U-1 && idx == labels[u]) {
                grad -= exp(smooth*(alphas[col] + betas[col+1]- logll[mb]+ logpk));
                 grad2+= exp(alphas[col] + betas[col+1]+logpy-logll[mb])*c_t_u_1;
            }
            grads[col * alphabet_size + idx] = grad + delay_scale*grad2;
    
            idx += NT;
        }
    }

}