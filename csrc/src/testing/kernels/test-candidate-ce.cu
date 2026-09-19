#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "kernels/kernels.h"

namespace {
template<class T> struct Device {
    T* ptr = nullptr;
    explicit Device(const std::vector<T>& values) {
        REQUIRE(cudaMalloc(&ptr, values.size()*sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMemcpy(ptr, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~Device() { cudaFree(ptr); }
    std::vector<T> read(int n) {
        std::vector<T> result(n);
        REQUIRE(cudaMemcpy(result.data(), ptr, n*sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
        return result;
    }
};

template<class T> void check_candidate_ce(int vocab, float softcap) {
    constexpr int rows=3, K=4;
    std::vector<T> logits(rows*vocab);
    for (int i=0;i<rows*vocab;++i) logits[i]=T(float((i*17)%41-20)/8.0f);
    logits[16]=T(40.0f); // A very large excluded logit must have zero derivative.
    std::vector<int> targets{2,13,-100}, ids{2,7,25,-1,1,13,-1,-1,-1,-1,-1,-1};
    std::vector<float> scale{0.7f,2.0f,1.0f}, lse(rows,0), losses(rows,0), expected(rows*vocab,0);
    std::vector<float> candidate_loss(rows,0);
    auto capped=[&](int row,int id){ double x=float(logits[row*vocab+id]);return softcap>0?softcap*std::tanh(x/softcap):x; };
    for (int row=0;row<2;++row) {
        double maximum=-INFINITY,sum=0;
        for(int j=0;j<vocab;++j) maximum=std::max(maximum,capped(row,j));
        for(int j=0;j<vocab;++j) sum+=std::exp(capped(row,j)-maximum);
        lse[row]=float(maximum+std::log(sum));
        losses[row]=lse[row]-float(capped(row,targets[row]));
        maximum=-INFINITY;sum=0;
        for(int k=0;k<K;++k) if(ids[row*K+k]>=0) maximum=std::max(maximum,capped(row,ids[row*K+k]));
        for(int k=0;k<K;++k) if(ids[row*K+k]>=0) sum+=std::exp(capped(row,ids[row*K+k])-maximum);
        const double restricted_lse=maximum+std::log(sum);
        candidate_loss[row]=float(restricted_lse-capped(row,targets[row]));
        for(int k=0;k<K;++k) {
            int id=ids[row*K+k]; if(id<0)continue;
            double g=(std::exp(capped(row,id)-restricted_lse)-(id==targets[row]))*scale[row];
            if(softcap>0)g*=1-std::pow(capped(row,id)/softcap,2);
            expected[row*vocab+id]=float(g);
        }
    }
    Device<T> buffer(logits);
    Device<int> d_targets(targets), d_ids(ids);
    Device<float> d_scale(scale), d_lse(lse), d_losses(std::vector<float>(rows,0)), accumulator(std::vector<float>{0});
    KdBackwardArgs args;
    args.candidate_only=true;args.ids=d_ids.ptr;args.K=K;
    args.kd_loss_accum=accumulator.ptr;
    candidate_cross_entropy_forward(buffer.ptr,d_losses.ptr,d_targets.ptr,d_ids.ptr,nullptr,nullptr,rows,vocab,vocab,K,softcap,nullptr);
    if(vocab>CROSS_ENTROPY_MAX_FUSED_SIZE) {
        chunked_cross_entropy_backward(buffer.ptr,buffer.ptr,d_lse.ptr,d_scale.ptr,d_targets.ptr,rows,vocab,vocab,softcap,nullptr,&args);
    } else {
        fused_cross_entropy_backward(buffer.ptr,buffer.ptr,d_lse.ptr,d_scale.ptr,d_targets.ptr,rows,vocab,vocab,softcap,nullptr,&args);
    }
    REQUIRE(cudaDeviceSynchronize()==cudaSuccess);
    auto grads=buffer.read(rows*vocab);
    for(int i=0;i<rows*vocab;++i) {
        if(expected[i]==0) REQUIRE(float(grads[i])==0.0f);
        else REQUIRE(float(grads[i])==Catch::Approx(expected[i]).margin(0.008));
    }
    const auto corrected=d_losses.read(rows);
    for(int row=0;row<rows;++row) REQUIRE(corrected[row]==Catch::Approx(candidate_loss[row]).margin(1e-5));
    REQUIRE(accumulator.read(1)[0]==Catch::Approx(candidate_loss[0]+candidate_loss[1]).margin(1e-5));
}
}

TEST_CASE("Candidate CE excludes vocabulary and handles aliasing, masking and softcap", "[candidate-ce]") {
    for(int vocab : {37, 131072}) for(float softcap : {0.0f,5.0f}) {
        check_candidate_ce<float>(vocab,softcap);
        check_candidate_ce<nv_bfloat16>(vocab,softcap);
    }
}
