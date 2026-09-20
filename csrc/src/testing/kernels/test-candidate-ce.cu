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

namespace {
// Independent double-precision hard-target oracle, including an explicit
// softmax Jacobian. No batch/K normalization except RPS's K-1 boundaries.
std::pair<double, std::vector<double>> proper_oracle(const std::vector<double>& z,
                                                  int gold, CandidateObjective objective) {
    const int count = static_cast<int>(z.size());
    std::vector<double> p(count), derivative(count, 0), gradient(count, 0);
    const double maximum = *std::max_element(z.begin(), z.end());
    double sum = 0, loss = 0;
    for (int i=0;i<count;++i) sum += p[i] = std::exp(z[i]-maximum);
    for (auto& value : p) value /= sum;
    if (objective == CandidateObjective::Brier) {
        for (int i=0;i<count;++i) {
            const double error = p[i] - double(i==gold);
            loss += error*error;
            derivative[i] = 2*error;
        }
    } else {
        double cdf = 0;
        for (int k=0;k<count-1;++k) {
            cdf += p[k] - double(k==gold);
            loss += cdf*cdf/(count-1);
            for (int i=0;i<=k;++i) derivative[i] += 2*cdf/(count-1);
        }
    }
    for (int i=0;i<count;++i) for (int j=0;j<count;++j) {
        gradient[i] += derivative[j]*p[j]*(double(i==j)-p[i]);
    }
    return {loss, gradient};
}

template<class T> void check_candidate_proper(int vocab, int allowed, float softcap,
                                             CandidateObjective objective) {
    constexpr int rows = 3;
    const int slots = 2*allowed+1;
    std::vector<T> logits(rows*vocab);
    for (int i=0;i<rows*vocab;++i) logits[i] = T(float((i*17)%41-20)/8.0f);
    std::vector<int> ids(rows*slots, -1), targets(rows, -100);
    std::vector<float> scale{0.7f,2.0f,1.0f}, expected(rows*vocab,0), losses(rows,0);
    int expected_correct = 0;
    for (int row=0;row<2;++row) {
        const int count = std::max(2, allowed-row);
        std::vector<double> capped;
        for (int k=0;k<count;++k) {
            // Nonmonotone IDs ensure ordinal order is the sidecar order.
            const int id = (k*7+3)%vocab;
            ids[row*slots+2*k+1] = id;
            const double raw = float(logits[row*vocab+id]);
            capped.push_back(softcap>0 ? softcap*std::tanh(raw/softcap) : raw);
        }
        targets[row] = ids[row*slots+2*(count/2)+1];
        const auto [loss, gradient] = proper_oracle(capped, count/2, objective);
        losses[row] = float(loss);
        expected_correct += int(std::max_element(capped.begin(),capped.end())-capped.begin()) == count/2;
        for (int k=0;k<count;++k) {
            double value = gradient[k]*scale[row];
            if (softcap>0) value *= 1-std::pow(capped[k]/softcap,2);
            expected[row*vocab+ids[row*slots+2*k+1]] = float(value);
        }
        // Huge excluded logits must affect neither the loss nor derivatives.
        int excluded = 0;
        while (std::find(ids.begin()+row*slots,ids.begin()+(row+1)*slots,excluded) != ids.begin()+(row+1)*slots) ++excluded;
        logits[row*vocab+excluded] = T(80.0f);
    }
    Device<T> buffer(logits);
    Device<int> d_targets(targets), d_ids(ids), valid(std::vector<int>{0}), correct(std::vector<int>{0});
    Device<float> d_scale(scale), d_losses(std::vector<float>(rows,0)), accumulator(std::vector<float>{0});
    KdBackwardArgs args;
    args.candidate_only=true; args.candidate_objective=objective; args.ids=d_ids.ptr;
    args.K=slots; args.kd_loss_accum=accumulator.ptr;
    candidate_cross_entropy_forward(buffer.ptr,d_losses.ptr,d_targets.ptr,d_ids.ptr,valid.ptr,correct.ptr,
                                    rows,vocab,vocab,slots,softcap,nullptr,objective);
    if (vocab>CROSS_ENTROPY_MAX_FUSED_SIZE) {
        chunked_cross_entropy_backward(buffer.ptr,buffer.ptr,nullptr,d_scale.ptr,d_targets.ptr,
                                       rows,vocab,vocab,softcap,nullptr,&args);
    } else {
        fused_cross_entropy_backward(buffer.ptr,buffer.ptr,nullptr,d_scale.ptr,d_targets.ptr,
                                     rows,vocab,vocab,softcap,nullptr,&args);
    }
    REQUIRE(cudaDeviceSynchronize()==cudaSuccess);
    const auto gradients=buffer.read(rows*vocab);
    const auto actual_loss=d_losses.read(rows);
    REQUIRE(valid.read(1)[0]==2);
    REQUIRE(correct.read(1)[0]==expected_correct);
    for (int row=0;row<rows;++row) {
        REQUIRE(actual_loss[row]==Catch::Approx(losses[row]).margin(2e-5));
        double gradient_sum=0;
        for (int id=0;id<vocab;++id) {
            const int index=row*vocab+id;
            if (expected[index]==0) REQUIRE(float(gradients[index])==0.0f);
            else REQUIRE(float(gradients[index])==Catch::Approx(expected[index]).margin(2e-6).epsilon(sizeof(T)==4 ? 1e-5 : 0.004));
            gradient_sum += float(gradients[index]);
        }
        if (softcap==0) REQUIRE(std::abs(gradient_sum)<0.004);
    }
    // Raw summed objectives, not multiplied by dloss, nor divided by rows/K.
    REQUIRE(accumulator.read(1)[0]==Catch::Approx(losses[0]+losses[1]).margin(2e-5));
}
}

TEST_CASE("Candidate Brier and RPS preserve masks, order, aliasing and FP32 loss math", "[candidate-proper]") {
    for (auto objective : {CandidateObjective::Brier, CandidateObjective::Rps}) {
        for (int vocab : {509,131072}) for (int allowed : {2,3,255}) for (float cap : {0.0f,5.0f}) {
            check_candidate_proper<float>(vocab,allowed,cap,objective);
            check_candidate_proper<nv_bfloat16>(vocab,allowed,cap,objective);
        }
    }
}

TEST_CASE("Proper-score oracle gradients and ordinal semantics", "[candidate-proper]") {
    const std::vector<double> z{0.2,-0.7,1.3};
    for (auto objective : {CandidateObjective::Brier,CandidateObjective::Rps}) {
        const auto [loss,gradient]=proper_oracle(z,0,objective);
        for (int i=0;i<3;++i) {
            auto plus=z,minus=z; plus[i]+=1e-5;minus[i]-=1e-5;
            const double numerical=(proper_oracle(plus,0,objective).first-proper_oracle(minus,0,objective).first)/2e-5;
            REQUIRE(gradient[i]==Catch::Approx(numerical).margin(1e-8));
        }
        const std::vector<double> permuted{z[1],z[0],z[2]};
        const auto changed=proper_oracle(permuted,1,objective).first;
        if (objective==CandidateObjective::Brier) REQUIRE(changed==Catch::Approx(loss));
        else REQUIRE(std::abs(changed-loss)>0.01);
    }
}

TEST_CASE("Binary proper-score scaling is explicit", "[candidate-proper]") {
    const std::vector<double> z{0.2,-0.8};
    const auto brier=proper_oracle(z,1,CandidateObjective::Brier);
    const auto rps=proper_oracle(z,1,CandidateObjective::Rps);
    REQUIRE(brier.first==Catch::Approx(2*rps.first));
    for (int k=0;k<2;++k) REQUIRE(brier.second[k]==Catch::Approx(2*rps.second[k]));
}

TEST_CASE("LoRA DP compensation reaches norm and optimizer gradient scale", "[candidate-proper]") {
    cudaDeviceProp properties{};
    REQUIRE(cudaGetDeviceProperties(&properties,0)==cudaSuccess);
    for (int world : {1,2}) for (int count : {1,3,6}) for (float clip : {0.0f,1.0f}) {
        Device<float> buffer(std::vector<float>{25,0}), amax(std::vector<float>{2});
        Device<int> valid(std::vector<int>{count});
        // The averaged gradient norm is10. Compensation produces the norm
        // of the globally summed gradient before the valid-token denominator.
        global_norm_sqrt_prescaled(buffer.ptr,nullptr,clip,valid.ptr,999,amax.ptr,
                                   properties,nullptr,float(world));
        REQUIRE(cudaDeviceSynchronize()==cudaSuccess);
        auto value=buffer.read(2);
        const float expected_norm=10.0f*world/count;
        const float expected_scale=float(world)/count*(clip>0 ? std::min(1.0f,clip/expected_norm) : 1.0f);
        REQUIRE(value[0]==Catch::Approx(expected_norm));
        REQUIRE(value[1]==Catch::Approx(expected_scale));
    }
}
