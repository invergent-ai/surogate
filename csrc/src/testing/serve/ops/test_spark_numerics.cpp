#include "api/ops/cast.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/sigmoid_mul.h"
#include "ops/op_tester.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {
int residual_case(int width, int tokens) {
    const auto count = static_cast<std::size_t>(width) * tokens;
    std::vector<float> input(count), weight(width), delta(count);
    fill_uniform(input, 412, -3, 3);
    fill_uniform(weight, 413, .5, 1.5);
    fill_uniform(delta, 414, -.02, .02);
    round_to_bf16(input); round_to_bf16(weight); round_to_bf16(delta);
    auto di = to_device_bf16(input), dw = to_device_bf16(weight), dy = to_device_bf16(delta);
    GuardedDeviceBuffer residual(count * sizeof(float)), output(count * 2);
    Tensor ti(di.p, DType::BF16, {width,tokens}), tw(dw.p, DType::BF16, {width});
    Tensor ty(dy.p, DType::BF16, {width,tokens});
    Tensor tx(residual.data(), DType::FP32, {width,tokens});
    Tensor to(output.data(), DType::BF16, {width,tokens});
    ops::cast_bf16_to_fp32(ti,tx,nullptr);
    // Repeated small updates must survive without BF16 residual rounding.
    for (int step=0;step<17;++step) {
        ops::residual_add(ty,tx,nullptr);
        for (std::size_t i=0;i<count;++i) { input[i] += delta[i]; }
    }
    ops::rmsnorm(tx,tw,1e-6F,false,to,nullptr);
    cuda_synchronize();
    const auto actual=from_device<float>(residual.data(),count);
    int failures=0;
    for (std::size_t i=0;i<count;++i) { if (actual[i]!=input[i]) { ++failures; break; } }
    std::vector<double> expected(count);
    for (int t=0;t<tokens;++t) {
        double squares=0;
        for(int d=0;d<width;++d) { const double x=input[t*width+d];squares+=x*x; }
        const double inv=1/std::sqrt(squares/width+1e-6);
        for(int d=0;d<width;++d) { expected[t*width+d]=input[t*width+d]*inv*weight[d]; }
    }
    failures+=verify_pointwise("FP32 residual RMSNorm",from_device_bf16(output.data(),count),expected,{1e-6,4e-3});
    failures+=residual.verify_guards("FP32 residual");
    failures+=output.verify_guards("RMSNorm BF16 output");
    return failures;
}

int gates_case(int dim,int heads,int tokens) {
    const int count=dim*heads*tokens;
    std::vector<float> gate(heads*tokens), input(count);
    fill_uniform(gate,501,-12,12);fill_uniform(input,502,-5,5);
    round_to_bf16(gate);round_to_bf16(input);
    auto dg=to_device_bf16(gate),dx=to_device_bf16(input);
    Tensor tg(dg.p,DType::BF16,{heads,tokens}),tx(dx.p,DType::BF16,{dim,heads,tokens});
    ops::headwise_sigmoid_mul(tg,tx,nullptr);cuda_synchronize();
    std::vector<std::uint16_t> expected(count);
    for(int i=0;i<count;++i) {
        const float scale=bf16_to_f32(f32_to_bf16(1.0F/(1.0F+std::exp(-gate[i/dim]))));
        expected[i]=f32_to_bf16(input[i]*scale);
    }
    return verify_exact("headwise rounded sigmoid",from_device<std::uint16_t>(dx.p,count),expected);
}

int gelu_case(int n) {
    std::vector<float> gate(n),up(n);
    fill_uniform(gate,601,-3,3);fill_uniform(up,602,-4,4);
    round_to_bf16(gate);round_to_bf16(up);
    auto dg=to_device_bf16(gate),du=to_device_bf16(up);
    GuardedDeviceBuffer output(n*2);
    Tensor tg(dg.p,DType::BF16,{n}),tu(du.p,DType::BF16,{n}),to(output.data(),DType::BF16,{n});
    ops::gelu_mul(tg,tu,ops::GeluMode::Exact,to,nullptr,true);cuda_synchronize();
    std::vector<std::uint16_t> expected(n);
    for(int i=0;i<n;++i) {
        const double gelu=.5*gate[i]*(1+std::erf(gate[i]/std::sqrt(2.0)));
        expected[i]=f32_to_bf16(bf16_to_f32(f32_to_bf16(static_cast<float>(gelu)))*up[i]);
    }
    return verify_exact("rounded exact GELU gate",from_device<std::uint16_t>(output.data(),n),expected)
        +output.verify_guards("rounded GELU output");
}
}
int main() {
    if(cuda_unavailable()) { return 77; }
    int failures=0;
    for(int width:{2048,2560,13}) for(int tokens:{1,17}) failures+=residual_case(width,tokens);
    for(int heads:{8,16}) for(int tokens:{1,3,129}) failures+=gates_case(256,heads,tokens);
    failures+=gelu_case(6656);failures+=gelu_case(10241);
    std::cout<<(failures?"FAIL":"OK")<<" Spark numerical semantics\n";
    return failures?1:0;
}
