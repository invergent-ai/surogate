#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "utilities/dtype.h"

namespace dsl {
namespace {

int normalize_dim(int dim, int rank, const char* op_name) {
    int out = dim;
    if (out < 0) {
        out += rank;
    }
    if (out < 0 || out >= rank) {
        throw std::runtime_error(std::string(op_name) + ": dim out of range");
    }
    return out;
}

long product_range(const Tensor& t, int start, int end) {
    long p = 1;
    for (int i = start; i < end; ++i) {
        p *= t.Sizes[i];
    }
    return p;
}

bool shape_equal(const Tensor& t, const std::vector<long>& shape) {
    return tensor_shape_matches(t, shape);
}

bool ends_with_local(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string shape_to_string(const Tensor& t) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < t.Rank; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << t.Sizes[i];
    }
    oss << "]";
    return oss.str();
}

std::string join_sizes(const std::vector<long>& values) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

std::string join_output_names(const std::vector<TensorRef>& outputs) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << outputs[i].name;
    }
    oss << "]";
    return oss.str();
}

}  // namespace

void CompiledExecutor::dispatch_concat(const CompiledOp& op) {
    if (op.inputs.empty()) {
        throw std::runtime_error("dispatch_concat: expected at least one input");
    }
    if (op.outputs.empty() || op.outputs[0].name.empty()) {
        throw std::runtime_error("dispatch_concat: expected one non-empty output");
    }

    std::vector<Tensor*> inputs;
    inputs.reserve(op.inputs.size());
    for (const auto& in_ref : op.inputs) {
        if (in_ref.name.empty()) {
            throw std::runtime_error("dispatch_concat: empty input name");
        }
        inputs.push_back(&resolve_tensor(in_ref));
    }

    const Tensor& first = *inputs[0];
    if (first.Rank <= 0) {
        throw std::runtime_error("dispatch_concat: input rank must be > 0");
    }
    const int rank = first.Rank;
    const int dim = normalize_dim(op.attrs.split_concat_dim, rank, "dispatch_concat");

    for (std::size_t i = 1; i < inputs.size(); ++i) {
        const Tensor& t = *inputs[i];
        if (t.DType != first.DType) {
            throw std::runtime_error("dispatch_concat: all inputs must have same dtype");
        }
        if (t.Rank != rank) {
            throw std::runtime_error("dispatch_concat: all inputs must have same rank");
        }
        for (int d = 0; d < rank; ++d) {
            if (d == dim) continue;
            if (t.Sizes[d] != first.Sizes[d]) {
                throw std::runtime_error("dispatch_concat: non-concat dimensions must match");
            }
        }
    }

    std::vector<long> out_shape(first.Sizes.begin(), first.Sizes.begin() + rank);
    long out_dim = 0;
    for (const Tensor* t : inputs) {
        out_dim += t->Sizes[dim];
    }
    out_shape[dim] = out_dim;

    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.DType != first.DType || !shape_equal(out, out_shape)) {
        out = mRunState.temp_alloc(first.DType, out_shape, "concat_output");
        mTemps.push_back(out);
    }

    const std::size_t elem_bytes = get_dtype_size(first.DType);
    const long outer = product_range(first, 0, dim);
    const long inner = product_range(first, dim + 1, rank);
    const long out_dim_size = out_shape[dim];
    std::byte* out_ptr = static_cast<std::byte*>(out.Data);

    if (outer > 0) {
        const std::size_t dst_pitch =
            static_cast<std::size_t>(out_dim_size) * static_cast<std::size_t>(inner) * elem_bytes;
        long out_offset_dim = 0;
        for (const Tensor* in_t : inputs) {
            const long in_dim = in_t->Sizes[dim];
            const std::size_t row_bytes =
                static_cast<std::size_t>(in_dim) * static_cast<std::size_t>(inner) * elem_bytes;
            std::byte* dst_base =
                out_ptr + static_cast<std::size_t>(out_offset_dim) * static_cast<std::size_t>(inner) * elem_bytes;
            const std::byte* src_base = static_cast<const std::byte*>(in_t->Data);
            if (outer == 1) {
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_base, src_base, row_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    dst_base, dst_pitch,
                    src_base, row_bytes,
                    row_bytes, static_cast<std::size_t>(outer),
                    cudaMemcpyDeviceToDevice,
                    mRunState.MainStream));
            }
            out_offset_dim += in_dim;
        }
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_split(const CompiledOp& op) {
    if (op.inputs.size() != 1) {
        throw std::runtime_error("dispatch_split: expected exactly one input");
    }
    if (op.outputs.empty()) {
        throw std::runtime_error("dispatch_split: expected at least one output");
    }

    Tensor& in = resolve_tensor(op.inputs[0]);
    if (in.Rank <= 0) {
        throw std::runtime_error("dispatch_split: input rank must be > 0");
    }
    const int rank = in.Rank;
    const int dim = normalize_dim(op.attrs.split_concat_dim, rank, "dispatch_split");
    const long in_dim = in.Sizes[dim];

    std::vector<long> split_sizes = op.attrs.split_sizes;
    if (split_sizes.size() == 1 && op.outputs.size() > 1) {
        const long chunk = split_sizes[0];
        split_sizes.clear();
        if (chunk <= 0) {
            throw std::runtime_error("dispatch_split: split_size must be > 0");
        }
        long remaining = in_dim;
        while (remaining > 0) {
            const long take = std::min(chunk, remaining);
            split_sizes.push_back(take);
            remaining -= take;
        }
    }

    if (split_sizes.empty()) {
        // Qwen3.5 full-attention backward: infer [Q,K,V] partitions from head ratio
        // when autodiff produced split without explicit sizes.
        const bool grad_concat_input = !op.inputs.empty() &&
                                       op.inputs[0].name.size() > 2 &&
                                       op.inputs[0].name[0] == 'd' &&
                                       op.inputs[0].name[1] == '_';
        const bool qkv_grad_outputs = op.outputs.size() == 3 &&
                                      ends_with_local(op.outputs[0].name, ".full_q") &&
                                      ends_with_local(op.outputs[1].name, ".full_k") &&
                                      ends_with_local(op.outputs[2].name, ".full_v");
        if (grad_concat_input && qkv_grad_outputs) {
            const long hq = static_cast<long>(mConfig.NumQueryHeads);
            const long hkv = static_cast<long>(mConfig.NumKeyValHeads);
            const long denom = hq + 2 * hkv;
            if (denom > 0 && in_dim % denom == 0) {
                const long scale = in_dim / denom;
                split_sizes = {scale * hq, scale * hkv, scale * hkv};
            }
        }
    }

    if (split_sizes.empty()) {
        bool inferred = true;
        split_sizes.reserve(op.outputs.size());
        for (const auto& out_ref : op.outputs) {
            if (out_ref.shape.size() != static_cast<std::size_t>(rank)) {
                inferred = false;
                break;
            }
            int out_dim = dim;
            if (out_dim < 0) {
                out_dim += static_cast<int>(out_ref.shape.size());
            }
            if (out_dim < 0 || out_dim >= static_cast<int>(out_ref.shape.size())) {
                inferred = false;
                break;
            }
            split_sizes.push_back(out_ref.shape[static_cast<std::size_t>(out_dim)]);
        }
        if (!inferred) {
            split_sizes.clear();
        }
    }

    if (split_sizes.empty()) {
        if (in_dim % static_cast<long>(op.outputs.size()) != 0) {
            throw std::runtime_error(
                "dispatch_split: cannot infer split sizes (input dim not divisible by output count)");
        }
        const long per = in_dim / static_cast<long>(op.outputs.size());
        split_sizes.assign(op.outputs.size(), per);
    }

    if (split_sizes.size() != op.outputs.size()) {
        throw std::runtime_error("dispatch_split: split size count must match number of outputs");
    }
    if (const char* dbg = std::getenv("SUROGATE_DEBUG_QWEN35_BWD");
        dbg && std::string(dbg) == "1") {
        const bool grad_concat_input = !op.inputs.empty() &&
                                       op.inputs[0].name.size() > 2 &&
                                       op.inputs[0].name[0] == 'd' &&
                                       op.inputs[0].name[1] == '_';
        const bool qkv_grad_outputs = op.outputs.size() == 3 &&
                                      ends_with_local(op.outputs[0].name, ".full_q") &&
                                      ends_with_local(op.outputs[1].name, ".full_k") &&
                                      ends_with_local(op.outputs[2].name, ".full_v");
        if (grad_concat_input && qkv_grad_outputs) {
            fprintf(stderr,
                    "[QWEN35_BWD][split] input=%s in_shape=%s dim=%d in_dim=%ld split_sizes=%s outputs=%s\n",
                    op.inputs[0].name.c_str(),
                    shape_to_string(in).c_str(),
                    dim,
                    in_dim,
                    join_sizes(split_sizes).c_str(),
                    join_output_names(op.outputs).c_str());
        }
    }
    const long sum = std::accumulate(split_sizes.begin(), split_sizes.end(), 0L);
    if (sum != in_dim) {
        std::ostringstream oss;
        oss << "dispatch_split: split sizes must sum to input size along dim"
            << " (input=" << op.inputs[0].name
            << ", in_shape=" << shape_to_string(in)
            << ", dim=" << dim
            << ", in_dim=" << in_dim
            << ", split_sizes=" << join_sizes(split_sizes)
            << ", outputs=" << join_output_names(op.outputs)
            << ")";
        throw std::runtime_error(oss.str());
    }

    const std::size_t elem_bytes = get_dtype_size(in.DType);
    const long outer = product_range(in, 0, dim);
    const long inner = product_range(in, dim + 1, rank);
    const std::byte* in_ptr = static_cast<const std::byte*>(in.Data);

    long dim_offset = 0;
    for (std::size_t i = 0; i < op.outputs.size(); ++i) {
        const long chunk = split_sizes[i];
        if (chunk <= 0) {
            throw std::runtime_error("dispatch_split: split sizes must be > 0");
        }

        std::vector<long> out_shape(in.Sizes.begin(), in.Sizes.begin() + rank);
        out_shape[dim] = chunk;

        if (op.outputs[i].name.empty()) {
            dim_offset += chunk;
            continue;
        }

        Tensor& out_ref = ensure_output_tensor(op.outputs[i]);
        Tensor out = out_ref;
        // Check if multiple outputs share the same slot/tensor_id (happens when
        // they're both mapped to the generic slot). If so, always alloc a temp
        // to prevent overlapping writes.
        bool shared_slot = false;
        for (std::size_t j = 0; j < op.outputs.size(); ++j) {
            if (j != i && op.outputs[j].tensor_id == op.outputs[i].tensor_id &&
                op.outputs[j].tensor_id >= 0) {
                shared_slot = true;
                break;
            }
        }
        if (shared_slot || !out.Data || out.DType != in.DType || !shape_equal(out, out_shape)) {
            out = mRunState.temp_alloc(in.DType, out_shape, "split_output");
            mTemps.push_back(out);
        }
        std::byte* out_ptr = static_cast<std::byte*>(out.Data);

        if (outer > 0) {
            const std::size_t src_pitch =
                static_cast<std::size_t>(in_dim) * static_cast<std::size_t>(inner) * elem_bytes;
            const std::size_t row_bytes =
                static_cast<std::size_t>(chunk) * static_cast<std::size_t>(inner) * elem_bytes;
            const std::byte* src_base =
                in_ptr + static_cast<std::size_t>(dim_offset) * static_cast<std::size_t>(inner) * elem_bytes;
            if (outer == 1) {
                CUDA_CHECK(cudaMemcpyAsync(
                    out_ptr, src_base, row_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
            } else {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    out_ptr, row_bytes,
                    src_base, src_pitch,
                    row_bytes, static_cast<std::size_t>(outer),
                    cudaMemcpyDeviceToDevice,
                    mRunState.MainStream));
            }
        }

        store_tensor(op.outputs[i], out);
        dim_offset += chunk;
    }
}

}  // namespace dsl
