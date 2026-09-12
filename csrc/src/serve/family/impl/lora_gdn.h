#pragma once

#include "family/impl/lora_hook.h"
#include "family/impl/lora_globals.h"
#include "api/ops/linear.h"
#include "core/layout.h"
#include <variant>

namespace sinfer::family {
inline constexpr int kGdnInputPort  = 8;
inline constexpr int kGdnZPort      = 9;
inline constexpr int kGdnAPort      = 10;
inline constexpr int kGdnBPort      = 11;
inline constexpr int kGdnOutputPort = 12;

template <class P>
const Weight& gdn_input_key(const P& p) {
    if constexpr (requires { std::variant_size<P>::value; }) {
        return std::visit([](const auto& value) -> const Weight& { return gdn_input_key(value); },
                          p);
    } else if constexpr (requires { p.input_projection; }) {
        return gdn_input_key(p.input_projection);
    } else if constexpr (requires { p.split; }) {
        return p.split ? p.split->query_key_value : p.query_key_value_z;
    } else if constexpr (requires { p.query_key_value; }) {
        return p.query_key_value;
    } else if constexpr (requires { p.query_key; }) {
        return p.query_key;
    } else {
        return p.query_key_value_z;
    }
}

template <class P>
const Weight& gdn_control_key(const P& p) {
    if constexpr (requires { std::variant_size<P>::value; }) {
        return std::visit([](const auto& value) -> const Weight& { return gdn_control_key(value); },
                          p);
    } else if constexpr (requires { p.control_projection; }) {
        return gdn_control_key(p.control_projection);
    } else if constexpr (requires { p.a_projection; }) {
        return p.a_projection;
    } else {
        return p.a_b_projection;
    }
}

template <class P>
void project_gdn_control(const Tensor& hidden, const P& p, Tensor& a, Tensor& b,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    if constexpr (requires { std::variant_size<P>::value; }) {
        std::visit(
            [&](const auto& value) { project_gdn_control(hidden, value, a, b, workspace, stream); },
            p);
    } else if constexpr (requires { p.control_projection; }) {
        project_gdn_control(hidden, p.control_projection, a, b, workspace, stream);
    } else if constexpr (requires { p.a_projection; }) {
        ops::linear_projections(hidden, {{p.a_projection, a}, {p.b_projection, b}}, nullptr, stream);
    } else {
        ops::linear_projections(hidden, {{p.a_b_projection, a, ops::LinearPolicy::A16Only, 0},
                                         {p.a_b_projection, b, ops::LinearPolicy::A16Only, a.ne[0]}},
                                &workspace, stream);
    }
}

template <class P>
bool gdn_lora_input_bound(const P& p) {
    const auto& key = gdn_input_key(p);
    return lora_bound(key, kGdnInputPort) || lora_bound(key, kGdnZPort);
}

template <class P>
void apply_lora_gdn_input(const P& p, const Tensor& hidden, Tensor& qkv, Tensor& z,
                          cudaStream_t stream) {
    const auto& key = gdn_input_key(p);
    apply_lora(key, kGdnInputPort, hidden, qkv, stream);
    apply_lora(key, kGdnZPort, hidden, z, stream);
}

template<class P, class Geometry>
void bind_gdn_input_base(ops::LoraStore& store, const P& p, const Geometry& g) {
    if constexpr (requires { std::variant_size<P>::value; }) {
        std::visit([&](const auto& v) { bind_gdn_input_base(store, v, g); }, p);
    } else if constexpr (requires { p.input_projection; }) {
        bind_gdn_input_base(store, p.input_projection, g);
    } else if constexpr (requires { p.split; }) {
        if (p.split) { bind_gdn_input_base(store, *p.split, g); }
        else {
            const auto& w = p.query_key_value_z;
            store.register_base(w.qdata, kGdnInputPort, {{w, 0, 0, g.convolution_dim()}});
            store.register_base(w.qdata, kGdnZPort, {{w, g.convolution_dim(), 0, g.value_dim()}});
        }
    } else if constexpr (requires { p.query_key_value; p.z; }) {
        store.register_base(p.query_key_value.qdata, kGdnInputPort, {{p.query_key_value, 0, 0, g.convolution_dim()}});
        store.register_base(p.query_key_value.qdata, kGdnZPort, {{p.z, 0, 0, g.value_dim()}});
    } else if constexpr (requires { p.query_key; p.value_z; }) {
        store.register_base(p.query_key.qdata, kGdnInputPort,
            {{p.query_key, 0, 0, p.query_key.n}, {p.value_z, 0, p.query_key.n, g.value_dim()}});
        store.register_base(p.query_key.qdata, kGdnZPort, {{p.value_z, g.value_dim(), 0, g.value_dim()}});
    } else {
        const auto& w = p.query_key_value_z;
        store.register_base(w.qdata, kGdnInputPort, {{w, 0, 0, g.convolution_dim()}});
        store.register_base(w.qdata, kGdnZPort, {{w, g.convolution_dim(), 0, g.value_dim()}});
    }
}
template<class P, class Geometry>
void bind_gdn_control_base(ops::LoraStore& store, const P& p, const Geometry& g) {
    if constexpr (requires { std::variant_size<P>::value; }) {
        std::visit([&](const auto& v) { bind_gdn_control_base(store, v, g); }, p);
    } else if constexpr (requires { p.control_projection; }) {
        bind_gdn_control_base(store, p.control_projection, g);
    } else if constexpr (requires { p.a_projection; }) {
        store.register_base(p.a_projection.qdata, kGdnAPort, {{p.a_projection, 0, 0, g.gdn_value_heads}});
        store.register_base(p.a_projection.qdata, kGdnBPort, {{p.b_projection, 0, 0, g.gdn_value_heads}});
    } else {
        store.register_base(p.a_b_projection.qdata, kGdnAPort, {{p.a_b_projection, 0, 0, g.gdn_value_heads}});
        store.register_base(p.a_b_projection.qdata, kGdnBPort, {{p.a_b_projection, g.gdn_value_heads, 0, g.gdn_value_heads}});
    }
}

template <class Linear, class Geometry>
void bind_lora_gdn(ops::LoraStore& store, int layer, const Linear& linear, const Geometry& g) {
    bind_lora_bias(store, layer, "linear_attn.dt_bias", linear.projection.dt_bias);
    bind_gdn_input_base(store, linear.projection, g);
    bind_gdn_control_base(store, linear.projection, g);
    const auto* input   = gdn_input_key(linear.projection).qdata;
    const auto* control = gdn_control_key(linear.projection).qdata;
    store.register_module(layer, "linear_attn.in_proj_qkv",
                          {input, kGdnInputPort, g.hidden, g.convolution_dim()});
    store.register_module(layer, "linear_attn.in_proj_z",
                          {input, kGdnZPort, g.hidden, g.value_dim()});
    store.register_module(layer, "linear_attn.in_proj_a",
                          {control, kGdnAPort, g.hidden, g.gdn_value_heads});
    store.register_module(layer, "linear_attn.in_proj_b",
                          {control, kGdnBPort, g.hidden, g.gdn_value_heads});
    store.register_module(layer, "linear_attn.out_proj",
                          {linear.output.qdata, kGdnOutputPort, g.value_dim(), g.hidden});
}

inline std::size_t lora_gdn_control_workspace(int heads, int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {heads, tokens});
    (void)layout.alloc(DType::BF16, {heads, tokens});
    return layout.peak_bytes(1);
}
} // namespace sinfer::family
