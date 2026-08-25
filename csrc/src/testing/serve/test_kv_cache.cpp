#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace {

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

int expect_page_ids(std::span<const std::int32_t> actual,
                    std::initializer_list<std::int32_t> expected, const char* label) {
    if (actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) {
        return 0;
    }
    std::cerr << label << " page IDs differ\n";
    return 1;
}

int expect_device_page_ids(const ninfer::Tensor& row, std::initializer_list<std::int32_t> expected,
                           const char* label) {
    std::vector<std::int32_t> actual(expected.size());
    const cudaError_t err = cudaMemcpy(
        actual.data(), row.data, actual.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    return expect_page_ids(actual, expected, label);
}

int expect_zeroed_pages(ninfer::PagedKVPool& pool, ninfer::PagedKVPlaneOrder order,
                        std::span<const std::int32_t> pages, cudaStream_t stream,
                        const char* label) {
    const ninfer::Tensor& plane = pool.plane(0);
    cudaError_t err             = cudaMemsetAsync(plane.data, 0x5a, plane.bytes(), stream);
    if (err != cudaSuccess) {
        std::cerr << label << " setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    pool.zero_pages(pages, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " synchronization failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::vector<unsigned char> actual(plane.bytes());
    err = cudaMemcpy(actual.data(), plane.data, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    std::vector<unsigned char> expected(actual.size(), 0x5a);
    for (const std::int32_t page : pages) {
        if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
            const std::size_t begin = static_cast<std::size_t>(page * plane.nb[3]);
            std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                      expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[3]), 0);
        } else {
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::size_t begin =
                    static_cast<std::size_t>(head * plane.nb[3] + page * plane.nb[2]);
                std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                          expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[2]), 0);
            }
        }
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " cleared bytes outside the selected physical pages\n";
    return 1;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto paged_plan = plan_paged_cache(10, 10, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);
    failures += expect_size(paged_pool.plane_count(), 4, "paged plane count");
    failures += check_shape(paged_pool.plane(0), {64, 64, 2, 10}, "paged code plane");
    failures += check_shape(paged_pool.plane(2), {1, 64, 2, 10}, "paged scale plane");
    failures += check_shape(paged_pool.block_table_row(0), {10, 1, 1, 1}, "paged block-table row");
    if (!paged_pool.plane(0).is_contiguous() || !paged_pool.plane(2).is_contiguous() ||
        !paged_pool.block_table_row(0).is_contiguous()) {
        ++failures;
        std::cerr << "Paged planes or block-table row are not contiguous\n";
    }
    const std::size_t expected_payload =
        2 * ninfer::Tensor(nullptr, ninfer::DType::I8, {64, 64, 2, 10}).bytes() +
        2 * ninfer::Tensor(nullptr, ninfer::DType::FP16, {1, 64, 2, 10}).bytes();
    failures +=
        expect_size(paged_plan.layout.payload_bytes(), expected_payload, "paged payload bytes");
    failures += expect_size(paged_plan.layout.metadata_bytes(), 10 * 2 * sizeof(std::int32_t),
                            "paged metadata bytes");
    const std::int32_t selected_pages[] = {1, 2, 6};
    failures += expect_zeroed_pages(paged_pool, ninfer::PagedKVPlaneOrder::PageMajor,
                                    selected_pages, ctx.stream, "page-major selective zero");

    auto head_major_plan = plan_paged_cache(10, 10, 1, {{ninfer::DType::BF16, 128, 8}},
                                            ninfer::PagedKVPlaneOrder::HeadMajor);
    ninfer::DeviceArena head_major_arena(head_major_plan.bytes);
    ninfer::PagedKVPool head_major_pool({head_major_arena.base(), head_major_arena.capacity()},
                                        head_major_plan.layout);
    failures += check_shape(head_major_pool.plane(0), {128, 64, 10, 8}, "head-major paged plane");
    failures += expect_zeroed_pages(head_major_pool, ninfer::PagedKVPlaneOrder::HeadMajor,
                                    selected_pages, ctx.stream, "head-major selective zero");

    auto allocation_a = paged_pool.reserve(3);
    auto allocation_b = paged_pool.reserve(3);
    allocation_a.materialize_pages(3);
    allocation_b.materialize_pages(3);
    allocation_a.bind_row(0);
    allocation_b.bind_row(1);
    failures += expect_device_page_ids(allocation_a.block_table(), {0, 1, 2}, "allocation A");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "allocation B");
    allocation_a.release();

    auto allocation_c = paged_pool.reserve(6);
    allocation_c.materialize_pages(6);
    allocation_c.bind_row(0);
    failures +=
        expect_page_ids(allocation_c.page_ids(), {0, 1, 2, 6, 7, 8}, "fragmented allocation C");
    failures +=
        expect_device_page_ids(allocation_c.block_table(), {0, 1, 2, 6, 7, 8}, "fragmented row C");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "isolated row B");

    allocation_c.trim_pages(2);
    if (paged_pool.can_reserve(2)) {
        ++failures;
        std::cerr << "Unmapped entitlement was exposed as reservable capacity\n";
    }
    allocation_c.cancel_unmapped_entitlement();
    if (!paged_pool.can_reserve(5)) {
        ++failures;
        std::cerr << "Cancelled entitlement did not return reservable capacity\n";
    }

    allocation_c.unbind_row();
    const std::int32_t retained_prefix[] = {allocation_c.page_ids()[0], allocation_c.page_ids()[1]};
    const ninfer::PagedKVResize claim[]  = {
        {.allocation = &allocation_c, .mapped_pages = 2, .page_entitlement = 5}};
    ninfer::resize_paged_kv_bundle(claim);
    allocation_c.bind_row(0);
    allocation_c.materialize_pages(5);
    if (allocation_c.page_ids()[0] != retained_prefix[0] ||
        allocation_c.page_ids()[1] != retained_prefix[1]) {
        ++failures;
        std::cerr << "Retained allocation moved its prefix pages\n";
    }
    allocation_c.trim_tokens(65);
    failures += expect_size(allocation_c.mapped_page_count(), 2, "partial-tail mapped pages");
    allocation_c.trim_tokens(64);
    failures += expect_size(allocation_c.mapped_page_count(), 1, "page-aligned mapped pages");

    auto main_plan    = plan_paged_cache(4, 4, 1, {{ninfer::DType::BF16, 16, 1}});
    auto backend_plan = plan_paged_cache(2, 2, 1, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena main_arena(main_plan.bytes);
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool main_pool({main_arena.base(), main_arena.capacity()}, main_plan.layout);
    ninfer::PagedKVPool backend_pool({backend_arena.base(), backend_arena.capacity()},
                                     backend_plan.layout);
    const ninfer::PagedKVReservation impossible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 3},
        {.pool = &backend_pool, .page_entitlement = 3},
    };
    try {
        auto unused = ninfer::reserve_paged_kv_bundle(impossible_bundle);
        (void)unused;
        ++failures;
        std::cerr << "Impossible multi-pool reservation succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(main_pool.entitled_pages(), 0, "failed bundle main entitlement");
    failures += expect_size(backend_pool.entitled_pages(), 0, "failed bundle backend entitlement");

    const ninfer::PagedKVReservation possible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 2},
        {.pool = &backend_pool, .page_entitlement = 2},
    };
    auto bundle = ninfer::reserve_paged_kv_bundle(possible_bundle);
    bundle[0].materialize_pages(2);
    bundle[1].materialize_pages(2);
    const ninfer::PagedKVResize impossible_resize[] = {
        {.allocation = &bundle[0], .mapped_pages = 1, .page_entitlement = 1},
        {.allocation = &bundle[1], .mapped_pages = 1, .page_entitlement = 3},
    };
    try {
        ninfer::resize_paged_kv_bundle(impossible_resize);
        ++failures;
        std::cerr << "Impossible multi-pool resize succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(bundle[0].mapped_page_count(), 2, "failed resize main mapping");
    failures += expect_size(bundle[0].page_entitlement(), 2, "failed resize main entitlement");
    failures += expect_size(bundle[1].mapped_page_count(), 2, "failed resize backend mapping");
    failures += expect_size(bundle[1].page_entitlement(), 2, "failed resize backend entitlement");

    return failures == 0 ? 0 : fail("kv cache test failed");
}
