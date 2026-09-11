#include "family/vision_standalone.h"
#include "core/arena.h"
#include "core/device.h"
#include <fstream>
#include <iostream>
#include <vector>

// The Python reference test supplies checkpoint weights, patches and an independent
// Transformers result. This executable exercises the same tower used by serving.
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << "Gemma checkpoint fixture not supplied\n";
        return 77;
    }
    try {
        sinfer::family::StandaloneVisionTower tower(argv[1], 0);
        const sinfer::family::VisionGrid grid{1, std::stoi(argv[3]), std::stoi(argv[4])};
        const auto& g = tower.geometry();
        std::vector<std::uint16_t> patches(std::size_t(grid.height) * grid.width * g.patch_dim);
        std::ifstream input(argv[2], std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(patches.data()), patches.size() * 2)) {
            throw std::runtime_error("short patch fixture");
        }
        sinfer::DeviceBuffer buffer(tower.output_bytes(grid));
        sinfer::Tensor out(buffer.p,
                           sinfer::DType::BF16,
                           {g.output_hidden, static_cast<int>(tower.merged_tokens(grid)), 1 + g.deepstack_layers});
        tower.encode(patches, grid, sinfer::family::PromptModality::Image, out);
        std::vector<std::uint16_t> values(out.bytes() / 2);
        buffer.copy_to_host(values.data(), out.bytes());
        std::ofstream output(argv[5], std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()), out.bytes());
        std::cout << "encoded " << grid.height << "x" << grid.width << " into " << out.ne[1] << " tokens\n";
        // Optional second pass exercises workspace reuse on the same encoder.
        if (std::stoi(argv[6])) {
            tower.encode(patches, grid, sinfer::family::PromptModality::Image, out);
            std::vector<std::uint16_t> replay(values.size());
            buffer.copy_to_host(replay.data(), out.bytes());
            if (values != replay) {
                throw std::runtime_error("vision replay changed values");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
