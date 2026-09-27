#include "runtime/generation/readout_policy.h"
#include "runtime/contract/types.h"

#include <cassert>
#include <iostream>

using sinfer::SpeculativeBackend;
using sinfer::runtime::can_pack_prefill_only;
using sinfer::runtime::target_only_readout;

template <class Options>
void decision_phases() {
    Options readout;
    readout.requested_output_tokens = 1;
    readout.next_token_candidates = {17, 23};
    assert(target_only_readout(SpeculativeBackend::DFlash, readout));
    // The policy must not change ordinary or MTP request planning.
    assert(!target_only_readout(SpeculativeBackend::None, readout));
    assert(!target_only_readout(SpeculativeBackend::Mtp, readout));

    // Thinking copies the decision options but requests a generated sequence.
    auto thought = readout;
    thought.requested_output_tokens = 512;
    thought.next_token_candidates.clear();
    assert(!target_only_readout(SpeculativeBackend::DFlash, thought));
    thought.next_token_candidates = {17, 23};
    assert(!target_only_readout(SpeculativeBackend::DFlash, thought));
    assert(target_only_readout(SpeculativeBackend::DFlash, readout)); // final scoring

    // A one-token chat is not a candidate-logit readout; retain its sampling path.
    Options chat;
    chat.requested_output_tokens = 1;
    assert(!target_only_readout(SpeculativeBackend::DFlash, chat));

    // Shared-prefix warmup has no candidates, but has always bypassed the drafter.
    chat.save_gpu_prefix = std::make_shared<sinfer::GpuPrefixKey>();
    for (const auto backend : {SpeculativeBackend::None, SpeculativeBackend::Mtp,
                              SpeculativeBackend::DFlash}) {
        assert(target_only_readout(backend, chat));
    }
    chat.gpu_prefix = chat.save_gpu_prefix;
    chat.save_gpu_prefix.reset();
    assert(target_only_readout(SpeculativeBackend::DFlash, chat));
}

int main() {
    // Admission uses public options and planning uses resolved options. They must
    // assign every phase to the same kind of batch.
    decision_phases<sinfer::ExecutionOptions>();
    decision_phases<sinfer::runtime::ResolvedExecutionOptions>();

    // Ordinary decision packs work without enabling DFlash's generating-prefill opt-in.
    assert(can_pack_prefill_only(SpeculativeBackend::DFlash, true, false, false, false));
    assert(!can_pack_prefill_only(SpeculativeBackend::DFlash, false, false, false, false));
    assert(can_pack_prefill_only(SpeculativeBackend::DFlash, false, true, false, false));

    // Under mixed load a thought gets its decode round, then decisions get one pack.
    // Continuous decision arrivals must not take successive packs ahead of the thought.
    assert(!can_pack_prefill_only(SpeculativeBackend::DFlash, true, false, true, false));
    assert(can_pack_prefill_only(SpeculativeBackend::DFlash, true, false, true, true));
    assert(!can_pack_prefill_only(SpeculativeBackend::DFlash, true, false, true, false));
    assert(!can_pack_prefill_only(SpeculativeBackend::DFlash, false, true, true, true));

    // Leave existing ordinary mixed-round and MTP scheduling policies unchanged.
    assert(can_pack_prefill_only(SpeculativeBackend::None, false, false, false, false));
    assert(!can_pack_prefill_only(SpeculativeBackend::None, false, false, true, true));
    assert(!can_pack_prefill_only(SpeculativeBackend::Mtp, true, true, false, false));
    std::cout << "Decision phases and fair DFlash readout packing passed\n";
}
