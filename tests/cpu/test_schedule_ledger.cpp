#include <algorithm>
#include <iostream>
#include <string>

#include "cross3dgs/schedule_ledger.h"

namespace {

cross3dgs::ScheduleDeviceEvidence MakeDevice(uint64_t range) {
    cross3dgs::ScheduleDeviceEvidence device;
    device.caps.vendor_id = 0x10de;
    device.caps.device_id = 1;
    device.caps.subgroup_size = 32;
    device.caps.max_shared_memory = 65536;
    device.caps.max_storage_buffer_bindings = 32;
    device.caps.float32_atomic_add = true;
    device.caps.dispatch_indirect = true;
    device.max_storage_buffer_range = range;
    device.evidence_key = "unit_test";
    return device;
}

cross3dgs::ScheduleDeviceEvidence MakeAdreno(uint64_t range) {
    auto device = MakeDevice(range);
    device.caps.vendor_id = 0x5143;
    device.caps.subgroup_size = 32;
    device.caps.float32_atomic_add = false;
    return device;
}

// The capability vector expected for MoltenVK on Apple Silicon: SIMD
// width 32, 32 KiB of threadgroup memory, Metal's 31 buffer binding slots, no
// VK_EXT_shader_atomic_float, indirect dispatch available. The numbers come
// from published MoltenVK and Metal limits.
cross3dgs::ScheduleDeviceEvidence MakeApple() {
    cross3dgs::ScheduleDeviceEvidence device;
    device.caps.vendor_id = 0x106b;
    device.caps.device_id = 0;
    device.caps.subgroup_size = 32;
    device.caps.max_shared_memory = 32768;
    device.caps.max_storage_buffer_bindings = 31;
    device.caps.float32_atomic_add = false;
    device.caps.dispatch_indirect = true;
    device.max_storage_buffer_range = 4294967295ull;
    device.evidence_key = "apple_moltenvk_caps";
    return device;
}

// bicycle images_4 with a fixed 54,275-Gaussian workload.
cross3dgs::ScheduleLedgerWorkload MakeFixedCommonWorkload() {
    return {.gaussian_capacity = 54275,
            .width = 1237,
            .height = 822,
            .resident_view_capacity = 169,
            .sh_degree = 0,
            .byte_targets = true,
            .enable_optimizer = true,
            .enable_igs = false};
}

cross3dgs::ScheduleLedgerWorkload MakeWorkload() {
    return {.gaussian_capacity = 1000000,
            .width = 4096,
            .height = 4096,
            .resident_view_capacity = 5,
            .sh_degree = 3,
            .byte_targets = true,
            .enable_optimizer = true,
            .enable_igs = false};
}

}  // namespace

int main() {
    const auto unknown = cross3dgs::EnumerateScheduleCandidates(
            MakeDevice(0), MakeWorkload());
    if (unknown.candidates.size() != 256 ||
        !unknown.policy_selected_candidate_id.empty() ||
        std::any_of(unknown.candidates.begin(), unknown.candidates.end(),
                    [](const auto& candidate) {
                        return candidate.accepted;
                    })) {
        std::cerr << "unknown descriptor limits must reject all candidates\n";
        return 1;
    }

    const auto bounded = cross3dgs::EnumerateScheduleCandidates(
            MakeDevice(256ull * 1024ull * 1024ull), MakeWorkload());
    if (bounded.policy_selected_candidate_id.empty() ||
        bounded.policy_selected_candidate_id.find(".streamed.") ==
                std::string::npos) {
        std::cerr << "out-of-range resident targets must select streamed\n";
        return 1;
    }
    for (const auto& candidate : bounded.candidates) {
        if (candidate.accepted &&
            candidate.fields.targets == "resident") {
            std::cerr << "resident candidate passed advertised range limit\n";
            return 1;
        }
        if (candidate.accepted &&
            std::any_of(candidate.constraints.begin(),
                        candidate.constraints.end(),
                        [](const auto& constraint) {
                            return constraint.result != "pass";
                        })) {
            std::cerr << "accepted candidate contains non-pass constraint\n";
            return 1;
        }
    }
    auto single_view = MakeWorkload();
    single_view.resident_view_capacity = 0;
    const auto adreno_single =
            cross3dgs::EnumerateScheduleCandidates(
                    MakeAdreno(256ull * 1024ull * 1024ull), single_view);
    if (adreno_single.policy_selected_candidate_id.find(".resident.") ==
            std::string::npos) {
        std::cerr << "Adreno single-view probe must admit resident target\n";
        return 1;
    }
    const auto adreno_multi = cross3dgs::EnumerateScheduleCandidates(
            MakeAdreno(256ull * 1024ull * 1024ull), MakeWorkload());
    if (adreno_multi.policy_selected_candidate_id.find(".streamed.") ==
            std::string::npos) {
        std::cerr << "Adreno multi-view training must select streamed target\n";
        return 1;
    }
    const auto adreno_per_splat_cas = std::find_if(
            adreno_multi.candidates.begin(), adreno_multi.candidates.end(),
            [](const auto& candidate) {
                return candidate.fields.work_assignment == "per_splat" &&
                       candidate.fields.global_merge == "u32_cas" &&
                       candidate.fields.submission == "per_pass_fenced" &&
                       candidate.fields.targets == "streamed" &&
                       candidate.fields.radix == "subgroup_independent" &&
                       candidate.fields.projection_adam == "split" &&
                       candidate.fields.pair_dispatch == "indirect";
            });
    if (adreno_per_splat_cas == adreno_multi.candidates.end() ||
        !adreno_per_splat_cas->accepted) {
        std::cerr << "Adreno PerSplat CAS schedule must be admitted\n";
        return 1;
    }
    const auto adreno_pixel_cas = std::find_if(
            adreno_multi.candidates.begin(), adreno_multi.candidates.end(),
            [](const auto& candidate) {
                return candidate.fields.work_assignment ==
                               "pixel_stream_subgroup" &&
                       candidate.fields.global_merge == "u32_cas";
            });
    if (adreno_pixel_cas == adreno_multi.candidates.end() ||
        adreno_pixel_cas->accepted) {
        std::cerr << "Adreno pixel/subgroup CAS must remain rejected\n";
        return 1;
    }
    // Adreno740 at one million Gaussians: 192 MB of SH-rest exceeds the
    // 128 MiB range, so the storage is split and only the split
    // projection+Adam passes can bind it. The policy must select split
    // itself rather than leave it to a runtime override.
    const auto adreno_1m = cross3dgs::EnumerateScheduleCandidates(
            MakeAdreno(128ull * 1024ull * 1024ull),
            {.gaussian_capacity = 1000000,
             .width = 1237,
             .height = 822,
             .resident_view_capacity = 169,
             .sh_degree = 3,
             .byte_targets = true,
             .enable_optimizer = true,
             .enable_igs = false,
             .enable_gradient_accumulator = true});
    const auto adreno_1m_accepted = std::count_if(
            adreno_1m.candidates.begin(), adreno_1m.candidates.end(),
            [](const auto& candidate) { return candidate.accepted; });
    const auto adreno_1m_fused = std::count_if(
            adreno_1m.candidates.begin(), adreno_1m.candidates.end(),
            [](const auto& candidate) {
                return candidate.accepted &&
                       candidate.fields.projection_adam == "fused";
            });
    if (adreno_1m_accepted != 6 || adreno_1m_fused != 0 ||
        adreno_1m.policy_selected_candidate_id !=
                "per_splat.u32_cas.none.per_pass_fenced.streamed."
                "subgroup_independent.split.indirect") {
        std::cerr << "Adreno at 1M must admit only the six split "
                     "schedules and select per-splat CAS split, got "
                  << adreno_1m_accepted << " accepted, " << adreno_1m_fused
                  << " fused, " << adreno_1m.policy_selected_candidate_id
                  << "\n";
        return 1;
    }
    // One SH-rest binding (4 GiB range): fused stays feasible.
    const auto fused_ok = cross3dgs::EnumerateScheduleCandidates(
            MakeDevice(4294967295ull), MakeWorkload());
    if (fused_ok.policy_selected_candidate_id.find(".fused.") ==
        std::string::npos) {
        std::cerr << "unsplit SH-rest storage must keep the fused schedule\n";
        return 1;
    }
    const auto adreno_128 = cross3dgs::EnumerateScheduleCandidates(
            MakeAdreno(128ull * 1024ull * 1024ull), MakeWorkload());
    if (adreno_128.policy_selected_candidate_id.empty()) {
        std::cerr << "Adreno 128 MiB must admit fixed 6+6 SH-rest shards\n";
        return 1;
    }
    auto too_large = MakeWorkload();
    too_large.gaussian_capacity = 1400000;
    const auto adreno_128_reject =
            cross3dgs::EnumerateScheduleCandidates(
                    MakeAdreno(128ull * 1024ull * 1024ull), too_large);
    if (!adreno_128_reject.policy_selected_candidate_id.empty()) {
        std::cerr << "a 6+6 shard above 128 MiB must be rejected\n";
        return 1;
    }

    auto gradient_workload = MakeWorkload();
    gradient_workload.enable_gradient_accumulator = true;
    auto binding_25 = MakeDevice(512ull * 1024ull * 1024ull);
    binding_25.caps.max_storage_buffer_bindings = 25;
    const auto gradient_25 = cross3dgs::EnumerateScheduleCandidates(
            binding_25, gradient_workload);
    if (std::any_of(gradient_25.candidates.begin(), gradient_25.candidates.end(),
                    [](const auto& candidate) {
                        return candidate.accepted &&
                               candidate.fields.projection_adam == "fused";
                    })) {
        std::cerr << "gradient accumulation must reject a 25-binding fused "
                     "projection\n";
        return 1;
    }
    auto binding_26 = binding_25;
    binding_26.caps.max_storage_buffer_bindings = 26;
    const auto gradient_26 = cross3dgs::EnumerateScheduleCandidates(
            binding_26, gradient_workload);
    if (std::none_of(gradient_26.candidates.begin(), gradient_26.candidates.end(),
                     [](const auto& candidate) {
                         return candidate.accepted &&
                                candidate.fields.projection_adam == "fused";
                     })) {
        std::cerr << "gradient accumulation must admit a 26-binding fused "
                     "projection\n";
        return 1;
    }

    // Apple / MoltenVK: the schedule the selector picks for these
    // capabilities. If the selector changes, this test fails rather than the
    // selection silently drifting.
    const std::string kApplePrimarySchedule =
            "pixel_stream_subgroup.u32_cas.none.batched_8.resident."
            "subgroup_dependent.fused.indirect";
    const auto apple = cross3dgs::EnumerateScheduleCandidates(
            MakeApple(), MakeFixedCommonWorkload());
    if (apple.policy_selected_candidate_id != kApplePrimarySchedule) {
        std::cerr << "Apple schedule changed: expected "
                  << kApplePrimarySchedule << " got "
                  << apple.policy_selected_candidate_id << "\n";
        return 1;
    }

    // The held-out claim is that the capability vector alone decides the
    // schedule, so the same caps under a vendor id the selector has never seen
    // must give the same answer. If this ever fails, someone added a
    // vendor-conditional rule and the Apple result stops being held out.
    auto unknown_vendor = MakeApple();
    unknown_vendor.caps.vendor_id = 0x0bad;
    unknown_vendor.evidence_key = "vendor_independence_probe";
    const auto neutral = cross3dgs::EnumerateScheduleCandidates(
            unknown_vendor, MakeFixedCommonWorkload());
    if (neutral.policy_selected_candidate_id != kApplePrimarySchedule) {
        std::cerr << "Apple caps became vendor-dependent: " << kApplePrimarySchedule
                  << " vs " << neutral.policy_selected_candidate_id << "\n";
        return 1;
    }

    // AMD alone ranks gradient methods by its own order, so the
    // RX 9070 XT capability vector selects per-splat + CAS under AMD's vendor
    // id. Under any other vendor the common order applies; since the tiled
    // rule admits 64-lane subgroups, that is tiled + native atomic.
    auto amd = MakeApple();
    amd.caps.vendor_id = 0x1002;
    amd.caps.subgroup_size = 64;
    amd.caps.max_shared_memory = 65536;
    amd.caps.float32_atomic_add = true;
    amd.evidence_key = "rx9070xt_caps";
    const auto amd_ledger = cross3dgs::EnumerateScheduleCandidates(
            amd, MakeFixedCommonWorkload());
    if (amd_ledger.policy_selected_candidate_id.rfind("per_splat.u32_cas.", 0) !=
        0) {
        std::cerr << "AMD must select per-splat + CAS: "
                  << amd_ledger.policy_selected_candidate_id << "\n";
        return 1;
    }
    auto amd_caps_other_vendor = amd;
    amd_caps_other_vendor.caps.vendor_id = 0x10de;
    if (cross3dgs::EnumerateScheduleCandidates(
                amd_caps_other_vendor, MakeFixedCommonWorkload())
                .policy_selected_candidate_id.rfind(
                        "tiled_8x8.native_atomic.", 0) != 0) {
        std::cerr << "the AMD order leaked to another vendor\n";
        return 1;
    }

    // Intel ranks gradient methods by its own order, so the UHD 730
    // capability vector (64 KiB shared, no float32 atomic add) selects
    // pixel/subgroup + CAS under Intel's vendor id and tiled + CAS under any
    // other. The native-atomic families go first in the Intel order, so Intel
    // caps with float atomics select what the common order selects.
    auto intel = MakeApple();
    intel.caps.vendor_id = 0x8086;
    intel.caps.subgroup_size = 32;
    intel.caps.max_shared_memory = 65536;
    intel.caps.float32_atomic_add = false;
    intel.evidence_key = "uhd730_caps";
    if (cross3dgs::EnumerateScheduleCandidates(intel, MakeFixedCommonWorkload())
                .policy_selected_candidate_id.rfind(
                        "pixel_stream_subgroup.u32_cas.", 0) != 0) {
        std::cerr << "Intel UHD 730 caps must select pixel/subgroup + CAS\n";
        return 1;
    }
    auto intel_caps_other_vendor = intel;
    intel_caps_other_vendor.caps.vendor_id = 0x10de;
    if (cross3dgs::EnumerateScheduleCandidates(
                intel_caps_other_vendor, MakeFixedCommonWorkload())
                .policy_selected_candidate_id.rfind("tiled_8x8.u32_cas.",
                                                    0) != 0) {
        std::cerr << "the Intel order leaked to another vendor\n";
        return 1;
    }
    auto iris_xe = intel;
    iris_xe.caps.max_shared_memory = 32768;
    if (cross3dgs::EnumerateScheduleCandidates(iris_xe,
                                               MakeFixedCommonWorkload())
                .policy_selected_candidate_id.rfind(
                        "pixel_stream_subgroup.u32_cas.", 0) != 0) {
        std::cerr << "Iris Xe caps must still select pixel/subgroup + CAS\n";
        return 1;
    }
    auto intel_native = intel;
    intel_native.caps.float32_atomic_add = true;
    if (cross3dgs::EnumerateScheduleCandidates(intel_native,
                                               MakeFixedCommonWorkload())
                .policy_selected_candidate_id.rfind(
                        "tiled_8x8.native_atomic.", 0) != 0) {
        std::cerr << "native-atomic families must keep their "
                     "common-order place on Intel\n";
        return 1;
    }
    auto intel_native_32k = intel_native;
    intel_native_32k.caps.max_shared_memory = 32768;
    if (cross3dgs::EnumerateScheduleCandidates(intel_native_32k,
                                               MakeFixedCommonWorkload())
                .policy_selected_candidate_id.rfind(
                        "pixel_stream_subgroup.native_atomic.", 0) != 0) {
        std::cerr << "Intel caps with float atomics and 32 KiB shared memory "
                     "must select pixel/subgroup + native\n";
        return 1;
    }

    // Three capability boundaries near the Apple vector.
    auto apple_bind_24 = MakeApple();
    apple_bind_24.caps.max_storage_buffer_bindings = 24;
    if (cross3dgs::EnumerateScheduleCandidates(
                apple_bind_24, MakeFixedCommonWorkload())
                .policy_selected_candidate_id.find(".split.") ==
        std::string::npos) {
        std::cerr << "24 bindings must fall back to a split projection\n";
        return 1;
    }
    auto apple_shared_64k = MakeApple();
    apple_shared_64k.caps.max_shared_memory = 65536;
    if (cross3dgs::EnumerateScheduleCandidates(
                apple_shared_64k, MakeFixedCommonWorkload())
                .policy_selected_candidate_id.rfind("tiled_8x8.", 0) != 0) {
        std::cerr << "64 KiB of shared memory must admit the tiled backward\n";
        return 1;
    }
    auto apple_range_256 = MakeApple();
    apple_range_256.max_storage_buffer_range = 256ull * 1024ull * 1024ull;
    if (cross3dgs::EnumerateScheduleCandidates(
                apple_range_256, MakeFixedCommonWorkload())
                .policy_selected_candidate_id.find(".streamed.") ==
        std::string::npos) {
        std::cerr << "a 256 MiB storage range must stream the target atlas\n";
        return 1;
    }
    // Recorded runs are identified by (catalog_version, catalog_sha256), so
    // the catalog bytes must never change as a side effect. A deliberate
    // change updates this pin.
    if (cross3dgs::ScheduleSha256Hex(
                cross3dgs::ScheduleConstraintCatalogJson()) !=
        "b2c73066a00085f940d9661c0db74e1e4c2454280c3dda6be0c55f34a5db8fe8") {
        std::cerr << "constraint catalog bytes changed without a new version "
                     "or a recorded hash alias: "
                  << cross3dgs::ScheduleSha256Hex(
                             cross3dgs::ScheduleConstraintCatalogJson())
                  << std::endl;
        return 1;
    }
    return 0;
}
