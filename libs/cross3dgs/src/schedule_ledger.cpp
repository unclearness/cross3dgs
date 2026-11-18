// Catalog selector_source strings retain their original gaussian_forward.cpp
// locations to preserve serialized output and the catalog hash. Runtime
// selection/allocation now lives in gaussian/program_setup.cpp; pass
// registration lives in gaussian/*_passes.cpp.

#include "cross3dgs/schedule_ledger.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>

namespace cross3dgs {

namespace {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), self-contained so every run can persist the catalog
// and policy hashes it selected under.

constexpr std::array<uint32_t, 64> kSha256K = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
        0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
        0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
        0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
        0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
        0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

uint32_t Rotr(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

}  // namespace

std::string ScheduleSha256Hex(const std::string& bytes) {
    std::array<uint32_t, 8> state = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                     0xa54ff53a, 0x510e527f, 0x9b05688c,
                                     0x1f83d9ab, 0x5be0cd19};
    std::string padded = bytes;
    const uint64_t bit_length = uint64_t{bytes.size()} * 8u;
    padded.push_back('\x80');
    while (padded.size() % 64 != 56) padded.push_back('\0');
    for (int shift = 56; shift >= 0; shift -= 8) {
        padded.push_back(static_cast<char>((bit_length >> shift) & 0xffu));
    }
    for (size_t offset = 0; offset < padded.size(); offset += 64) {
        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; i++) {
            w[i] = (static_cast<uint8_t>(padded[offset + i * 4]) << 24) |
                   (static_cast<uint8_t>(padded[offset + i * 4 + 1]) << 16) |
                   (static_cast<uint8_t>(padded[offset + i * 4 + 2]) << 8) |
                   static_cast<uint8_t>(padded[offset + i * 4 + 3]);
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^
                                (w[i - 15] >> 3);
            const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^
                                (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, h] = state;
        for (int i = 0; i < 64; i++) {
            const uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + ch + kSha256K[i] + w[i];
            const uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
    std::ostringstream hex;
    hex << std::hex;
    for (const uint32_t word : state) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            hex << ((word >> shift) & 0xfu);
        }
    }
    return hex.str();
}

namespace {

// ---------------------------------------------------------------------------
// Candidate space

struct BackwardCandidate {
    const char* work_assignment;
    const char* global_merge;
    const char* tile_shape;
    const char* legacy;
};

constexpr std::array<BackwardCandidate, 8> kBackward = {{
        {"tiled_8x8", "native_atomic", "8x8", "tiled_scatter"},
        {"tiled_8x8", "u32_cas", "8x8", "tiled_scatter"},
        {"pixel_stream_subgroup", "native_atomic", "none",
         "pixel_subgroup_native"},
        {"pixel_stream_subgroup", "u32_cas", "none", "cas_scatter"},
        {"per_splat", "native_atomic", "none", "per_splat_native"},
        {"per_splat", "u32_cas", "none", "per_splat_cas"},
        {"pair_partial", "gaussian_major_fixed_reduce", "none",
         "exact_pair_gather"},
        {"per_splat", "gaussian_major_fixed_reduce", "none",
         "per_splat_pair_reduce"},
}};
// The vendor-keyed variants of the gradient-method order.
// AMD (0x1002) ranks kBackward entries in this order. Intel (0x8086) uses
// kBackwardOrderIntel. Every other vendor uses the kBackward order itself.
constexpr uint32_t kAmdVendorId = 0x1002;
constexpr std::array<size_t, 8> kBackwardOrderAmd = {
        5,  // per_splat + u32_cas
        6,  // pair_partial + gaussian_major_fixed_reduce
        4,  // per_splat + native_atomic
        7,  // per_splat + gaussian_major_fixed_reduce
        3,  // pixel_stream_subgroup + u32_cas
        2,  // pixel_stream_subgroup + native_atomic
        0,  // tiled_8x8 + native_atomic
        1,  // tiled_8x8 + u32_cas
};
// Intel (0x8086) order. The three native-atomic families go first in their
// kBackward order, so every Intel device with float32 atomic add selects what
// the common order selects.
constexpr uint32_t kIntelVendorId = 0x8086;
constexpr std::array<size_t, 8> kBackwardOrderIntel = {
        0,  // tiled_8x8 + native_atomic
        2,  // pixel_stream_subgroup + native_atomic
        4,  // per_splat + native_atomic
        3,  // pixel_stream_subgroup + u32_cas
        6,  // pair_partial + gaussian_major_fixed_reduce
        7,  // per_splat + gaussian_major_fixed_reduce
        5,  // per_splat + u32_cas
        1,  // tiled_8x8 + u32_cas
};
constexpr std::array<const char*, 2> kSubmission = {"batched_8",
                                                    "per_pass_fenced"};
constexpr std::array<const char*, 2> kTargets = {"resident", "streamed"};
constexpr std::array<const char*, 2> kRadix = {"subgroup_dependent",
                                               "subgroup_independent"};
constexpr std::array<const char*, 2> kProjectionAdam = {"fused", "split"};
constexpr std::array<const char*, 2> kPairDispatch = {"indirect", "direct"};

// SH-rest SoA layout: 12 vec4 = 192 bytes per Gaussian slot.
constexpr uint64_t kShRestBytesPerGaussian = 192;
// Fused projection+Adam descriptor bindings (26 with the IGS score buffer).
constexpr uint32_t kFusedBindings = 25;
constexpr uint32_t kFusedBindingsIgs = 26;
constexpr uint32_t kQualcommVendorId = 0x5143;

uint64_t RadixSharedBytes(uint32_t subgroup_size) {
    // radix_sort.cpp AddRadixSortU32: workgroup 256, partition 2048,
    // histogram capacity max(2048, 256 * subgroup_count) + 512 guard words.
    const uint32_t workgroup = 256;
    const uint32_t subgroup =
            subgroup_size == 0 ? 1u : std::min(subgroup_size, workgroup);
    const uint32_t subgroup_count = (workgroup + subgroup - 1) / subgroup;
    const uint32_t histogram =
            std::max(2048u, 256u * subgroup_count);
    return (uint64_t{histogram} + 512u) * 4u;
}

std::string HexId(uint32_t value) {
    std::ostringstream stream;
    stream << "0x" << std::hex << value;
    return stream.str();
}

struct ConstraintCheck {
    ScheduleConstraintResult result;
};

void AddPass(std::vector<ScheduleConstraintResult>& list, const char* id,
             const std::string& reason, const std::string& evidence) {
    list.push_back({id, "pass", reason, evidence});
}

void AddFail(std::vector<ScheduleConstraintResult>& list, const char* id,
             const std::string& reason, const std::string& evidence) {
    list.push_back({id, "fail", reason, evidence});
}

void AddUnknown(std::vector<ScheduleConstraintResult>& list, const char* id,
                const std::string& reason, const std::string& evidence) {
    list.push_back({id, "unknown", reason, evidence});
}

std::string JsonEscape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char character : text) {
        if (character == '"' || character == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    return escaped;
}

// An advertised descriptor range is a validity boundary. A driver completing
// an out-of-range binding is retained as behavioral evidence, not admitted as
// a valid schedule.
void EvaluateRangeConstraint(std::vector<ScheduleConstraintResult>& results,
                             const char* id, uint64_t required_bytes,
                             const ScheduleDeviceEvidence& device,
                             const std::string& caps_evidence) {
    if (device.max_storage_buffer_range == 0) {
        AddUnknown(results, id,
                   "requires " + std::to_string(required_bytes) +
                           " bytes; maxStorageBufferRange not provided",
                   caps_evidence);
        return;
    }
    if (required_bytes <= device.max_storage_buffer_range) {
        AddPass(results, id,
                std::to_string(required_bytes) + " bytes <= advertised "
                        "maxStorageBufferRange " +
                        std::to_string(device.max_storage_buffer_range),
                caps_evidence);
        return;
    }
    AddFail(results, id,
            std::to_string(required_bytes) +
                    " bytes exceed advertised maxStorageBufferRange " +
                    std::to_string(device.max_storage_buffer_range),
            device.caps.vendor_id == kQualcommVendorId
                    ? "failure_witness:adreno_descriptor_range_boundary"
                    : caps_evidence);
}

}  // namespace

std::string ScheduleCandidateId(const ScheduleCandidateFields& fields) {
    return fields.work_assignment + "." + fields.global_merge + "." +
           fields.tile_shape + "." + fields.submission + "." + fields.targets +
           "." + fields.radix + "." + fields.projection_adam + "." +
           fields.pair_dispatch;
}

ScheduleCandidateFields FieldsFromScheduleIR(const GaussianScheduleIR& ir) {
    ScheduleCandidateFields fields;
    fields.work_assignment = ir.backward_work_assignment;
    fields.global_merge = ir.backward_global_merge;
    fields.tile_shape = ir.backward_tile_shape;
    fields.backward = ir.tiled_backward ? "tiled_scatter"
                      : ir.atomic_free_exact_backward && ir.per_splat_backward
                              ? "per_splat_pair_reduce"
                      : ir.atomic_free_exact_backward ? "exact_pair_gather"
                      : ir.per_splat_backward
                              ? (ir.native_float_atomic_add
                                         ? "per_splat_native"
                                         : "per_splat_cas")
                      : ir.native_float_atomic_add ? "pixel_subgroup_native"
                                                   : "cas_scatter";
    fields.submission =
            ir.fenced_execution ? "per_pass_fenced" : "batched_8";
    fields.targets = ir.streamed_target ? "streamed" : "resident";
    fields.radix = ir.portable_radix_sort ? "subgroup_independent"
                                          : "subgroup_dependent";
    fields.projection_adam =
            ir.projection_optimizer_fusion ? "fused" : "split";
    fields.pair_dispatch =
            ir.indirect_pair_dispatch ? "indirect" : "direct";
    return fields;
}

ScheduleCandidateLedger EnumerateScheduleCandidates(
        const ScheduleDeviceEvidence& device,
        const ScheduleLedgerWorkload& workload) {
    const auto& caps = device.caps;
    const std::string caps_evidence = device.evidence_key.empty()
            ? std::string("device_caps")
            : device.evidence_key;
    ScheduleCandidateLedger ledger;
    ledger.policy_id = kSchedulePolicyId;
    ledger.catalog_version = kScheduleCatalogVersion;
    ledger.catalog_sha256 = ScheduleSha256Hex(ScheduleConstraintCatalogJson());
    ledger.policy_sha256 = ScheduleSha256Hex(SchedulePolicyJson());

    for (const auto& backward : kBackward)
    for (const char* submission : kSubmission)
    for (const char* targets : kTargets)
    for (const char* radix : kRadix)
    for (const char* projection_adam : kProjectionAdam)
    for (const char* pair_dispatch : kPairDispatch) {
        ScheduleCandidateRecord record;
        record.fields = {backward.work_assignment, backward.global_merge,
                         backward.tile_shape, backward.legacy,
                         submission, targets,
                         radix, projection_adam, pair_dispatch};
        record.candidate_id = ScheduleCandidateId(record.fields);
        auto& results = record.constraints;
        const bool is_tiled =
                record.fields.work_assignment == "tiled_8x8";
        const bool is_scatter = record.fields.global_merge == "native_atomic" ||
                record.fields.global_merge == "u32_cas";
        const bool is_native =
                record.fields.global_merge == "native_atomic";
        const bool is_batched = record.fields.submission == "batched_8";
        const bool is_resident = record.fields.targets == "resident";
        const bool is_subgroup_radix =
                record.fields.radix == "subgroup_dependent";
        const bool is_fused = record.fields.projection_adam == "fused";
        const bool is_indirect =
                record.fields.pair_dispatch == "indirect";

        if (is_tiled) {
            if (caps.subgroup_size >= kTiledBackwardMinSubgroupSize &&
                caps.max_shared_memory >= kTiledBackwardSharedBytes) {
                AddPass(results, "HC-SUBGROUP-TILED-01",
                        "subgroup_size " +
                                std::to_string(caps.subgroup_size) +
                                " >= 8 and shared memory " +
                                std::to_string(caps.max_shared_memory) +
                                " >= 49152",
                        caps_evidence);
            } else {
                AddFail(results, "HC-SUBGROUP-TILED-01",
                        "requires subgroup_size >= 8 (have " +
                                std::to_string(caps.subgroup_size) +
                                ") and shared memory >= 49152 (have " +
                                std::to_string(caps.max_shared_memory) + ")",
                        caps_evidence);
            }
        }
        if (is_native) {
            if (caps.float32_atomic_add) {
                AddPass(results, "HC-ATOMIC-NATIVE-01",
                        "native float32 atomic add capability present",
                        caps_evidence);
            } else {
                AddFail(results, "HC-ATOMIC-NATIVE-01",
                        "native float32 atomic add capability absent",
                        caps_evidence);
            }
        }
        if (is_scatter) {
            const bool is_per_splat_cas =
                    record.fields.work_assignment == "per_splat" &&
                    record.fields.global_merge == "u32_cas";
            if (caps.vendor_id == kQualcommVendorId &&
                !is_per_splat_cas) {
                AddFail(results, "HC-ATOMIC-SCATTER-QCOM-01",
                        "native, tiled, and pixel/subgroup scatter remain "
                        "implementation-tested unreliable on Qualcomm "
                        "0x5143",
                        "failure_witness:adreno_scatter_backward");
            } else {
                AddPass(results, "HC-ATOMIC-SCATTER-QCOM-01",
                        is_per_splat_cas &&
                                        caps.vendor_id == kQualcommVendorId
                                ? "per-splat u32 CAS is outside the retained "
                                  "pixel-oriented failure boundary"
                                : "vendor " + HexId(caps.vendor_id) +
                                          " has no scatter reject evidence",
                        is_per_splat_cas &&
                                        caps.vendor_id == kQualcommVendorId
                                ? "measured:adreno_per_splat_cas"
                                : caps_evidence);
            }
        }
        if (is_batched) {
            if (caps.vendor_id == kQualcommVendorId) {
                AddFail(results, "HC-SUBMISSION-BATCHED-QCOM-01",
                        "large multi-pass batched submit hangs on Qualcomm "
                        "0x5143 even with full barriers",
                        "failure_witness:adreno_batched_submit_hang");
            } else {
                AddPass(results, "HC-SUBMISSION-BATCHED-QCOM-01",
                        "vendor " + HexId(caps.vendor_id) +
                                " has no batched-submit reject evidence",
                        caps_evidence);
            }
        }
        if (is_resident) {
            if (caps.vendor_id == kQualcommVendorId &&
                workload.resident_view_capacity > 1) {
                AddFail(results, "HC-TARGET-RESIDENT-QCOM-01",
                        "repeated copies into the resident device-local "
                        "target SSBO fault the Qualcomm 0x5143 driver",
                        "failure_witness:adreno_resident_target_fault");
            } else {
                AddPass(results, "HC-TARGET-RESIDENT-QCOM-01",
                        workload.resident_view_capacity <= 1
                                ? "single-view target does not exercise the "
                                  "repeated multi-view resident-copy fault"
                                : "vendor " + HexId(caps.vendor_id) +
                                  " has no resident-target reject evidence",
                        caps_evidence);
            }
            const uint64_t target_bytes = uint64_t{workload.width} *
                    workload.height *
                    std::max(workload.resident_view_capacity, 1u) *
                    (workload.byte_targets ? 4u : 16u);
            EvaluateRangeConstraint(results,
                                    "HC-DESCRIPTOR-TARGET-RANGE-01",
                                    target_bytes, device, caps_evidence);
        } else {
            if (workload.enable_optimizer &&
                workload.resident_view_capacity > 0) {
                AddPass(results, "HC-TARGET-STREAMED-OPT-01",
                        "streamed target host-fill step path is available "
                        "for optimizer training sets",
                        caps_evidence);
            } else {
                AddFail(results, "HC-TARGET-STREAMED-OPT-01",
                        "streamed targets require optimizer training with "
                        "a resident view set",
                        caps_evidence);
            }
        }
        EvaluateRangeConstraint(
                results, "HC-DESCRIPTOR-SHREST-RANGE-01",
                uint64_t{workload.gaussian_capacity} *
                        (kShRestBytesPerGaussian / 2u),
                device, caps_evidence);
        if (is_fused) {
            const uint32_t required =
                    (workload.enable_igs ||
                     workload.enable_gradient_accumulator)
                            ? kFusedBindingsIgs
                            : kFusedBindings;
            if (caps.max_storage_buffer_bindings >= required) {
                AddPass(results, "HC-BINDING-FUSED-01",
                        std::to_string(caps.max_storage_buffer_bindings) +
                                " storage-buffer bindings >= " +
                                std::to_string(required),
                        caps_evidence);
            } else {
                AddFail(results, "HC-BINDING-FUSED-01",
                        "fused projection+Adam needs " +
                                std::to_string(required) +
                                " storage-buffer bindings (have " +
                                std::to_string(
                                        caps.max_storage_buffer_bindings) +
                                ")",
                        caps_evidence);
            }
            // SH-rest storage above the advertised range is kept as two
            // 6+6 shards (program_setup.cpp split_sh_rest_storage), and only
            // the split projection and Adam passes bind both shards.
            const uint64_t sh_rest_bytes =
                    uint64_t{workload.gaussian_capacity} *
                    kShRestBytesPerGaussian;
            if (device.max_storage_buffer_range == 0) {
                AddUnknown(results, "HC-BINDING-SHREST-SPLIT-01",
                           "requires " + std::to_string(sh_rest_bytes) +
                                   " SH-rest bytes in one binding; "
                                   "maxStorageBufferRange not provided",
                           caps_evidence);
            } else if (sh_rest_bytes <= device.max_storage_buffer_range) {
                AddPass(results, "HC-BINDING-SHREST-SPLIT-01",
                        std::to_string(sh_rest_bytes) +
                                " SH-rest bytes fit one binding of "
                                "maxStorageBufferRange " +
                                std::to_string(
                                        device.max_storage_buffer_range),
                        caps_evidence);
            } else {
                AddFail(results, "HC-BINDING-SHREST-SPLIT-01",
                        std::to_string(sh_rest_bytes) +
                                " SH-rest bytes exceed maxStorageBufferRange " +
                                std::to_string(
                                        device.max_storage_buffer_range) +
                                " and are stored in two shards, which only "
                                "the split projection+Adam passes bind",
                        caps_evidence);
            }
        }
        if (is_subgroup_radix) {
            if (caps.vendor_id == kQualcommVendorId) {
                AddFail(results, "HC-RADIX-SUBGROUP-QCOM-01",
                        "subgroup-dependent radix spine/downsweep corrupts "
                        "sorted keys on Qualcomm 0x5143",
                        "failure_witness:adreno_subgroup_radix_corruption");
            } else {
                AddPass(results, "HC-RADIX-SUBGROUP-QCOM-01",
                        "vendor " + HexId(caps.vendor_id) +
                                " has no subgroup-radix reject evidence",
                        caps_evidence);
            }
        }
        {
            const uint64_t shared_bytes =
                    RadixSharedBytes(caps.subgroup_size);
            if (shared_bytes <= caps.max_shared_memory) {
                AddPass(results, "HC-RADIX-SHARED-01",
                        "radix histogram layout " +
                                std::to_string(shared_bytes) +
                                " bytes <= shared memory " +
                                std::to_string(caps.max_shared_memory),
                        caps_evidence);
            } else {
                AddFail(results, "HC-RADIX-SHARED-01",
                        "radix histogram layout " +
                                std::to_string(shared_bytes) +
                                " bytes exceed shared memory " +
                                std::to_string(caps.max_shared_memory),
                        caps_evidence);
            }
        }
        if (is_indirect) {
            if (caps.dispatch_indirect) {
                AddPass(results, "HC-DISPATCH-INDIRECT-FEATURE-01",
                        "dispatch_indirect capability present",
                        caps_evidence);
            } else {
                AddFail(results, "HC-DISPATCH-INDIRECT-FEATURE-01",
                        "device does not expose indirect dispatch",
                        caps_evidence);
            }
        }

        record.accepted = std::all_of(
                results.begin(), results.end(),
                [](const ScheduleConstraintResult& check) {
                    return check.result == "pass";
                });
        ledger.candidates.push_back(std::move(record));
    }

    // Frozen policy: preserve the production selector while PerSplat remains
    // an evaluated alternative: tiled > pixel/subgroup > PerSplat > gather,
    // with native atomic preferred over CAS within the same assignment;
    // submission batched > fenced,
    // targets resident > streamed, radix subgroup > portable,
    // projection fused > split, dispatch indirect > direct.
    // AMD and Intel replace only the gradient-method order
    // (kBackwardOrderAmd, kBackwardOrderIntel).
    const std::array<size_t, 8>* vendor_backward_order =
            caps.vendor_id == kAmdVendorId     ? &kBackwardOrderAmd
            : caps.vendor_id == kIntelVendorId ? &kBackwardOrderIntel
                                               : nullptr;
    const auto preference_rank = [vendor_backward_order](
                                         const ScheduleCandidateFields& fields) {
        const auto index = [](const auto& options,
                              const std::string& value) {
            for (size_t position = 0; position < options.size();
                 position++) {
                if (value == options[position]) return position;
            }
            return options.size();
        };
        size_t backward_rank=kBackward.size();
        for(size_t position=0;position<kBackward.size();position++) {
            if(fields.work_assignment==kBackward[position].work_assignment &&
               fields.global_merge==kBackward[position].global_merge &&
               fields.tile_shape==kBackward[position].tile_shape) {
                backward_rank=position;
                break;
            }
        }
        if (vendor_backward_order != nullptr &&
            backward_rank < kBackward.size()) {
            for (size_t position = 0; position < vendor_backward_order->size();
                 position++) {
                if ((*vendor_backward_order)[position] == backward_rank) {
                    backward_rank = position;
                    break;
                }
            }
        }
        return std::array<size_t, 6>{
                backward_rank,
                index(kSubmission, fields.submission),
                index(kTargets, fields.targets),
                index(kRadix, fields.radix),
                index(kProjectionAdam, fields.projection_adam),
                index(kPairDispatch, fields.pair_dispatch)};
    };
    const ScheduleCandidateRecord* best = nullptr;
    for (const auto& candidate : ledger.candidates) {
        if (!candidate.accepted) continue;
        if (best == nullptr ||
            preference_rank(candidate.fields) <
                    preference_rank(best->fields)) {
            best = &candidate;
        }
    }
    if (best != nullptr) {
        ledger.policy_selected_candidate_id = best->candidate_id;
        for (auto& candidate : ledger.candidates) {
            candidate.policy_selected =
                    candidate.candidate_id == best->candidate_id;
        }
    }
    return ledger;
}

std::string ScheduleConstraintCatalogJson() {
    // Stable IDs and predicates. Selector sources are referenced so each
    // predicate can be audited against the implementation.
    // The selector_source strings are part of the hashed payload, so moving
    // a source file changes catalog_sha256 without changing any predicate.
    std::ostringstream json;
    json << "{\n"
         << "  \"catalog_version\": \"" << kScheduleCatalogVersion << "\",\n"
         << "  \"constraints\": [\n";
    const auto entry = [&json](const char* id, const char* clazz,
                               const char* predicate, const char* evidence,
                               const char* unsafe_candidates,
                               const char* fallback, const char* witness,
                               const char* source, bool last = false) {
        json << "    {\n"
             << "      \"constraint_id\": \"" << id << "\",\n"
             << "      \"class\": \"" << clazz << "\",\n"
             << "      \"predicate\": \"" << predicate << "\",\n"
             << "      \"required_evidence\": \"" << evidence << "\",\n"
             << "      \"unsafe_candidates\": \"" << unsafe_candidates
             << "\",\n"
             << "      \"safe_fallback\": \"" << fallback << "\",\n"
             << "      \"failure_witness_key\": \"" << witness << "\",\n"
             << "      \"selector_source\": \"" << source << "\"\n"
             << "    }" << (last ? "\n" : ",\n");
    };
    entry("HC-SUBGROUP-TILED-01", "HC-SUBGROUP",
          "work_assignment == tiled_8x8 requires subgroup_size >= 8 and "
          "max_shared_memory >= 49152",
          "capability dump (subgroup_size, max_shared_memory)",
          "tiled_scatter on subgroup < 8 or shared < 49152",
          "cas_scatter", "none_required_capability_gate",
          "libs/cross3dgs/src/gaussian_forward.cpp use_tiled_backward");
    entry("HC-ATOMIC-NATIVE-01", "HC-ATOMIC",
          "global_merge == native_atomic requires float32_atomic_add",
          "capability dump (float32_atomic_add)",
          "native_atomic without the advertised device feature", "u32_cas",
          "none_required_capability_gate",
          "libs/cross3dgs/src/gaussian_forward.cpp "
          "use_native_float_atomic_add");
    entry("HC-ATOMIC-SCATTER-QCOM-01", "HC-ATOMIC",
          "on vendor_id 0x5143, native atomic and tiled or pixel/subgroup "
          "u32-CAS scatter are rejected; per-splat u32 CAS is admitted for "
          "the current implementation",
          "vendor id + retained pixel-scatter failure boundary + per-splat "
          "CAS completion and update-conformance evidence",
          "native, pixel/subgroup, or tiled scatter on Qualcomm Adreno",
          "exact_pair_gather", "adreno_scatter_backward",
          "android/app/src/main/cpp/jni_bridge.cpp atomic_free_exact");
    entry("HC-SUBMISSION-BATCHED-QCOM-01", "HC-SUBMISSION",
          "submission == batched_8 requires vendor_id != 0x5143",
          "vendor id + bounded firstHangingPass(timeout) diagnostic",
          "batched multi-pass submit on Qualcomm Adreno",
          "per_pass_fenced", "adreno_batched_submit_hang",
          "libs/cross3dgs/src/gaussian_forward.cpp "
          "use_fenced_execution");
    entry("HC-TARGET-RESIDENT-QCOM-01", "HC-DESCRIPTOR",
          "multi-view targets == resident require vendor_id != 0x5143; "
          "single-view probes are admitted",
          "vendor id + recorded device-fault log for repeated target copies",
          "resident multi-view target SSBO on Qualcomm Adreno",
          "streamed", "adreno_resident_target_fault",
          "libs/cross3dgs/src/gaussian_forward.cpp "
          "stream_resident_target");
    entry("HC-DESCRIPTOR-TARGET-RANGE-01", "HC-DESCRIPTOR",
          "targets == resident: width*height*views*(byte_targets ? 4 : 16) "
          "bytes must not exceed advertised maxStorageBufferRange; a missing "
          "limit is unknown and therefore fail-closed",
          "queried maxStorageBufferRange + computed required bytes",
          "resident targets whose atlas exceeds the advertised descriptor "
          "range or whose limit is unavailable",
          "streamed", "adreno_descriptor_range_boundary",
          "libs/cross3dgs/src/gaussian_forward.cpp target buffer");
    entry("HC-DESCRIPTOR-SHREST-RANGE-01", "HC-DESCRIPTOR",
          "gaussian_capacity * 96 bytes (each fixed 6+6 SH-rest shard) over "
          "maxStorageBufferRange reject; a missing limit is unknown and "
          "therefore fail-closed",
          "queried maxStorageBufferRange + computed required bytes",
          "any schedule at a capacity whose SH-rest shard exceeds an "
          "enforced range",
          "reduce capacity (separate operating point)",
          "adreno_descriptor_range_boundary",
          "libs/cross3dgs/src/gaussian_forward.cpp sh_rest buffer");
    entry("HC-TARGET-STREAMED-OPT-01", "HC-DESCRIPTOR",
          "targets == streamed requires optimizer training with "
          "resident_view_capacity > 0 (host step-fill path)",
          "workload configuration",
          "streamed target outside optimizer training", "resident",
          "streamed_target_stale_view",
          "libs/cross3dgs/src/gaussian_forward.cpp "
          "use_host_step_fill");
    entry("HC-BINDING-FUSED-01", "HC-BINDING",
          "projection_adam == fused requires max_storage_buffer_bindings >= "
          "25 (26 with IGS)",
          "queried per-stage storage-buffer binding limit",
          "fused projection+Adam pipeline over the binding limit", "split",
          "none_required_capability_gate",
          "libs/cross3dgs/src/gaussian_forward.cpp "
          "fuse_projection_optimizer");
    entry("HC-BINDING-SHREST-SPLIT-01", "HC-BINDING",
          "projection_adam == fused requires gaussian_capacity * 192 bytes "
          "(the whole SH-rest buffer) <= maxStorageBufferRange; above it the "
          "SH-rest storage is two 6+6 shards that only the split passes "
          "bind; a missing limit is unknown and therefore fail-closed",
          "queried maxStorageBufferRange + computed required bytes",
          "fused projection+Adam over split SH-rest storage", "split",
          "none_required_capability_gate",
          "libs/cross3dgs/src/gaussian/program_setup.cpp "
          "split_sh_rest_storage");
    entry("HC-RADIX-SUBGROUP-QCOM-01", "HC-SUBGROUP",
          "radix == subgroup_dependent requires vendor_id != 0x5143",
          "vendor id + sorted-key integrity counters "
          "(debugPairIntegrity)",
          "subgroup-dependent radix spine/downsweep on Qualcomm Adreno",
          "subgroup_independent", "adreno_subgroup_radix_corruption",
          "libs/cross3dgs/src/radix_sort.cpp portable_radix");
    entry("HC-RADIX-SHARED-01", "HC-SUBGROUP",
          "(max(2048, 256*ceil(256/subgroup_size)) + 512) * 4 bytes <= "
          "max_shared_memory",
          "capability dump (subgroup_size, max_shared_memory)",
          "radix histogram layout over shared memory",
          "none (hard reject of the workload)",
          "none_required_capability_gate",
          "libs/cross3dgs/src/radix_sort.cpp shared_bytes check");
    entry("HC-DISPATCH-INDIRECT-FEATURE-01", "HC-DISPATCH",
          "pair_dispatch == indirect requires the dispatch_indirect "
          "capability",
          "capability dump (dispatch_indirect)",
          "indirect pair dispatch without device support", "direct",
          "none_required_capability_gate",
          "libs/cross3dgs/src/gaussian_forward.cpp indirect_pairs", true);
    json << "  ]\n}\n";
    return json.str();
}

std::string SchedulePolicyJson() {
    std::ostringstream json;
    json << "{\n"
         << "  \"policy_id\": \"" << kSchedulePolicyId << "\",\n"
         << "  \"rule\": \"among accepted candidates select the "
            "lexicographically most-preferred tuple, evaluating fields in "
            "the listed order\",\n"
         << "  \"inputs\": \"device capability vector and explicit driver "
            "evidence only; every vendor-conditional predicate carries a "
            "constraint ID in the catalog; the vendor-keyed preferences "
            "are backward_candidate_vendor_0x1002 and "
            "backward_candidate_vendor_0x8086, which replace the "
            "backward_candidate order for AMD and Intel\",\n"
         << "  \"field_order\": [\"backward_candidate\", \"submission\", \"targets\", "
            "\"radix\", \"projection_adam\", \"pair_dispatch\"],\n"
         << "  \"preferences\": {\n"
         << "    \"backward_candidate\": ["
            "\"tiled_8x8+native_atomic\", "
            "\"tiled_8x8+u32_cas\", "
            "\"pixel_stream_subgroup+native_atomic\", "
            "\"pixel_stream_subgroup+u32_cas\", "
            "\"per_splat+native_atomic\", \"per_splat+u32_cas\", "
            "\"pair_partial+gaussian_major_fixed_reduce\", "
            "\"per_splat+gaussian_major_fixed_reduce\"],\n"
         << "    \"backward_candidate_vendor_0x1002\": ["
            "\"per_splat+u32_cas\", "
            "\"pair_partial+gaussian_major_fixed_reduce\", "
            "\"per_splat+native_atomic\", "
            "\"per_splat+gaussian_major_fixed_reduce\", "
            "\"pixel_stream_subgroup+u32_cas\", "
            "\"pixel_stream_subgroup+native_atomic\", "
            "\"tiled_8x8+native_atomic\", \"tiled_8x8+u32_cas\"],\n"
         << "    \"backward_candidate_vendor_0x1002_source\": \"one-time "
            "offline screen on the RX 9070 XT development configuration at "
            "the trained one-million-Gaussian state\",\n"
         << "    \"backward_candidate_vendor_0x8086\": ["
            "\"tiled_8x8+native_atomic\", "
            "\"pixel_stream_subgroup+native_atomic\", "
            "\"per_splat+native_atomic\", "
            "\"pixel_stream_subgroup+u32_cas\", "
            "\"pair_partial+gaussian_major_fixed_reduce\", "
            "\"per_splat+gaussian_major_fixed_reduce\", "
            "\"per_splat+u32_cas\", \"tiled_8x8+u32_cas\"],\n"
         << "    \"backward_candidate_vendor_0x8086_source\": \"one-time "
            "offline screen on the UHD Graphics 730 development configuration "
            "at the trained one-million-Gaussian state; the three "
            "native-atomic entries were not measurable there and go first "
            "in their backward_candidate order\",\n"
         << "    \"submission\": [\"batched_8\", \"per_pass_fenced\"],\n"
         << "    \"targets\": [\"resident\", \"streamed\"],\n"
         << "    \"radix\": [\"subgroup_dependent\", "
            "\"subgroup_independent\"],\n"
         << "    \"projection_adam\": [\"fused\", \"split\"],\n"
         << "    \"pair_dispatch\": [\"indirect\", \"direct\"]\n"
         << "  }\n"
         << "}\n";
    return json.str();
}

std::string ScheduleCandidateSchemaJson() {
    std::ostringstream json;
    json << "{\n"
         << "  \"schema_id\": \"cross3dgs-schedule-candidate-v2\",\n"
         << "  \"candidate_fields\": {\n"
         << "    \"work_assignment\": [\"pixel_stream_subgroup\", "
            "\"per_splat\", \"tiled_8x8\", \"pair_partial\"],\n"
         << "    \"global_merge\": [\"native_atomic\", \"u32_cas\", "
            "\"gaussian_major_fixed_reduce\"],\n"
         << "    \"tile_shape\": [\"none\", \"8x8\"],\n"
         << "    \"backward\": \"legacy compatibility label\",\n"
         << "    \"submission\": [\"batched_8\", \"per_pass_fenced\"],\n"
         << "    \"targets\": [\"resident\", \"streamed\"],\n"
         << "    \"radix\": [\"subgroup_dependent\", "
            "\"subgroup_independent\"],\n"
         << "    \"projection_adam\": [\"fused\", \"split\"],\n"
         << "    \"pair_dispatch\": [\"indirect\", \"direct\"]\n"
         << "  },\n"
         << "  \"semantic_class\": \"every candidate field selects among "
            "semantics-preserving lowerings of one logical optimizer "
            "update; acceptance is validity-constrained per device\",\n"
         << "  \"quality_changing_operating_point\": [\"sh_degree\", "
            "\"density_strategy\", \"gaussian_capacity\", "
            "\"stochastic_backward_samples\"],\n"
         << "  \"notes\": \"byte_targets is bitwise exact for 8-bit "
            "sources and recorded in schedule_ir; Sobol sampled backward "
            "is quality-changing and outside the candidate tuple\",\n"
         << "  \"constraint_result_values\": [\"pass\", \"fail\", "
            "\"unknown\"],\n"
         << "  \"ledger_record\": {\n"
         << "    \"candidate_id\": \"string\",\n"
         << "    \"fields\": \"candidate_fields\",\n"
         << "    \"constraints\": [{\"constraint_id\": \"HC-...\", "
            "\"result\": \"pass|fail|unknown\", \"reason\": \"string\", "
            "\"evidence_key\": \"string\"}],\n"
         << "    \"accepted\": \"bool (every applicable constraint passes; unknown is rejected)\",\n"
         << "    \"policy_selected\": \"bool\"\n"
         << "  }\n"
         << "}\n";
    return json.str();
}

std::string ScheduleLedgerJson(const ScheduleCandidateLedger& ledger,
                               const ScheduleDeviceEvidence& device,
                               const ScheduleLedgerWorkload& workload,
                               const GaussianScheduleIR* ir) {
    std::ostringstream json;
    json << "{\n"
         << "  \"schema_id\": \"cross3dgs-candidate-ledger-v2\",\n"
         << "  \"catalog_version\": \"" << ledger.catalog_version << "\",\n"
         << "  \"catalog_sha256\": \"" << ledger.catalog_sha256 << "\",\n"
         << "  \"policy_id\": \"" << ledger.policy_id << "\",\n"
         << "  \"policy_sha256\": \"" << ledger.policy_sha256 << "\",\n"
         << "  \"device\": {\n"
         << "    \"vendor_id\": " << device.caps.vendor_id << ",\n"
         << "    \"device_id\": " << device.caps.device_id << ",\n"
         << "    \"subgroup_size\": " << device.caps.subgroup_size << ",\n"
         << "    \"max_shared_memory\": " << device.caps.max_shared_memory
         << ",\n"
         << "    \"max_storage_buffer_bindings\": "
         << device.caps.max_storage_buffer_bindings << ",\n"
         << "    \"float32_atomic_add\": "
         << (device.caps.float32_atomic_add ? "true" : "false") << ",\n"
         << "    \"dispatch_indirect\": "
         << (device.caps.dispatch_indirect ? "true" : "false") << ",\n"
         << "    \"f16_storage\": "
         << (device.caps.f16_storage ? "true" : "false") << ",\n"
         << "    \"max_storage_buffer_range\": "
         << device.max_storage_buffer_range << ",\n"
         << "    \"evidence_key\": \"" << JsonEscape(device.evidence_key)
         << "\"\n"
         << "  },\n"
         << "  \"workload\": {\n"
         << "    \"gaussian_capacity\": " << workload.gaussian_capacity
         << ",\n"
         << "    \"width\": " << workload.width << ",\n"
         << "    \"height\": " << workload.height << ",\n"
         << "    \"resident_view_capacity\": "
         << workload.resident_view_capacity << ",\n"
         << "    \"sh_degree\": " << workload.sh_degree << ",\n"
         << "    \"byte_targets\": "
         << (workload.byte_targets ? "true" : "false") << ",\n"
         << "    \"enable_optimizer\": "
         << (workload.enable_optimizer ? "true" : "false") << ",\n"
         << "    \"enable_igs\": "
         << (workload.enable_igs ? "true" : "false") << ",\n"
         << "    \"enable_gradient_accumulator\": "
         << (workload.enable_gradient_accumulator ? "true" : "false")
         << "\n"
         << "  },\n"
         << "  \"policy_selected_candidate_id\": \""
         << ledger.policy_selected_candidate_id << "\",\n";
    if (ir != nullptr) {
        const auto configured = FieldsFromScheduleIR(*ir);
        const std::string configured_id=ScheduleCandidateId(configured);
        const auto configured_record=std::find_if(
                ledger.candidates.begin(),ledger.candidates.end(),
                [&](const ScheduleCandidateRecord& candidate) {
                    return candidate.candidate_id==configured_id;
                });
        json << "  \"configured_candidate_id\": \""
             << configured_id << "\",\n"
             << "  \"configured_accepted\": "
             << (configured_record!=ledger.candidates.end() &&
                         configured_record->accepted ? "true" : "false")
             << ",\n"
             << "  \"configured_matches_policy\": "
             << (configured_id ==
                                 ledger.policy_selected_candidate_id
                         ? "true"
                         : "false")
             << ",\n";
    }
    size_t accepted_count = 0;
    for (const auto& candidate : ledger.candidates) {
        if (candidate.accepted) accepted_count++;
    }
    json << "  \"candidate_count\": " << ledger.candidates.size() << ",\n"
         << "  \"accepted_count\": " << accepted_count << ",\n"
         << "  \"candidates\": [\n";
    for (size_t index = 0; index < ledger.candidates.size(); index++) {
        const auto& candidate = ledger.candidates[index];
        json << "    {\n"
             << "      \"candidate_id\": \"" << candidate.candidate_id
             << "\",\n"
             << "      \"fields\": {\n"
             << "        \"work_assignment\": \""
             << candidate.fields.work_assignment << "\",\n"
             << "        \"global_merge\": \""
             << candidate.fields.global_merge << "\",\n"
             << "        \"tile_shape\": \""
             << candidate.fields.tile_shape << "\",\n"
             << "        \"backward\": \"" << candidate.fields.backward
             << "\",\n"
             << "        \"submission\": \"" << candidate.fields.submission
             << "\",\n"
             << "        \"targets\": \"" << candidate.fields.targets
             << "\",\n"
             << "        \"radix\": \"" << candidate.fields.radix << "\",\n"
             << "        \"projection_adam\": \""
             << candidate.fields.projection_adam << "\",\n"
             << "        \"pair_dispatch\": \""
             << candidate.fields.pair_dispatch << "\"\n"
             << "      },\n"
             << "      \"constraints\": [\n";
        for (size_t check = 0; check < candidate.constraints.size();
             check++) {
            const auto& result = candidate.constraints[check];
            json << "        {\"constraint_id\": \"" << result.constraint_id
                 << "\", \"result\": \"" << result.result
                 << "\", \"reason\": \"" << JsonEscape(result.reason)
                 << "\", \"evidence_key\": \""
                 << JsonEscape(result.evidence_key) << "\"}"
                 << (check + 1 < candidate.constraints.size() ? ",\n"
                                                              : "\n");
        }
        json << "      ],\n"
             << "      \"accepted\": "
             << (candidate.accepted ? "true" : "false") << ",\n"
             << "      \"policy_selected\": "
             << (candidate.policy_selected ? "true" : "false") << "\n"
             << "    }"
             << (index + 1 < ledger.candidates.size() ? ",\n" : "\n");
    }
    json << "  ]\n}\n";
    return json.str();
}

}  // namespace cross3dgs
