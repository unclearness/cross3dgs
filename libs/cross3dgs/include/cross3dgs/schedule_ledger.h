#ifndef CROSS3DGS_SCHEDULE_LEDGER_H
#define CROSS3DGS_SCHEDULE_LEDGER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cross3dgs/gaussian_forward.h"
#include "cross3dgs/gpu/gpu_program.h"

namespace cross3dgs {

// Machine-readable schedule audit artifact.
//
// This module enumerates a candidate schedule tuple space, evaluates hard
// constraints with stable IDs against a device capability vector, and records
// a policy choice. GaussianForwardProgram enumerates this ledger from live
// DeviceCaps and the fixed workload, validates any explicit experiment
// override against it, and derives executable flags from the accepted tuple.
// Shader/pass construction remains implemented in src/gaussian/ and
// radix_sort.cpp; host wrappers supply the fixed run descriptor.

// Identifiers of the constraint catalog and the selection policy. A run's
// catalog and policy are identified by these together with the SHA-256 of
// their JSON payloads.
inline constexpr const char* kScheduleCatalogVersion =
        "cross3dgs-schedule-catalog";
inline constexpr const char* kSchedulePolicyId =
        "cross3dgs-selection-policy";

// Device limits of the tiled_8x8 backward (HC-SUBGROUP-TILED-01), shared by
// the ledger and the runtime. The kernel declares 49152 bytes of shared
// memory, and its only cross-lane step is a subgroupBarrier inside aligned
// groups of 8 lanes, so any subgroup of at least 8 lanes runs it.
inline constexpr uint32_t kTiledBackwardMinSubgroupSize = 8;
inline constexpr uint32_t kTiledBackwardSharedBytes = 49152;

// Candidate schedule tuple. Every field selects among semantics-preserving
// lowerings only; quality-changing choices (SH degree, density strategy,
// Sobol sampled backward, capacity) are a separate operating point and are
// deliberately not part of this tuple.
struct ScheduleCandidateFields {
    // Canonical backward axes. `backward` is retained as a legacy label for
    // readers of v1 artifacts and is not used to imply a single mixed axis.
    std::string work_assignment;  // pixel_stream_subgroup | per_splat | tiled_8x8 | pair_partial
    std::string global_merge;     // native_atomic | u32_cas | gaussian_major_fixed_reduce
    std::string tile_shape;     // none | 8x8
    std::string backward;         // legacy compatibility label
    std::string submission;       // batched_8 | per_pass_fenced
    std::string targets;          // resident | streamed
    std::string radix;            // subgroup_dependent | subgroup_independent
    std::string projection_adam;  // fused | split
    std::string pair_dispatch;    // indirect | direct
};

struct ScheduleConstraintResult {
    std::string constraint_id;
    std::string result;  // pass | fail | unknown
    std::string reason;
    std::string evidence_key;
};

struct ScheduleCandidateRecord {
    std::string candidate_id;
    ScheduleCandidateFields fields;
    std::vector<ScheduleConstraintResult> constraints;
    bool accepted = false;
    bool policy_selected = false;
};

// Device capability evidence. The descriptor-range limit is copied from the
// live DeviceCaps query. The
// separate field is retained for offline/spec tooling (0 = unknown; range
// constraints then evaluate to "unknown", never "pass").
struct ScheduleDeviceEvidence {
    gpu::DeviceCaps caps;
    uint64_t max_storage_buffer_range = 0;
    std::string evidence_key;  // provenance of the capability values
};

struct ScheduleLedgerWorkload {
    size_t gaussian_capacity = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t resident_view_capacity = 0;
    uint32_t sh_degree = 0;
    bool byte_targets = true;
    bool enable_optimizer = true;
    bool enable_igs = false;
    // MCMC gradient-weighted sampling reuses the IGS score buffer without
    // enabling IGS densification. It still consumes the extra descriptor.
    bool enable_gradient_accumulator = false;
};

struct ScheduleCandidateLedger {
    std::vector<ScheduleCandidateRecord> candidates;
    std::string policy_selected_candidate_id;  // empty if none accepted
    std::string policy_id;
    std::string catalog_version;
    std::string catalog_sha256;  // SHA-256 of ScheduleConstraintCatalogJson()
    std::string policy_sha256;   // SHA-256 of SchedulePolicyJson()
};

// Static specification artifacts (device-independent, frozen with the code).
std::string ScheduleConstraintCatalogJson();
std::string SchedulePolicyJson();
std::string ScheduleCandidateSchemaJson();

// Enumerate all candidates, evaluate constraints, apply the frozen policy.
ScheduleCandidateLedger EnumerateScheduleCandidates(
        const ScheduleDeviceEvidence& device,
        const ScheduleLedgerWorkload& workload);

// Map a live program's selected schedule back onto candidate fields.
ScheduleCandidateFields FieldsFromScheduleIR(const GaussianScheduleIR& ir);

std::string ScheduleCandidateId(const ScheduleCandidateFields& fields);

// Serialize the full ledger. When `ir` is non-null the configured schedule of
// the live run is recorded and compared with the policy selection.
std::string ScheduleLedgerJson(const ScheduleCandidateLedger& ledger,
                               const ScheduleDeviceEvidence& device,
                               const ScheduleLedgerWorkload& workload,
                               const GaussianScheduleIR* ir);

std::string ScheduleSha256Hex(const std::string& bytes);

}  // namespace cross3dgs

#endif
