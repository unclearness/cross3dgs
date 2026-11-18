// Offline diagnostic schedule-ledger tool.
//
// Emits the frozen candidate schema, hard-constraint catalog, and selection
// policy, and regenerates candidate accept/reject ledgers from a recorded
// device capability vector without needing the device connected. The live
// per-run ledger (with schedule-IR cross-check) is written by the training
// CLI; this tool exists so development-device ledgers can be regenerated
// from their recorded capability dumps.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cross3dgs/schedule_ledger.h"

namespace {

void WriteFile(const std::filesystem::path& path,
               const std::string& content) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << content;
    std::cout << path.string() << " sha256="
              << cross3dgs::ScheduleSha256Hex(content) << "\n";
}

uint32_t ParseU32(const std::string& value) {
    return static_cast<uint32_t>(std::stoul(value, nullptr, 0));
}

}  // namespace

int main(int argc, char** argv) try {
    using namespace cross3dgs;
    std::filesystem::path spec_dir;
    std::filesystem::path ledger_output;
    ScheduleDeviceEvidence device;
    ScheduleLedgerWorkload workload;
    workload.gaussian_capacity = 54275;
    workload.width = 1237;
    workload.height = 822;
    workload.resident_view_capacity = 169;
    workload.sh_degree = 0;
    bool have_caps = false;
    for (int i = 1; i < argc; i++) {
        const std::string argument = argv[i];
        const auto value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            return argv[i];
        };
        if (argument == "--emit-spec") spec_dir = value();
        else if (argument == "--ledger-output") ledger_output = value();
        else if (argument == "--vendor-id") {
            device.caps.vendor_id = ParseU32(value());
            have_caps = true;
        } else if (argument == "--device-id")
            device.caps.device_id = ParseU32(value());
        else if (argument == "--subgroup-size")
            device.caps.subgroup_size = ParseU32(value());
        else if (argument == "--max-shared-memory")
            device.caps.max_shared_memory = ParseU32(value());
        else if (argument == "--max-storage-buffer-bindings")
            device.caps.max_storage_buffer_bindings = ParseU32(value());
        else if (argument == "--float32-atomic-add")
            device.caps.float32_atomic_add = ParseU32(value()) != 0;
        else if (argument == "--dispatch-indirect")
            device.caps.dispatch_indirect = ParseU32(value()) != 0;
        else if (argument == "--f16-storage")
            device.caps.f16_storage = ParseU32(value()) != 0;
        else if (argument == "--max-storage-buffer-range")
            device.max_storage_buffer_range =
                    std::stoull(value(), nullptr, 0);
        else if (argument == "--evidence-key")
            device.evidence_key = value();
        else if (argument == "--gaussian-capacity")
            workload.gaussian_capacity = std::stoull(value());
        else if (argument == "--width") workload.width = ParseU32(value());
        else if (argument == "--height") workload.height = ParseU32(value());
        else if (argument == "--resident-view-capacity")
            workload.resident_view_capacity = ParseU32(value());
        else if (argument == "--sh-degree")
            workload.sh_degree = ParseU32(value());
        else if (argument == "--byte-targets")
            workload.byte_targets = ParseU32(value()) != 0;
        else if (argument == "--enable-igs")
            workload.enable_igs = ParseU32(value()) != 0;
        else if (argument == "--enable-gradient-accumulator")
            workload.enable_gradient_accumulator = ParseU32(value()) != 0;
        else if (argument == "--help") {
            std::cout << "cross_3dgs_schedule_spec [--emit-spec DIR] "
                         "[--ledger-output PATH --vendor-id 0x.. "
                         "--device-id N --subgroup-size N "
                         "--max-shared-memory N "
                         "--max-storage-buffer-bindings N "
                         "--float32-atomic-add 0|1 --dispatch-indirect 0|1 "
                         "--f16-storage 0|1 "
                         "[--max-storage-buffer-range BYTES] "
                         "--evidence-key TEXT "
                         "[--gaussian-capacity N --width N --height N "
                         "--resident-view-capacity N --sh-degree N "
                         "--byte-targets 0|1 --enable-igs 0|1 "
                         "--enable-gradient-accumulator 0|1]]\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown argument " + argument);
        }
    }
    if (spec_dir.empty() && ledger_output.empty()) {
        throw std::invalid_argument(
                "nothing to do; use --emit-spec and/or --ledger-output");
    }
    if (!spec_dir.empty()) {
        WriteFile(spec_dir / "candidate_schema.json",
                  ScheduleCandidateSchemaJson());
        WriteFile(spec_dir / "constraint_catalog.json",
                  ScheduleConstraintCatalogJson());
        WriteFile(spec_dir / "selection_policy.json", SchedulePolicyJson());
    }
    if (!ledger_output.empty()) {
        if (!have_caps) {
            throw std::invalid_argument(
                    "--ledger-output requires the recorded capability "
                    "vector (--vendor-id ...)");
        }
        const auto ledger = EnumerateScheduleCandidates(device, workload);
        WriteFile(ledger_output,
                  ScheduleLedgerJson(ledger, device, workload, nullptr));
        std::cout << "policy_selected="
                  << ledger.policy_selected_candidate_id << "\n";
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
}
