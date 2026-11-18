// cross_3dgs iOS bench shell.
//
// The app is a thin harness around the desktop CLI entry point
// (cross3dgs_train_main): it substitutes sandbox paths into the launch
// arguments, keeps the device awake, tees all console output into
// <run root>/console.log, records a capability probe, a device-state
// snapshot, and process telemetry (thermal, available memory, process
// physical footprint, battery, Low Power Mode, screen brightness,
// lifecycle events), and reports completion through
// <run root>/run_status.json so the host-side harness can poll artifacts
// without a console attached.
//
// App-level arguments (stripped before the trainer sees argv):
//   --ios-run-root <path>   directory (after @DOCUMENTS@/@BUNDLE@
//                           substitution) receiving console.log,
//                           telemetry.csv, caps.json, device_state.json and
//                           run_status.json. Default: Documents, which keeps
//                           the historical layout scripts/benchmark_ios.py
//                           expects.
//   --ios-screen-brightness <0..1>
//                           screen brightness the app sets at launch and
//                           whenever it becomes active again, so power
//                           runs do not depend on the brightness slider.
//
// Process-footprint sampler contract:
//   API:       task_info(mach_task_self(), TASK_VM_INFO_REV1,
//              ...).phys_footprint — the same accounting jetsam uses.
//   Scope:     this app process only (PID recorded per row).
//   Interval:  5 seconds, from a detached thread started at launch.
//   Unit:      bytes.
//   Missing samples: a failed task_info() call writes -1 for the
//              footprint column and the row is still emitted; the host
//              validator counts and bounds such rows.
//   Termination: the thread never exits; the final row before process
//              death is whatever was last flushed (rows are flushed
//              per-sample). os_proc_available_memory() is recorded as
//              available-memory telemetry only, never as a process peak.

#import "AppDelegate.h"

#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
#include <os/proc.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// tools/train_cli/main.cpp compiled with CROSS_3DGS_TRAIN_MAIN_AS_LIBRARY.
int cross3dgs_train_main(int argc, char** argv);

// android/app/src/main/cpp/device_caps.cpp (platform-neutral Vulkan probe;
// the namespace name is historical).
namespace cross3dgs_android {
std::string QueryDeviceCapsJson();
}

// tests/gpu/test_cpu_oracle.cpp compiled with CROSS_3DGS_CPU_ORACLE_LIBRARY,
// the same on-device embedding the Android app uses. Runs the 20-case
// selected-schedule GPU-vs-CPU-double oracle on this device and writes
// cpu_oracle.json into the given directory.
namespace cross3dgs_probe {
bool RunSelectedScheduleCpuOracle(const std::filesystem::path& output_dir,
                                  bool force_split_sh_rest_storage);
}

@interface AppDelegate ()
@property(strong, nonatomic) UITextView* logView;
// Negative: leave the system brightness alone.
@property(nonatomic) double screenBrightness;
@end

namespace {

NSString* g_run_root = nil;
FILE* g_telemetry_file = nullptr;
std::mutex g_telemetry_mutex;
std::atomic<long long> g_phys_footprint_peak_bytes{-1};
std::atomic<long long> g_telemetry_sample_count{0};
NSDate* g_telemetry_start = nil;

NSString* DocumentsPath() {
    return NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
                                               NSUserDomainMask, YES)[0];
}

NSString* RunRootPath() {
    return g_run_root != nil ? g_run_root : DocumentsPath();
}

std::string Utf8(NSString* value) {
    return std::string([value UTF8String] ?: "");
}

NSString* NowUtc() {
    NSISO8601DateFormatter* formatter = [[NSISO8601DateFormatter alloc] init];
    return [formatter stringFromDate:[NSDate date]];
}

const char* ThermalStateName(NSProcessInfoThermalState state) {
    switch (state) {
        case NSProcessInfoThermalStateNominal: return "nominal";
        case NSProcessInfoThermalStateFair: return "fair";
        case NSProcessInfoThermalStateSerious: return "serious";
        case NSProcessInfoThermalStateCritical: return "critical";
    }
    return "unknown";
}

const char* BatteryStateName(UIDeviceBatteryState state) {
    switch (state) {
        case UIDeviceBatteryStateUnplugged: return "unplugged";
        case UIDeviceBatteryStateCharging: return "charging";
        case UIDeviceBatteryStateFull: return "full";
        case UIDeviceBatteryStateUnknown: return "unknown";
    }
    return "unknown";
}

void WriteStringToFile(NSString* path, NSString* contents) {
    [contents writeToFile:path
               atomically:YES
                 encoding:NSUTF8StringEncoding
                    error:nil];
}

// Process physical footprint in bytes (jetsam accounting scope), or -1 when
// task_info fails.
long long PhysFootprintBytes() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_REV1_COUNT;
    const kern_return_t result = task_info(
            mach_task_self(), TASK_VM_INFO,
            reinterpret_cast<task_info_t>(&info), &count);
    if (result != KERN_SUCCESS ||
        count < TASK_VM_INFO_REV1_COUNT) {
        return -1;
    }
    return static_cast<long long>(info.phys_footprint);
}

std::string Sha256OfFile(NSString* path) {
    FILE* file = fopen([path fileSystemRepresentation], "rb");
    if (!file) {
        return "unreadable";
    }
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::vector<unsigned char> buffer(1 << 20);
    size_t read_bytes = 0;
    while ((read_bytes = fread(buffer.data(), 1, buffer.size(), file)) > 0) {
        CC_SHA256_Update(&context, buffer.data(),
                         static_cast<CC_LONG>(read_bytes));
    }
    fclose(file);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int index = 0; index < CC_SHA256_DIGEST_LENGTH; index++) {
        snprintf(hex + 2 * index, 3, "%02x", digest[index]);
    }
    return std::string(hex, 2 * CC_SHA256_DIGEST_LENGTH);
}

// Battery, Low Power Mode and screen state must be read on the main thread;
// marshal from the telemetry thread.
struct MainThreadDeviceState {
    float battery_level = -1.0f;
    const char* battery_state = "unknown";
    int low_power_mode = -1;
    float screen_brightness = -1.0f;
};

MainThreadDeviceState ReadMainThreadDeviceState() {
    __block MainThreadDeviceState state;
    const auto read_block = ^{
      state.battery_level = [UIDevice currentDevice].batteryLevel;
      state.battery_state =
              BatteryStateName([UIDevice currentDevice].batteryState);
      state.low_power_mode =
              [[NSProcessInfo processInfo] isLowPowerModeEnabled] ? 1 : 0;
      state.screen_brightness =
              static_cast<float>(UIScreen.mainScreen.brightness);
    };
    if ([NSThread isMainThread]) {
        read_block();
    } else {
        dispatch_sync(dispatch_get_main_queue(), read_block);
    }
    return state;
}

// Substitute the placeholders the host harness uses so it does not need to
// know sandbox paths: @BUNDLE@ -> app bundle root, @DOCUMENTS@ -> Documents.
std::string SubstitutePlaceholders(std::string arg) {
    const std::string bundle = Utf8([[NSBundle mainBundle] bundlePath]);
    const std::string documents = Utf8(DocumentsPath());
    const auto replace_all = [&arg](const std::string& from,
                                    const std::string& to) {
        size_t pos = 0;
        while ((pos = arg.find(from, pos)) != std::string::npos) {
            arg.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all("@BUNDLE@", bundle);
    replace_all("@DOCUMENTS@", documents);
    return arg;
}

// One telemetry/event row. Regular samples use event="", lifecycle and
// memory-warning rows reuse the same schema with the event column set so a
// single chronological file orders samples and events together.
void AppendTelemetryRow(const char* event) {
    const long long footprint = PhysFootprintBytes();
    if (footprint >= 0) {
        long long expected = g_phys_footprint_peak_bytes.load();
        while (footprint > expected &&
               !g_phys_footprint_peak_bytes.compare_exchange_weak(expected,
                                                                  footprint)) {
        }
    }
    const size_t available = os_proc_available_memory();
    const char* thermal =
            ThermalStateName([[NSProcessInfo processInfo] thermalState]);
    const MainThreadDeviceState state = ReadMainThreadDeviceState();
    const std::string now = Utf8(NowUtc());
    const double elapsed =
            g_telemetry_start != nil
                    ? [[NSDate date] timeIntervalSinceDate:g_telemetry_start]
                    : -1.0;
    std::lock_guard<std::mutex> lock(g_telemetry_mutex);
    if (!g_telemetry_file) {
        return;
    }
    fprintf(g_telemetry_file,
            "%s,%.3f,%d,%s,%zu,%lld,%.3f,%s,%d,%.3f,%s\n", now.c_str(),
            elapsed, getpid(), thermal, available, footprint,
            state.battery_level, state.battery_state, state.low_power_mode,
            state.screen_brightness, event);
    fflush(g_telemetry_file);
    g_telemetry_sample_count.fetch_add(1);
}

}  // namespace

@implementation AppDelegate

- (void)appendLog:(NSString*)line {
    dispatch_async(dispatch_get_main_queue(), ^{
      UITextView* view = self.logView;
      if (view == nil) {
          return;
      }
      NSString* updated = [view.text stringByAppendingString:line];
      // Keep the on-screen tail bounded; the full log is in console.log.
      const NSUInteger limit = 20000;
      if (updated.length > limit) {
          updated = [updated substringFromIndex:updated.length - limit];
      }
      view.text = updated;
      [view scrollRangeToVisible:NSMakeRange(updated.length, 0)];
    });
}

// Tee stdout+stderr: back to the original console (so `devicectl … --console`
// still works), into <run root>/console.log, and onto the screen.
- (void)startConsoleTee {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);
    int original = dup(STDOUT_FILENO);
    int fds[2];
    if (pipe(fds) != 0) {
        return;
    }
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);
    const int read_fd = fds[0];
    NSString* log_path =
            [RunRootPath() stringByAppendingPathComponent:@"console.log"];
    FILE* log_file = fopen([log_path fileSystemRepresentation], "a");
    AppDelegate* __weak weak_self = self;
    std::thread([weak_self, read_fd, original, log_file] {
        std::vector<char> buffer(4096);
        for (;;) {
            const ssize_t n = read(read_fd, buffer.data(), buffer.size());
            if (n <= 0) {
                break;
            }
            write(original, buffer.data(), static_cast<size_t>(n));
            if (log_file) {
                fwrite(buffer.data(), 1, static_cast<size_t>(n), log_file);
                fflush(log_file);
            }
            NSString* chunk = [[NSString alloc]
                    initWithBytes:buffer.data()
                           length:static_cast<NSUInteger>(n)
                         encoding:NSUTF8StringEncoding];
            if (chunk != nil) {
                [weak_self appendLog:chunk];
            }
        }
    }).detach();
}

- (void)startTelemetry {
    NSString* path =
            [RunRootPath() stringByAppendingPathComponent:@"telemetry.csv"];
    {
        std::lock_guard<std::mutex> lock(g_telemetry_mutex);
        g_telemetry_file = fopen([path fileSystemRepresentation], "a");
        if (g_telemetry_file) {
            fprintf(g_telemetry_file,
                    "utc,elapsed_seconds,pid,thermal_state,"
                    "available_memory_bytes,phys_footprint_bytes,"
                    "battery_level,battery_state,low_power_mode,"
                    "screen_brightness,event\n");
            fflush(g_telemetry_file);
        }
        g_telemetry_start = [NSDate date];
    }
    std::thread([] {
        for (;;) {
            AppendTelemetryRow("");
            const long long footprint = PhysFootprintBytes();
            printf("[telemetry] thermal=%s available_memory=%zu "
                   "phys_footprint=%lld\n",
                   ThermalStateName(
                           [[NSProcessInfo processInfo] thermalState]),
                   os_proc_available_memory(), footprint);
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }).detach();
}

- (void)writeRunStatus:(NSString*)state exitCode:(int)exit_code {
    NSString* path =
            [RunRootPath() stringByAppendingPathComponent:@"run_status.json"];
    NSString* json = [NSString
            stringWithFormat:
                    @"{\"state\": \"%@\", \"exit_code\": %d, "
                    @"\"utc\": \"%@\", \"pid\": %d, "
                    @"\"phys_footprint_peak_bytes\": %lld, "
                    @"\"telemetry_sample_count\": %lld}\n",
                    state, exit_code, NowUtc(), getpid(),
                    g_phys_footprint_peak_bytes.load(),
                    g_telemetry_sample_count.load()];
    WriteStringToFile(path, json);
}

- (void)runCapsProbe {
    const std::string caps = cross3dgs_android::QueryDeviceCapsJson();
    printf("[caps] %s\n", caps.c_str());
    NSString* path =
            [RunRootPath() stringByAppendingPathComponent:@"caps.json"];
    WriteStringToFile(path, [NSString stringWithUTF8String:caps.c_str()]);
}

- (void)writeDeviceState {
    struct utsname names;
    uname(&names);
    const MainThreadDeviceState state = ReadMainThreadDeviceState();
    NSString* executable = [[NSBundle mainBundle] executablePath];
    NSDictionary* filesystem = [[NSFileManager defaultManager]
            attributesOfFileSystemForPath:DocumentsPath()
                                    error:nil];
    long long free_bytes =
            [filesystem[NSFileSystemFreeSize] longLongValue];
    NSString* json = [NSString
            stringWithFormat:
                    @"{\n"
                    @"  \"utc\": \"%@\",\n"
                    @"  \"hardware_machine\": \"%s\",\n"
                    @"  \"system_name\": \"%@\",\n"
                    @"  \"system_version\": \"%@\",\n"
                    @"  \"process_pid\": %d,\n"
                    @"  \"bundle_path\": \"%@\",\n"
                    @"  \"bundle_identifier\": \"%@\",\n"
                    @"  \"executable_path\": \"%@\",\n"
                    @"  \"executable_sha256\": \"%s\",\n"
                    @"  \"thermal_state\": \"%s\",\n"
                    @"  \"available_memory_bytes\": %zu,\n"
                    @"  \"phys_footprint_bytes\": %lld,\n"
                    @"  \"battery_level\": %.3f,\n"
                    @"  \"battery_state\": \"%s\",\n"
                    @"  \"low_power_mode\": %d,\n"
                    @"  \"screen_brightness\": %.3f,\n"
                    @"  \"documents_free_bytes\": %lld,\n"
                    @"  \"idle_timer_disabled\": true,\n"
                    @"  \"run_root\": \"%@\"\n"
                    @"}\n",
                    NowUtc(), names.machine,
                    [UIDevice currentDevice].systemName,
                    [UIDevice currentDevice].systemVersion, getpid(),
                    [[NSBundle mainBundle] bundlePath],
                    [[NSBundle mainBundle] bundleIdentifier], executable,
                    Sha256OfFile(executable).c_str(),
                    ThermalStateName(
                            [[NSProcessInfo processInfo] thermalState]),
                    os_proc_available_memory(), PhysFootprintBytes(),
                    state.battery_level, state.battery_state,
                    state.low_power_mode, state.screen_brightness,
                    free_bytes, RunRootPath()];
    NSString* path = [RunRootPath()
            stringByAppendingPathComponent:@"device_state.json"];
    WriteStringToFile(path, json);
    printf("[device_state] machine=%s ios=%s battery=%.2f/%s lpm=%d "
           "free_bytes=%lld\n",
           names.machine, Utf8([UIDevice currentDevice].systemVersion).c_str(),
           state.battery_level, state.battery_state, state.low_power_mode,
           free_bytes);
}

- (void)startTrainingWithArgs:(NSArray<NSString*>*)args {
    // Keep the argv storage alive for the lifetime of the training thread.
    auto storage = std::make_shared<std::vector<std::string>>();
    storage->push_back("cross3dgs_train");
    for (NSString* arg in args) {
        storage->push_back(SubstitutePlaceholders(Utf8(arg)));
    }
    [self writeRunStatus:@"running" exitCode:-1];
    AppDelegate* __weak weak_self = self;
    std::thread([weak_self, storage] {
        std::vector<char*> argv;
        argv.reserve(storage->size());
        for (auto& arg : *storage) {
            argv.push_back(arg.data());
        }
        printf("[run] starting:");
        for (const auto& arg : *storage) {
            printf(" %s", arg.c_str());
        }
        printf("\n");
        int exit_code = 1;
        try {
            exit_code = cross3dgs_train_main(static_cast<int>(argv.size()),
                                             argv.data());
        } catch (const std::exception& error) {
            fprintf(stderr, "[run] uncaught exception: %s\n", error.what());
        } catch (...) {
            fprintf(stderr, "[run] uncaught non-standard exception\n");
        }
        printf("[run] finished exit_code=%d\n", exit_code);
        AppDelegate* strong_self = weak_self;
        if (strong_self != nil) {
            [strong_self writeRunStatus:@"finished" exitCode:exit_code];
        }
    }).detach();
}

- (BOOL)application:(UIApplication*)application
        didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    UIViewController* controller = [[UIViewController alloc] init];
    UITextView* logView = [[UITextView alloc] initWithFrame:CGRectZero];
    logView.editable = NO;
    logView.backgroundColor = UIColor.blackColor;
    logView.textColor = UIColor.greenColor;
    logView.font = [UIFont monospacedSystemFontOfSize:9
                                               weight:UIFontWeightRegular];
    logView.translatesAutoresizingMaskIntoConstraints = NO;
    [controller.view addSubview:logView];
    [NSLayoutConstraint activateConstraints:@[
        [logView.topAnchor constraintEqualToAnchor:controller.view
                                                           .safeAreaLayoutGuide
                                                           .topAnchor],
        [logView.bottomAnchor constraintEqualToAnchor:controller.view
                                                              .bottomAnchor],
        [logView.leadingAnchor
                constraintEqualToAnchor:controller.view.leadingAnchor],
        [logView.trailingAnchor
                constraintEqualToAnchor:controller.view.trailingAnchor],
    ]];
    self.logView = logView;
    self.window.rootViewController = controller;
    [self.window makeKeyAndVisible];

    // The bench must survive long unattended runs.
    application.idleTimerDisabled = YES;
    [UIDevice currentDevice].batteryMonitoringEnabled = YES;

    // devicectl passes launch arguments through to the process; argv[0] is
    // the executable path. App-level arguments are consumed here and removed
    // before the remainder is handed to the trainer.
    self.screenBrightness = -1.0;
    NSMutableArray<NSString*>* args =
            [[[NSProcessInfo processInfo] arguments] mutableCopy];
    [args removeObjectAtIndex:0];
    NSUInteger run_root_index = [args indexOfObject:@"--ios-run-root"];
    if (run_root_index != NSNotFound && run_root_index + 1 < args.count) {
        NSString* raw = args[run_root_index + 1];
        g_run_root = [NSString
                stringWithUTF8String:SubstitutePlaceholders(Utf8(raw))
                                             .c_str()];
        [args removeObjectsInRange:NSMakeRange(run_root_index, 2)];
        [[NSFileManager defaultManager] createDirectoryAtPath:g_run_root
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];
    }
    NSUInteger oracle_index = [args indexOfObject:@"--ios-cpu-oracle"];
    NSString* oracle_output = nil;
    if (oracle_index != NSNotFound && oracle_index + 1 < args.count) {
        oracle_output = [NSString
                stringWithUTF8String:SubstitutePlaceholders(
                                             Utf8(args[oracle_index + 1]))
                                             .c_str()];
        [args removeObjectsInRange:NSMakeRange(oracle_index, 2)];
    }
    NSUInteger brightness_index =
            [args indexOfObject:@"--ios-screen-brightness"];
    if (brightness_index != NSNotFound && brightness_index + 1 < args.count) {
        self.screenBrightness = std::clamp(
                [args[brightness_index + 1] doubleValue], 0.0, 1.0);
        [args removeObjectsInRange:NSMakeRange(brightness_index, 2)];
        UIScreen.mainScreen.brightness = self.screenBrightness;
    }

    [self startConsoleTee];
    printf("[app] cross3dgs iOS bench started %s\n",
           Utf8(NowUtc()).c_str());
    printf("[app] bundle=%s\n",
           Utf8([[NSBundle mainBundle] bundlePath]).c_str());
    printf("[app] documents=%s\n", Utf8(DocumentsPath()).c_str());
    printf("[app] run_root=%s\n", Utf8(RunRootPath()).c_str());
    [self startTelemetry];

    dispatch_async(
            dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
              [self writeDeviceState];
              [self runCapsProbe];
              if (oracle_output != nil) {
                  printf("[app] running selected-schedule CPU oracle\n");
                  [self writeRunStatus:@"running" exitCode:-1];
                  bool oracle_pass = false;
                  try {
                      oracle_pass =
                              cross3dgs_probe::RunSelectedScheduleCpuOracle(
                                      std::filesystem::path(
                                              Utf8(oracle_output)),
                                      false);
                  } catch (const std::exception& error) {
                      fprintf(stderr, "[oracle] uncaught exception: %s\n",
                              error.what());
                  }
                  printf("[oracle] pass=%d\n", oracle_pass ? 1 : 0);
                  [self writeRunStatus:@"finished"
                              exitCode:oracle_pass ? 0 : 1];
              } else if (args.count == 0) {
                  printf("[app] no launch arguments; capability probe only\n");
                  [self writeRunStatus:@"caps_only" exitCode:0];
              } else {
                  [self startTrainingWithArgs:args];
              }
            });
    return YES;
}

- (void)applicationDidReceiveMemoryWarning:(UIApplication*)application {
    printf("[app] memory warning; available_memory=%zu phys_footprint=%lld\n",
           os_proc_available_memory(), PhysFootprintBytes());
    std::thread([] { AppendTelemetryRow("memory_warning"); }).detach();
}

- (void)applicationWillResignActive:(UIApplication*)application {
    printf("[app] lifecycle: will_resign_active\n");
    std::thread([] { AppendTelemetryRow("will_resign_active"); }).detach();
}

- (void)applicationDidEnterBackground:(UIApplication*)application {
    printf("[app] lifecycle: did_enter_background\n");
    AppendTelemetryRow("did_enter_background");
}

- (void)applicationDidBecomeActive:(UIApplication*)application {
    printf("[app] lifecycle: did_become_active\n");
    if (self.screenBrightness >= 0.0) {
        UIScreen.mainScreen.brightness = self.screenBrightness;
    }
    std::thread([] { AppendTelemetryRow("did_become_active"); }).detach();
}

@end
