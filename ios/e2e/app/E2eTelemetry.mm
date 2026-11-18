#import "E2eTelemetry.h"

#import <UIKit/UIKit.h>
#include <mach/mach.h>
#include <os/proc.h>
#include <time.h>
#include <unistd.h>

namespace {
NSLock* g_lock = [[NSLock alloc] init];
NSFileHandle* g_file = nil;
NSString* g_path = nil;
NSString* g_stage = @"capture";
dispatch_source_t g_timer = nil;
long long g_peak = -1;

long long PhysFootprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_REV1_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return -1;
    }
    return static_cast<long long>(info.phys_footprint);
}

void Sample() {
    const long long footprint = PhysFootprint();
    const long long available = static_cast<long long>(os_proc_available_memory());
    const NSInteger thermal = NSProcessInfo.processInfo.thermalState;
    __block float battery = -1;
    __block NSInteger batteryState = -1;
    dispatch_sync(dispatch_get_main_queue(), ^{
        battery = UIDevice.currentDevice.batteryLevel;
        batteryState = UIDevice.currentDevice.batteryState;
    });
    [g_lock lock];
    if (g_file != nil) {
        if (footprint > g_peak) g_peak = footprint;
        NSString* row = [NSString stringWithFormat:@"%llu,%lld,%@,%d,%lld,%lld,%ld,%.3f,%ld\n",
                         clock_gettime_nsec_np(CLOCK_UPTIME_RAW),
                         (long long)(NSDate.date.timeIntervalSince1970 * 1000.0), g_stage, getpid(),
                         footprint, available, (long)thermal, battery >= 0 ? battery * 100.0 : -1.0,
                         (long)batteryState];
        [g_file writeData:[row dataUsingEncoding:NSUTF8StringEncoding]];
    }
    [g_lock unlock];
}
}  // namespace

@implementation E2eTelemetry

+ (void)startWithFile:(NSString*)path stage:(NSString*)stage {
    [g_lock lock];
    const BOOL same = g_file != nil && [g_path isEqualToString:path];
    g_stage = [stage copy];
    [g_lock unlock];
    if (same) return;
    [self stop];
    dispatch_async(dispatch_get_main_queue(), ^{
        UIDevice.currentDevice.batteryMonitoringEnabled = YES;
    });
    NSFileManager* manager = NSFileManager.defaultManager;
    if (![manager fileExistsAtPath:path]) {
        [@"uptime_raw_ns,wall_time_ms,stage,pid,phys_footprint_bytes,"
         @"os_proc_available_memory_bytes,thermal_state,battery_level_percent,battery_state\n"
                writeToFile:path atomically:NO encoding:NSUTF8StringEncoding error:nil];
    }
    [g_lock lock];
    g_file = [NSFileHandle fileHandleForWritingAtPath:path];
    [g_file seekToEndOfFile];
    g_path = [path copy];
    g_peak = -1;
    g_timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                     dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    dispatch_source_set_timer(g_timer, dispatch_time(DISPATCH_TIME_NOW, 0), NSEC_PER_SEC,
                              NSEC_PER_SEC / 10);
    dispatch_source_set_event_handler(g_timer, ^{ Sample(); });
    dispatch_resume(g_timer);
    [g_lock unlock];
}

+ (void)setStage:(NSString*)stage {
    [g_lock lock];
    g_stage = [stage copy];
    [g_lock unlock];
}

+ (void)stop {
    [g_lock lock];
    if (g_timer != nil) {
        dispatch_source_cancel(g_timer);
        g_timer = nil;
    }
    [g_file synchronizeFile];
    [g_file closeFile];
    g_file = nil;
    g_path = nil;
    [g_lock unlock];
}

+ (long long)peakFootprintBytes {
    [g_lock lock];
    const long long peak = g_peak;
    [g_lock unlock];
    return peak;
}

@end
