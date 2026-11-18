#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Once-per-second process/thermal/battery samples across the pipeline
// (Android ProjectTelemetry). Columns are iOS-native and scope-labeled:
// process physical footprint (task_info TASK_VM_INFO phys_footprint, the
// jetsam accounting), os_proc_available_memory, NSProcessInfo thermal state
// and UIDevice battery level/state. Unavailable values are written as -1.
@interface E2eTelemetry : NSObject
+ (void)startWithFile:(NSString*)path stage:(NSString*)stage;
+ (void)setStage:(NSString*)stage;
+ (void)stop;
// Largest phys_footprint observed since the last start, or -1.
+ (long long)peakFootprintBytes;
@end

NS_ASSUME_NONNULL_END
