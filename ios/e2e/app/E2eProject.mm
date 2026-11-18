#import "E2eProject.h"

#import <UIKit/UIKit.h>
#include <sys/utsname.h>

#include <string>

#include "E2ePipeline.h"

NSString* E2eStageName(E2eStage stage) {
    switch (stage) {
        case E2eStageCapture: return @"capture";
        case E2eStageExtract: return @"extract";
        case E2eStageSfm: return @"sfm";
        case E2eStageOptimize: return @"optimize";
        case E2eStageView: return @"view";
        case E2eStageComplete: return @"complete";
        case E2eStageFailed: return @"failed";
    }
    return @"failed";
}

E2eStage E2eStageFromName(NSString* name) {
    for (E2eStage stage = E2eStageCapture; stage <= E2eStageFailed;
         stage = static_cast<E2eStage>(stage + 1)) {
        if ([E2eStageName(stage) isEqualToString:name]) return stage;
    }
    return E2eStageFailed;
}

@implementation E2ePreset
// Same budgets and recipe as OptimizationPreset in ProjectManifest.kt.
+ (NSInteger)iterationsForPreset:(NSString*)preset {
    return [preset isEqualToString:@"full"] ? 30000 : 3000;
}
+ (NSInteger)capacityForPreset:(NSString*)preset {
    // Multiples of 32, so the trainer can use the reordered SH-rest layout
    // (gated on capacity % 32 == 0); 250,016 is 250,000 rounded up.
    return [preset isEqualToString:@"full"] ? 1000000 : 250016;
}
+ (BOOL)fullImageSsim { return YES; }
+ (float)mcmcCloneOffset { return 0.5f; }
+ (float)mcmcGradientSampling { return 1.5f; }
@end

NSString* E2eUniqueId(NSString* prefix) {
    NSDateFormatter* formatter = [[NSDateFormatter alloc] init];
    formatter.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
    formatter.dateFormat = @"yyyyMMdd_HHmmss_SSS";
    NSString* uuid = [[NSUUID UUID].UUIDString lowercaseString];
    return [NSString stringWithFormat:@"%@_%@_%@", prefix,
                                      [formatter stringFromDate:[NSDate date]],
                                      [uuid substringToIndex:8]];
}

NSString* E2eSha256(NSString* path) {
    return [NSString stringWithUTF8String:
            cross3dgs_ios::Sha256File(path.fileSystemRepresentation).c_str()];
}

static NSArray<NSString*>* FilesUnder(NSString* root) {
    NSMutableArray<NSString*>* files = [NSMutableArray array];
    BOOL directory = NO;
    NSFileManager* manager = NSFileManager.defaultManager;
    if (![manager fileExistsAtPath:root isDirectory:&directory]) return files;
    if (!directory) {
        [files addObject:root];
        return files;
    }
    NSDirectoryEnumerator* enumerator = [manager enumeratorAtPath:root];
    for (NSString* relative in enumerator) {
        NSString* path = [root stringByAppendingPathComponent:relative];
        BOOL isDirectory = NO;
        if ([manager fileExistsAtPath:path isDirectory:&isDirectory] &&
            !isDirectory) {
            [files addObject:path];
        }
    }
    return files;
}

BOOL E2eDirectoryHasFiles(NSString* path) {
    return FilesUnder(path).count > 0;
}

BOOL E2eCopyTreeNoOverwrite(NSString* source, NSString* destination,
                            NSError** error) {
    NSFileManager* manager = NSFileManager.defaultManager;
    for (NSString* file in FilesUnder(source)) {
        NSString* relative = [file substringFromIndex:source.length];
        NSString* target = [destination stringByAppendingString:relative];
        if ([manager fileExistsAtPath:target]) {
            if (error) {
                *error = [NSError errorWithDomain:@"cross3dgs" code:1 userInfo:@{
                    NSLocalizedDescriptionKey :
                            [@"refusing to overwrite " stringByAppendingString:target]
                }];
            }
            return NO;
        }
        [manager createDirectoryAtPath:target.stringByDeletingLastPathComponent
           withIntermediateDirectories:YES attributes:nil error:nil];
        if (![manager copyItemAtPath:file toPath:target error:error]) return NO;
    }
    return YES;
}

@implementation E2eWorkspace

- (instancetype)initWithRoot:(NSString*)root {
    if ((self = [super init])) {
        _root = [root copy];
        _captureDir = [root stringByAppendingPathComponent:@"input/capture"];
        _videoFile = [_captureDir stringByAppendingPathComponent:@"video.mp4"];
        _sensorFile = [_captureDir stringByAppendingPathComponent:@"sensors.csv"];
        _rawPosesFile = [_captureDir stringByAppendingPathComponent:@"arkit_poses.txt"];
        _framePriorsFile = [root stringByAppendingPathComponent:@"extract/frame_priors.txt"];
        _sourceImages = [root stringByAppendingPathComponent:@"extract/images"];
        _sfmWorkspace = [root stringByAppendingPathComponent:@"sfm"];
        _sfmSparse = [_sfmWorkspace stringByAppendingPathComponent:@"sparse"];
        _trainingRoot = [root stringByAppendingPathComponent:@"training"];
        _trainingImages = [root stringByAppendingPathComponent:@"training/images"];
        _trainingSparse = [root stringByAppendingPathComponent:@"training/sparse/0"];
        _output = [root stringByAppendingPathComponent:@"output"];
        _screenshotsDir = [root stringByAppendingPathComponent:@"screenshots"];
        _manifestFile = [root stringByAppendingPathComponent:@"project_manifest.json"];
        _pipelineLog = [root stringByAppendingPathComponent:@"pipeline.log"];
        _telemetryFile = [root stringByAppendingPathComponent:@"telemetry.csv"];
    }
    return self;
}

+ (instancetype)openInRunsRoot:(NSString*)runsRoot runId:(NSString*)runId {
    E2eWorkspace* workspace = [[E2eWorkspace alloc]
            initWithRoot:[runsRoot stringByAppendingPathComponent:runId]];
    [workspace ensureLayout];
    return workspace;
}

+ (nullable instancetype)inspect:(NSString*)root {
    BOOL directory = NO;
    if (![NSFileManager.defaultManager fileExistsAtPath:root isDirectory:&directory] ||
        !directory) {
        return nil;
    }
    return [[E2eWorkspace alloc] initWithRoot:root];
}

- (void)ensureLayout {
    for (NSString* directory in @[
             _captureDir, _sourceImages, _sfmSparse, _trainingImages,
             _trainingSparse, _output,
             _framePriorsFile.stringByDeletingLastPathComponent, _screenshotsDir
         ]) {
        [NSFileManager.defaultManager createDirectoryAtPath:directory
                                withIntermediateDirectories:YES
                                                 attributes:nil
                                                      error:nil];
    }
}

- (void)saveManifest:(NSDictionary*)manifest {
    [self ensureLayout];
    NSError* error = nil;
    NSData* data = [NSJSONSerialization
            dataWithJSONObject:manifest
                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys |
                               NSJSONWritingWithoutEscapingSlashes
                         error:&error];
    if (data == nil) {
        [NSException raise:@"E2eManifest"
                    format:@"manifest is not JSON: %@", error.localizedDescription];
    }
    // Atomic replace, like the Android pending-file + ATOMIC_MOVE.
    if (![data writeToFile:_manifestFile options:NSDataWritingAtomic error:&error]) {
        [NSException raise:@"E2eManifest"
                    format:@"failed to save manifest: %@", error.localizedDescription];
    }
}

- (nullable NSMutableDictionary*)loadManifest {
    NSData* data = [NSData dataWithContentsOfFile:_manifestFile];
    if (data == nil) return nil;
    id json = [NSJSONSerialization JSONObjectWithData:data
                                              options:NSJSONReadingMutableContainers
                                                error:nil];
    return [json isKindOfClass:NSMutableDictionary.class] ? json : nil;
}

- (void)requireEmpty:(NSString*)directory label:(NSString*)label {
    if (E2eDirectoryHasFiles(directory)) {
        [NSException raise:@"E2eWorkspace"
                    format:@"%@ output already exists; create a new run instead "
                           @"of overwriting",
                           label];
    }
}

- (NSString*)relative:(NSString*)path {
    if ([path hasPrefix:[_root stringByAppendingString:@"/"]]) {
        return [path substringFromIndex:_root.length + 1];
    }
    return path;
}

- (NSDictionary<NSString*, NSString*>*)inventory {
    NSMutableDictionary<NSString*, NSString*>* hashes = [NSMutableDictionary dictionary];
    for (NSString* relativeRoot in @[
             @"input/capture", @"extract", @"sfm", @"training", @"output",
             @"screenshots", @"pipeline.log", @"telemetry.csv"
         ]) {
        for (NSString* file in FilesUnder([_root stringByAppendingPathComponent:relativeRoot])) {
            if ([file isEqualToString:_manifestFile] || [file hasSuffix:@".pending"]) {
                continue;
            }
            hashes[[self relative:file]] = E2eSha256(file);
        }
    }
    return hashes;
}

@end

@implementation E2eCaptureRecord

- (instancetype)initWithRoot:(NSString*)root {
    if ((self = [super init])) {
        _root = [root copy];
        _captureId = root.lastPathComponent;
        _videoFile = [root stringByAppendingPathComponent:@"video.mp4"];
        _rawPosesFile = [root stringByAppendingPathComponent:@"arkit_poses.txt"];
        _sensorFile = [root stringByAppendingPathComponent:@"sensors.csv"];
        _screenshotFile = [root stringByAppendingPathComponent:@"capture.png"];
        _manifestFile = [root stringByAppendingPathComponent:@"capture_manifest.json"];
    }
    return self;
}

static unsigned long long FileSize(NSString* path) {
    return [[NSFileManager.defaultManager attributesOfItemAtPath:path error:nil] fileSize];
}

- (nullable NSString*)completenessError {
    for (NSString* path in @[ _videoFile, _rawPosesFile, _sensorFile, _screenshotFile ]) {
        if (FileSize(path) == 0) {
            return [NSString stringWithFormat:@"capture %@ has no %@", _captureId,
                                              path.lastPathComponent];
        }
    }
    return nil;
}

- (nullable NSString*)sealError {
    NSString* incomplete = [self completenessError];
    if (incomplete != nil) return incomplete;
    NSData* data = [NSData dataWithContentsOfFile:_manifestFile];
    if (data == nil) return [NSString stringWithFormat:@"capture %@ is not sealed", _captureId];
    NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![json[@"capture_id"] isEqual:_captureId]) {
        return @"capture manifest id does not match directory";
    }
    NSDictionary* hashes = json[@"artifact_sha256"];
    if (![hashes isKindOfClass:NSDictionary.class] || hashes.count < 4) {
        return @"capture has an incomplete SHA-256 inventory";
    }
    for (NSString* name in hashes) {
        if ([name containsString:@"/"]) return @"capture artifact must be a direct child";
        NSString* path = [_root stringByAppendingPathComponent:name];
        if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
            return [@"capture artifact is missing: " stringByAppendingString:name];
        }
        if (![E2eSha256(path) isEqualToString:[hashes[name] lowercaseString]]) {
            return [@"capture artifact SHA-256 mismatch: " stringByAppendingString:name];
        }
    }
    return nil;
}

- (nullable NSString*)seal {
    if ([NSFileManager.defaultManager fileExistsAtPath:_manifestFile]) {
        return [self sealError];
    }
    NSString* incomplete = [self completenessError];
    if (incomplete != nil) return incomplete;
    NSMutableDictionary* hashes = [NSMutableDictionary dictionary];
    for (NSString* path in @[ _videoFile, _rawPosesFile, _sensorFile, _screenshotFile ]) {
        hashes[path.lastPathComponent] = E2eSha256(path);
    }
    NSDictionary* json = @{
        @"schema_version" : @1,
        @"capture_id" : _captureId,
        @"sealed_at_epoch_ms" : @((long long)(NSDate.date.timeIntervalSince1970 * 1000.0)),
        @"artifact_sha256" : hashes,
    };
    NSData* data = [NSJSONSerialization
            dataWithJSONObject:json
                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                         error:nil];
    if (![data writeToFile:_manifestFile options:NSDataWritingWithoutOverwriting error:nil]) {
        return @"failed to seal immutable capture manifest";
    }
    return [self sealError];
}

@end

@implementation E2eCaptureStore {
    NSString* _root;
}

- (instancetype)initWithRoot:(NSString*)root {
    if ((self = [super init])) {
        _root = [root copy];
        [NSFileManager.defaultManager createDirectoryAtPath:root
                                withIntermediateDirectories:YES
                                                 attributes:nil
                                                      error:nil];
    }
    return self;
}

- (E2eCaptureRecord*)create {
    NSString* path = [_root stringByAppendingPathComponent:E2eUniqueId(@"capture")];
    NSError* error = nil;
    if (![NSFileManager.defaultManager createDirectoryAtPath:path
                                 withIntermediateDirectories:NO
                                                  attributes:nil
                                                       error:&error]) {
        [NSException raise:@"E2eCaptureStore"
                    format:@"failed to create unique capture directory: %@",
                           error.localizedDescription];
    }
    return [[E2eCaptureRecord alloc] initWithRoot:path];
}

- (NSArray<E2eCaptureRecord*>*)listComplete {
    NSMutableArray<E2eCaptureRecord*>* records = [NSMutableArray array];
    NSArray* names = [NSFileManager.defaultManager contentsOfDirectoryAtPath:_root error:nil];
    for (NSString* name in [names sortedArrayUsingSelector:@selector(compare:)].reverseObjectEnumerator) {
        E2eCaptureRecord* record =
                [[E2eCaptureRecord alloc] initWithRoot:[_root stringByAppendingPathComponent:name]];
        if ([record sealError] == nil) [records addObject:record];
    }
    return records;
}

@end

NSDictionary* E2ePipelineConfig(NSInteger frameTargetCount,
                                NSDictionary* optimizationConfig,
                                NSString* preset, NSInteger tileBinCapacity,
                                BOOL sfmLivePreview) {
    // Keys and values follow ProjectEvidence.pipelineConfig on Android; the
    // iOS-only entries name the capture API and the decoder.
    return @{
        @"platform" : @"ios",
        @"capture_api" : @"arkit_world_tracking_avassetwriter_h264",
        @"frame_extractor" : @"uniform_nx2_candidates_sharpest_luma_v2",
        @"frame_decoder" : @"avassetimagegenerator_zero_tolerance",
        @"frame_target_count" : [@(frameTargetCount) stringValue],
        @"frame_candidates_per_target" : @"2",
        @"frame_max_decode_dimension" : @"1920",
        @"frame_jpeg_quality" : @"92",
        @"sfm" : @"incremental_colmap_cpu_sift8192_single_camera_v1",
        @"sfm_feature_max_image_size" : @"2400",
        @"sfm_max_sift_features" : @"8192",
        @"sfm_pairing" : @"exhaustive_le_100_else_sequential",
        @"sfm_exhaustive_block_size" : @"20",
        @"sfm_sequential_overlap" : @"15",
        @"sfm_init_min_num_inliers" : @"50",
        @"sfm_position_priors" : @"enabled_when_present",
        @"sfm_gravity_priors" : @"recorded_not_full_6dof_initialization",
        // Display-only observer that shares the CPU with SfM; recorded for timing.
        @"sfm_live_preview" : sfmLivePreview ? @"library_rendered_max_4fps" : @"disabled",
        @"optimizer_initialization" : @"colmap_sparse_points",
        @"optimizer_training_view_seed" : @"42",
        @"optimizer_eval_interval" : @"8",
        @"optimizer_holdout_rule" : @"sorted_image_index_modulo_eval_interval_zero_based",
        @"optimizer_holdout_evaluator" : @"native_psnr_plus_render_target_png_for_host_rescoring",
        @"optimizer_iterations" : [optimizationConfig[@"iterations"] stringValue],
        @"optimizer_gaussian_capacity" : [optimizationConfig[@"gaussian_capacity"] stringValue],
        @"optimizer_sh_degree" : [optimizationConfig[@"sh_degree"] stringValue],
        @"optimizer_density_strategy" : optimizationConfig[@"density_strategy"],
        @"optimizer_precision" : @"fp32",
        @"optimizer_preset" : preset,
        @"optimizer_full_image_ssim" : E2ePreset.fullImageSsim ? @"true" : @"false",
        @"optimizer_mcmc_clone_offset" : [NSString stringWithFormat:@"%g", E2ePreset.mcmcCloneOffset],
        @"optimizer_mcmc_gradient_sampling" :
                [NSString stringWithFormat:@"%g", E2ePreset.mcmcGradientSampling],
        @"optimizer_tile_bin_capacity" : [@(tileBinCapacity) stringValue],
    };
}

NSDictionary* E2eEnvironment(void) {
    struct utsname name {};
    uname(&name);
    NSDictionary* info = NSBundle.mainBundle.infoDictionary;
    return @{
        @"manufacturer" : @"Apple",
        @"model" : UIDevice.currentDevice.model,
        @"device" : [NSString stringWithUTF8String:name.machine],
        @"ios_version" : UIDevice.currentDevice.systemVersion,
        @"kernel" : [NSString stringWithUTF8String:name.version],
        @"application_id" : NSBundle.mainBundle.bundleIdentifier ?: @"unknown",
        @"application_version" : info[@"CFBundleShortVersionString"] ?: @"unknown",
        @"application_build" : info[@"CFBundleVersion"] ?: @"unknown",
    };
}

NSMutableDictionary* E2eNewManifest(NSString* runId, NSString* preset,
                                    NSInteger frameTargetCount,
                                    NSDictionary* optimizationConfig,
                                    NSDictionary* environment) {
    return [@{
        @"schema_version" : @6,
        @"project_id" : runId,
        @"capture_id" : @"",
        @"run_id" : runId,
        @"auto_run_to_end" : @NO,
        @"stage" : @"capture",
        @"preset" : preset,
        @"optimization_config" : [optimizationConfig mutableCopy],
        @"capture_source" : @"",
        @"image_directory" : @"training/images",
        @"sparse_directory" : @"training/sparse/0",
        @"selected_sfm_model" : @"",
        @"frame_target_count" : @(frameTargetCount),
        @"extracted_frames" : @0,
        @"registered_images" : @0,
        @"mean_reprojection_error" : NSNull.null,
        @"position_prior_count" : @0,
        @"gravity_prior_count" : @0,
        @"intrinsics_prior_present" : @NO,
        @"capture_pose_samples" : @0,
        @"capture_sensor_samples" : @0,
        @"stage_times_ms" : [NSMutableDictionary dictionary],
        @"viewer_frame_sweep" : NSNull.null,
        @"artifact_sha256" : [NSMutableDictionary dictionary],
        @"screenshots" : [NSMutableDictionary dictionary],
        @"environment" : environment,
        @"pipeline_config" : @{},
        @"last_error" : NSNull.null,
    } mutableCopy];
}
