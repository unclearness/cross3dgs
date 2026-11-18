// Immutable capture/run storage and the resumable run manifest for the iOS
// capture-to-view app. Mirrors ProjectManifest.kt / ProjectWorkspace /
// CaptureStore of the Android :e2e module: the same directory layout under
// Documents/capture_to_view/, the same project_manifest.json keys (schema 6),
// create-once artifacts and SHA-256 sealed captures.
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, E2eStage) {
    E2eStageCapture,
    E2eStageExtract,
    E2eStageSfm,
    E2eStageOptimize,
    E2eStageView,
    E2eStageComplete,
    E2eStageFailed,
};

NSString* E2eStageName(E2eStage stage);
E2eStage E2eStageFromName(NSString* name);

// Demo/Full budgets and the shared quality recipe (Android OptimizationPreset).
@interface E2ePreset : NSObject
+ (NSInteger)iterationsForPreset:(NSString*)preset;
+ (NSInteger)capacityForPreset:(NSString*)preset;
+ (BOOL)fullImageSsim;
+ (float)mcmcCloneOffset;
+ (float)mcmcGradientSampling;
@end

NSString* E2eUniqueId(NSString* prefix);
NSString* E2eSha256(NSString* path);
// Copies every file under source into destination; fails on any existing
// target instead of overwriting.
BOOL E2eCopyTreeNoOverwrite(NSString* source, NSString* destination,
                            NSError** error);
BOOL E2eDirectoryHasFiles(NSString* path);

@interface E2eWorkspace : NSObject
@property(nonatomic, readonly) NSString* root;
@property(nonatomic, readonly) NSString* captureDir;
@property(nonatomic, readonly) NSString* videoFile;
@property(nonatomic, readonly) NSString* sensorFile;
@property(nonatomic, readonly) NSString* rawPosesFile;
@property(nonatomic, readonly) NSString* framePriorsFile;
@property(nonatomic, readonly) NSString* sourceImages;
@property(nonatomic, readonly) NSString* sfmWorkspace;
@property(nonatomic, readonly) NSString* sfmSparse;
@property(nonatomic, readonly) NSString* trainingRoot;
@property(nonatomic, readonly) NSString* trainingImages;
@property(nonatomic, readonly) NSString* trainingSparse;
@property(nonatomic, readonly) NSString* output;
@property(nonatomic, readonly) NSString* screenshotsDir;
@property(nonatomic, readonly) NSString* manifestFile;
@property(nonatomic, readonly) NSString* pipelineLog;
@property(nonatomic, readonly) NSString* telemetryFile;

+ (instancetype)openInRunsRoot:(NSString*)runsRoot runId:(NSString*)runId;
+ (nullable instancetype)inspect:(NSString*)root;
- (void)ensureLayout;
- (void)saveManifest:(NSDictionary*)manifest;
- (nullable NSMutableDictionary*)loadManifest;
// Throws (NSException) when directory already holds files.
- (void)requireEmpty:(NSString*)directory label:(NSString*)label;
- (NSString*)relative:(NSString*)path;
// SHA-256 of every artifact under the evidence roots (Android
// ProjectEvidence.inventory), excluding the manifest itself.
- (NSDictionary<NSString*, NSString*>*)inventory;
@end

@interface E2eCaptureRecord : NSObject
@property(nonatomic, readonly) NSString* root;
@property(nonatomic, readonly) NSString* captureId;
@property(nonatomic, readonly) NSString* videoFile;
@property(nonatomic, readonly) NSString* rawPosesFile;
@property(nonatomic, readonly) NSString* sensorFile;
@property(nonatomic, readonly) NSString* screenshotFile;
@property(nonatomic, readonly) NSString* manifestFile;
- (instancetype)initWithRoot:(NSString*)root;
- (nullable NSString*)completenessError;
- (nullable NSString*)sealError;
// Writes capture_manifest.json (SHA-256 inventory) once; verifies if present.
- (nullable NSString*)seal;
@end

@interface E2eCaptureStore : NSObject
- (instancetype)initWithRoot:(NSString*)root;
- (E2eCaptureRecord*)create;
- (NSArray<E2eCaptureRecord*>*)listComplete;
@end

// Default manifest for a new run (Android ProjectManifest.create plus the
// pipeline configuration record).
NSMutableDictionary* E2eNewManifest(NSString* runId, NSString* preset,
                                    NSInteger frameTargetCount,
                                    NSDictionary* optimizationConfig,
                                    NSDictionary* environment);
NSDictionary* E2ePipelineConfig(NSInteger frameTargetCount,
                                NSDictionary* optimizationConfig,
                                NSString* preset, NSInteger tileBinCapacity,
                                BOOL sfmLivePreview);
NSDictionary* E2eEnvironment(void);

NS_ASSUME_NONNULL_END
