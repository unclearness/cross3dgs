// Capture-to-view pipeline screen: ARKit capture -> sharp-frame extraction
// -> CPU COLMAP SfM with AR priors -> Cross3DGS optimization -> free-view
// orbit viewer. Port of android/e2e/.../E2eActivity.kt; the run layout,
// manifest keys, stage gating, presets, automatic stage screenshots, viewer
// frame sweep and final SHA-256 inventory follow the Android app.
//
// Launch arguments (xcrun devicectl ... process launch <bundle> -key value),
// read through NSUserDefaults' argument domain:
//   -run_id <id>               open (or create) a specific run
//   -preset demo|full          preset for the fresh run
//   -frame_target_count <n>    frames to extract (8..240)
//   -auto_start YES            continue the opened run to the end
//   -image_folder <path>       start a run from a folder of JPEGs
//                              (Documents-relative or absolute), skipping
//                              capture/extract; no AR priors
//   -saved_capture <id>        start a run from a sealed capture
//   -viewer_reference_image <name>  viewer reference camera
//   -sfm_preview NO            no SfM live preview for new runs (timing
//                              A/Bs); recorded in pipeline_config
//   -screen_brightness <0..1>  screen brightness set at launch and whenever
//                              the app becomes active again (power runs)
#import "E2eViewController.h"

#import <AVFoundation/AVFoundation.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <string>
#include <vector>

#import "E2eCaptureViewController.h"
#include "E2eEngine.h"
#include "cross3dgs/orbit_view.h"
#import "E2eFrameExtractor.h"
#include "E2ePipeline.h"
#import "E2eProject.h"
#import "E2eTelemetry.h"
#include "cross3dgs_sfm.h"

namespace {
// Per-tile pair budget for capture-resolution training (see E2eActivity
// TILE_BIN_CAPACITY: 2048 completed the workstation 30k/1M replays with zero
// overflow).
constexpr NSInteger kTileBinCapacity = 2048;
constexpr NSInteger kDefaultFrameTargetCount = 60;
constexpr NSInteger kMinFrameTargetCount = 8;
constexpr NSInteger kMaxFrameTargetCount = 240;
constexpr NSInteger kMinIterations = 1;
constexpr NSInteger kMaxIterations = 100000;
constexpr NSInteger kMinCapacity = 10000;
constexpr NSInteger kMaxCapacity = 2000000;
constexpr int kViewerSweepFrames = 60;
constexpr float kViewerSweepYawSpan = 0.7f;
constexpr float kOrbitTurnPerSweep = 3.14159f;
NSArray<NSString*>* RequiredScreenshotStages() {
    return @[ @"capture", @"extract", @"sfm", @"optimize", @"view" ];
}
// SIFT keypoints and the pair matrix during SfM; absent when the live preview
// is off.
NSString* const kSiftScreenshotStage = @"sfm_features";
NSString* const kMatchingScreenshotStage = @"sfm_matching";
NSArray<NSString*>* OptionalScreenshotStages() { return @[ kSiftScreenshotStage, kMatchingScreenshotStage ]; }

long long NowMs() { return (long long)(CACurrentMediaTime() * 1000.0); }

// Copies `pixels` (RGBA8, row-major) into a new image.
UIImage* ImageFromRgbaBytes(const unsigned char* pixels, size_t width, size_t height) {
    NSData* data = [NSData dataWithBytes:pixels length:width * height * 4];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGImageRef cg = CGImageCreate(width, height, 8, 32, width * 4, space,
                                  kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big, provider,
                                  nullptr, false, kCGRenderingIntentDefault);
    UIImage* result = [UIImage imageWithCGImage:cg];
    CGImageRelease(cg);
    CGColorSpaceRelease(space);
    CGDataProviderRelease(provider);
    return result;
}

UIImage* ImageFromRgba(const cross3dgs_ios::RgbaImage& image) {
    return ImageFromRgbaBytes(image.pixels.data(), image.width, image.height);
}

std::string Std(NSString* value) { return value.UTF8String ?: ""; }

// The SfM preview caption (sfm_preview.cc) with its step spelled out, after
// "SfM: ".
NSString* SfmStatus(NSString* caption) {
    NSArray<NSArray<NSString*>*>* steps = @[
        @[ @"SIFT ", @"SIFT extraction " ],
        @[ @"Matching ", @"Feature matching " ],
        @[ @"Incremental SfM ", @"Incremental mapping " ],
        @[ @"Sparse model ", @"Incremental mapping done " ],
    ];
    for (NSArray<NSString*>* step in steps) {
        if ([caption hasPrefix:step[0]]) {
            caption = [step[1] stringByAppendingString:[caption substringFromIndex:step[0].length]];
            break;
        }
    }
    return [@"SfM: " stringByAppendingString:caption];
}

// The optimize screenshot taken while training runs.
enum TrainingShot : NSInteger { kTrainingShotNone, kTrainingShotPending, kTrainingShotDone, kTrainingShotFailed };
}  // namespace

@interface E2eViewController () <UIGestureRecognizerDelegate>
@end

@implementation E2eViewController {
    NSString* _storageRoot;
    NSString* _runsRoot;
    E2eCaptureStore* _captureStore;
    E2eWorkspace* _workspace;
    NSMutableDictionary* _manifest;  // guarded by @synchronized(self)

    NSString* _preset;
    NSInteger _frameTargetCount;
    NSMutableDictionary* _optimizationConfig;

    UILabel* _status;
    UIButton* _captureButton;
    UIButton* _extractButton;
    UIButton* _sfmButton;
    UIButton* _optimizeButton;
    UIButton* _openFinishedButton;
    UIButton* _savedCaptureButton;
    UIButton* _frozenReplayButton;
    UIButton* _presetButton;
    UIButton* _frameTargetButton;
    UIButton* _settingsButton;
    UIButton* _viewButton;
    UIButton* _resetViewButton;
    UIButton* _cancelButton;
    UIButton* _shareButton;
    UIProgressView* _progress;
    UIImageView* _surface;
    UITextView* _log;
    BOOL _busy;

    dispatch_queue_t _stageQueue;
    dispatch_queue_t _viewerQueue;
    std::atomic<int> _activeStages;
    std::atomic<bool> _viewerReady;
    std::atomic<bool> _viewerPending;
    std::atomic<bool> _viewerDirty;
    float _orbitYaw, _orbitPitch, _orbitZoom, _orbitPanX, _orbitPanY;
    float _orbitMinPitch, _orbitMaxPitch;  // from the prepared viewer
    NSLock* _orbitLock;
    CGSize _surfaceSize;
    CGFloat _pinchStartZoom;
    CGPoint _lastPan;
    NSInteger _lastPanTouches;

    long long _optimizationStartedMs;
    // Main thread: the optimize screenshot taken during training, and what
    // onTrainingFinished continues with once it has settled.
    TrainingShot _trainingShot;
    void (^_afterTrainingShot)(void);
    // Main thread: panel captions over the surface while training.
    NSArray<UILabel*>* _trainingLabels;
    cross3dgs::TrainingPreviewLayout _trainingLayout;
    NSMutableSet<NSString*>* _previewShotsRequested;  // main thread
    long long _viewerStartedMs;
    NSLock* _sfmTimingLock;
    NSString* _sfmSubstage;
    long long _sfmSubstageStartedMs;
    NSMutableDictionary* _sfmStageTimes;
    NSFileHandle* _logFile;
}

#pragma mark - Lifecycle

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.systemBackgroundColor;
    _stageQueue = dispatch_queue_create("cross3dgs.e2e.stage", DISPATCH_QUEUE_SERIAL);
    _viewerQueue = dispatch_queue_create("cross3dgs.e2e.viewer", DISPATCH_QUEUE_SERIAL);
    _orbitLock = [[NSLock alloc] init];
    _sfmTimingLock = [[NSLock alloc] init];
    _sfmStageTimes = [NSMutableDictionary dictionary];
    _orbitZoom = 1.0f;
    _orbitMinPitch = -cross3dgs::kMaxOrbitPitch;
    _orbitMaxPitch = cross3dgs::kMaxOrbitPitch;

    NSString* documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
    _storageRoot = [documents stringByAppendingPathComponent:@"capture_to_view"];
    _captureStore = [[E2eCaptureStore alloc] initWithRoot:[_storageRoot stringByAppendingPathComponent:@"captures"]];
    _runsRoot = [_storageRoot stringByAppendingPathComponent:@"runs"];
    [NSFileManager.defaultManager createDirectoryAtPath:_runsRoot withIntermediateDirectories:YES attributes:nil error:nil];

    NSUserDefaults* args = NSUserDefaults.standardUserDefaults;
    NSString* requested = [args stringForKey:@"run_id"] ?: E2eUniqueId(@"run");
    NSCharacterSet* invalid = [[NSCharacterSet characterSetWithCharactersInString:
            @"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"] invertedSet];
    if ([requested rangeOfCharacterFromSet:invalid].location != NSNotFound) requested = E2eUniqueId(@"run");
    NSString* launchPreset = [[args stringForKey:@"preset"] lowercaseString];
    if (![launchPreset isEqualToString:@"demo"] && ![launchPreset isEqualToString:@"full"]) launchPreset = nil;
    if ([args objectForKey:@"screen_brightness"] != nil) {
        const CGFloat brightness = std::clamp([args doubleForKey:@"screen_brightness"], 0.0, 1.0);
        UIScreen.mainScreen.brightness = brightness;
        [NSNotificationCenter.defaultCenter addObserverForName:UIApplicationDidBecomeActiveNotification
                                                        object:nil
                                                         queue:NSOperationQueue.mainQueue
                                                    usingBlock:^(NSNotification*) {
                                                        UIScreen.mainScreen.brightness = brightness;
                                                    }];
    }

    _workspace = [E2eWorkspace openInRunsRoot:_runsRoot runId:requested];
    _manifest = [_workspace loadManifest];
    _preset = launchPreset ?: _manifest[@"preset"] ?: @"demo";
    _frameTargetCount = [args integerForKey:@"frame_target_count"] ?: [_manifest[@"frame_target_count"] integerValue]
                                                                        ?: kDefaultFrameTargetCount;
    _frameTargetCount = std::clamp(_frameTargetCount, kMinFrameTargetCount, kMaxFrameTargetCount);
    if (_manifest != nil && launchPreset == nil) {
        _optimizationConfig = [_manifest[@"optimization_config"] mutableCopy];
    } else {
        _optimizationConfig = [self configForPreset:_preset];
    }
    if (_manifest == nil) {
        _manifest = E2eNewManifest(requested, _preset, _frameTargetCount, _optimizationConfig, E2eEnvironment());
        _manifest[@"pipeline_config"] = [self pipelineConfig];
        [_workspace saveManifest:_manifest];
    }
    [self openLogFile];
    if (![self.stage isEqualToString:@"complete"]) {
        [E2eTelemetry startWithFile:_workspace.telemetryFile stage:self.stage];
    }

    [self buildUi];
    [self installEngineCallbacks];
    [self refresh];
    dispatch_async(dispatch_get_main_queue(), ^{
        [self configureAutomation];
    });
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    const CGFloat scale = UIScreen.mainScreen.scale;
    const CGSize size = CGSizeMake(_surface.bounds.size.width * scale, _surface.bounds.size.height * scale);
    const BOOL changed = !CGSizeEqualToSize(size, _surfaceSize);
    _surfaceSize = size;
    if (changed && _viewerReady) [self requestOrbitRender];
}

- (NSMutableDictionary*)configForPreset:(NSString*)preset {
    return [@{
        @"iterations" : @([E2ePreset iterationsForPreset:preset]),
        @"gaussian_capacity" : @([E2ePreset capacityForPreset:preset]),
        @"sh_degree" : @3,
        @"density_strategy" : @"mcmc",
    } mutableCopy];
}

- (NSDictionary*)pipelineConfig {
    NSUserDefaults* args = NSUserDefaults.standardUserDefaults;
    const BOOL preview = [args objectForKey:@"sfm_preview"] == nil || [args boolForKey:@"sfm_preview"];
    return E2ePipelineConfig(_frameTargetCount, _optimizationConfig, _preset, kTileBinCapacity, preview);
}

- (NSString*)stage {
    @synchronized(self) {
        return _manifest[@"stage"];
    }
}

- (void)updateManifest:(void (^)(NSMutableDictionary* manifest))mutate {
    @synchronized(self) {
        mutate(_manifest);
        [_workspace saveManifest:_manifest];
    }
}

- (void)setStageTime:(NSString*)key ms:(long long)ms {
    [self updateManifest:^(NSMutableDictionary* manifest) {
        NSMutableDictionary* times = [manifest[@"stage_times_ms"] mutableCopy] ?: [NSMutableDictionary dictionary];
        times[key] = @(ms);
        manifest[@"stage_times_ms"] = times;
    }];
}

#pragma mark - UI

- (UIButton*)button:(NSString*)title action:(SEL)action {
    UIButtonConfiguration* config = [UIButtonConfiguration tintedButtonConfiguration];
    config.title = title;
    config.titleLineBreakMode = NSLineBreakByTruncatingTail;
    config.contentInsets = NSDirectionalEdgeInsetsMake(6, 6, 6, 6);
    config.titleTextAttributesTransformer = ^NSDictionary*(NSDictionary* attributes) {
        NSMutableDictionary* updated = [attributes mutableCopy];
        updated[NSFontAttributeName] = [UIFont systemFontOfSize:13 weight:UIFontWeightMedium];
        return updated;
    };
    UIButton* button = [UIButton buttonWithConfiguration:config primaryAction:nil];
    [button addTarget:self action:action forControlEvents:UIControlEventTouchUpInside];
    return button;
}

- (void)setTitle:(NSString*)title ofButton:(UIButton*)button {
    UIButtonConfiguration* config = button.configuration;
    config.title = title;
    button.configuration = config;
}

- (UIStackView*)row:(NSArray<UIView*>*)views {
    UIStackView* row = [[UIStackView alloc] initWithArrangedSubviews:views];
    row.axis = UILayoutConstraintAxisHorizontal;
    row.distribution = UIStackViewDistributionFillEqually;
    row.spacing = 6;
    return row;
}

- (void)buildUi {
    _status = [[UILabel alloc] init];
    _status.font = [UIFont systemFontOfSize:13];
    // Three lines, so a long SfM caption (matching with thousands of pairs)
    // wraps without losing its legend line.
    _status.numberOfLines = 3;
    _captureButton = [self button:@"Capture" action:@selector(startCapture)];
    _extractButton = [self button:@"Extract" action:@selector(extractFrames)];
    _sfmButton = [self button:@"SfM" action:@selector(runSfm)];
    _optimizeButton = [self button:@"Optimize" action:@selector(runOptimization)];
    _openFinishedButton = [self button:@"Open finished run" action:@selector(chooseFinishedRun)];
    _savedCaptureButton = [self button:@"Run saved capture" action:@selector(chooseSavedCapture)];
    _frozenReplayButton = [self button:@"Replay frozen SfM input" action:@selector(chooseFrozenReplay)];
    _presetButton = [self button:@"" action:@selector(togglePreset)];
    _frameTargetButton = [self button:@"" action:@selector(showFrameTargetDialog)];
    _settingsButton = [self button:@"" action:@selector(showOptimizationSettings)];
    _viewButton = [self button:@"View result" action:@selector(prepareViewerAndRender)];
    _resetViewButton = [self button:@"Reset view" action:@selector(resetView)];
    _cancelButton = [self button:@"Cancel stage" action:@selector(cancelStage)];
    _shareButton = [self button:@"Share PLY" action:@selector(sharePly)];
    _progress = [[UIProgressView alloc] initWithProgressViewStyle:UIProgressViewStyleDefault];

    _surface = [[UIImageView alloc] init];
    _surface.backgroundColor = UIColor.blackColor;
    _surface.contentMode = UIViewContentModeScaleAspectFit;
    _surface.userInteractionEnabled = YES;
    _surface.clipsToBounds = YES;
    UIPanGestureRecognizer* pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handlePan:)];
    pan.maximumNumberOfTouches = 2;
    pan.delegate = self;
    UIPinchGestureRecognizer* pinch = [[UIPinchGestureRecognizer alloc] initWithTarget:self action:@selector(handlePinch:)];
    pinch.delegate = self;
    [_surface addGestureRecognizer:pan];
    [_surface addGestureRecognizer:pinch];
    NSMutableArray<UILabel*>* trainingLabels = [NSMutableArray array];
    for (const std::string& caption : cross3dgs::TrainingPreviewLabels()) {
        UILabel* label = [[UILabel alloc] init];
        label.text = [NSString stringWithUTF8String:caption.c_str()];
        label.font = [UIFont systemFontOfSize:13];
        label.textColor = UIColor.whiteColor;
        label.textAlignment = NSTextAlignmentCenter;
        label.numberOfLines = 0;
        label.backgroundColor = [UIColor colorWithWhite:0 alpha:0.6];
        label.layer.cornerRadius = 4;
        label.clipsToBounds = YES;
        label.hidden = YES;
        [_surface addSubview:label];
        [trainingLabels addObject:label];
    }
    _trainingLabels = trainingLabels;

    _log = [[UITextView alloc] init];
    _log.editable = NO;
    _log.font = [UIFont monospacedSystemFontOfSize:9 weight:UIFontWeightRegular];
    _log.backgroundColor = UIColor.secondarySystemBackgroundColor;

    UIStackView* stack = [[UIStackView alloc] initWithArrangedSubviews:@[
        _status,
        [self row:@[ _captureButton, _extractButton, _sfmButton, _optimizeButton ]],
        [self row:@[ _openFinishedButton, _savedCaptureButton ]],
        [self row:@[ _frozenReplayButton, _presetButton ]],
        [self row:@[ _frameTargetButton, _settingsButton ]],
        _progress,
        _surface,
        [self row:@[ _viewButton, _resetViewButton, _cancelButton, _shareButton ]],
        _log,
    ]];
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 6;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:stack];
    UILayoutGuide* safe = self.view.safeAreaLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:safe.topAnchor constant:8],
        [stack.bottomAnchor constraintEqualToAnchor:safe.bottomAnchor constant:-8],
        [stack.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:10],
        [stack.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-10],
        [_log.heightAnchor constraintEqualToAnchor:_surface.heightAnchor multiplier:0.5],
    ]];
    [_surface setContentHuggingPriority:UILayoutPriorityDefaultLow forAxis:UILayoutConstraintAxisVertical];
    [_surface setContentCompressionResistancePriority:UILayoutPriorityDefaultLow
                                              forAxis:UILayoutConstraintAxisVertical];
}

- (void)refresh {
    NSDictionary* manifest;
    @synchronized(self) {
        manifest = [_manifest copy];
    }
    NSString* stage = manifest[@"stage"];
    [self setTitle:[NSString stringWithFormat:@"Preset: %@", _preset.uppercaseString] ofButton:_presetButton];
    [self setTitle:[NSString stringWithFormat:@"Extract frames: %ld", (long)_frameTargetCount]
          ofButton:_frameTargetButton];
    [self setTitle:[NSString stringWithFormat:@"3DGS %@it %ldk SH%@ %@",
                                              _optimizationConfig[@"iterations"],
                                              (long)([_optimizationConfig[@"gaussian_capacity"] integerValue] / 1000),
                                              _optimizationConfig[@"sh_degree"],
                                              [_optimizationConfig[@"density_strategy"] uppercaseString]]
          ofButton:_settingsButton];
    if (!_busy) {
        NSString* error = [manifest[@"last_error"] isKindOfClass:NSString.class] ? manifest[@"last_error"] : nil;
        _status.text = [NSString stringWithFormat:@"Run %@ | stage %@%@", manifest[@"run_id"], stage.uppercaseString,
                                                  error ? [@" | " stringByAppendingString:error] : @""];
    }
    NSFileManager* manager = NSFileManager.defaultManager;
    const BOOL idle = !_busy;
    _captureButton.enabled = idle;
    _openFinishedButton.enabled = idle;
    _savedCaptureButton.enabled = idle;
    _frozenReplayButton.enabled = idle;
    _presetButton.enabled = idle;
    _frameTargetButton.enabled = idle;
    _settingsButton.enabled = idle;
    _extractButton.enabled = idle && [stage isEqualToString:@"extract"] && [manager fileExistsAtPath:_workspace.videoFile];
    _sfmButton.enabled = idle && [stage isEqualToString:@"sfm"] && E2eDirectoryHasFiles(_workspace.sourceImages);
    _optimizeButton.enabled = idle && [stage isEqualToString:@"optimize"] &&
                              [manager fileExistsAtPath:[_workspace.trainingSparse stringByAppendingPathComponent:@"images.bin"]];
    NSString* checkpoint = [_workspace.output stringByAppendingPathComponent:@"checkpoint.d3gs"];
    _viewButton.enabled = idle && ([stage isEqualToString:@"view"] || [stage isEqualToString:@"complete"]) &&
                          [manager fileExistsAtPath:checkpoint];
    _shareButton.enabled = [manager fileExistsAtPath:[_workspace.output stringByAppendingPathComponent:@"splat.ply"]];
    _cancelButton.enabled = _busy;
}

- (void)setBusy:(BOOL)busy message:(nullable NSString*)message {
    void (^apply)(void) = ^{
        self->_busy = busy;
        if (message != nil) self->_status.text = message;
        UIApplication.sharedApplication.idleTimerDisabled = busy;
        [self refresh];
        if (message != nil && busy) self->_status.text = message;
    };
    if (NSThread.isMainThread) apply(); else dispatch_async(dispatch_get_main_queue(), apply);
}

- (void)openLogFile {
    [_logFile closeFile];
    if (![NSFileManager.defaultManager fileExistsAtPath:_workspace.pipelineLog]) {
        [NSFileManager.defaultManager createFileAtPath:_workspace.pipelineLog contents:nil attributes:nil];
    }
    _logFile = [NSFileHandle fileHandleForWritingAtPath:_workspace.pipelineLog];
    [_logFile seekToEndOfFile];
}

- (void)appendLog:(NSString*)line {
    NSString* stamped = [NSString stringWithFormat:@"%lld %@\n",
                         (long long)(NSDate.date.timeIntervalSince1970 * 1000.0), line];
    @synchronized(self) {
        // A sealed run's log is part of its hashed inventory.
        if (![_manifest[@"stage"] isEqualToString:@"complete"]) {
            [_logFile writeData:[stamped dataUsingEncoding:NSUTF8StringEncoding]];
        }
    }
    NSLog(@"%@", line);
    dispatch_async(dispatch_get_main_queue(), ^{
        NSString* text = [self->_log.text stringByAppendingString:stamped];
        if (text.length > 60000) text = [text substringFromIndex:text.length - 50000];
        self->_log.text = text;
        [self->_log scrollRangeToVisible:NSMakeRange(text.length - 1, 1)];
    });
}

- (void)showSurfaceImage:(UIImage*)image {
    if (NSThread.isMainThread) {
        _surface.image = image;
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{
            self->_surface.image = image;
        });
    }
}

- (void)alert:(NSString*)title message:(NSString*)message {
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:title
                                                                   message:message
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (BOOL)refuseWhileRunning {
    if (_activeStages.load() == 0) return NO;
    [self alert:@"Busy" message:@"A run is still working. Cancel it first."];
    return YES;
}

#pragma mark - Settings

- (BOOL)pendingConfigurable {
    @synchronized(self) {
        return [_manifest[@"stage"] isEqualToString:@"capture"] && [_manifest[@"capture_id"] length] == 0;
    }
}

- (void)persistPendingConfiguration {
    if (![self pendingConfigurable]) return;
    [self updateManifest:^(NSMutableDictionary* manifest) {
        manifest[@"preset"] = self->_preset;
        manifest[@"frame_target_count"] = @(self->_frameTargetCount);
        manifest[@"optimization_config"] = [self->_optimizationConfig mutableCopy];
        manifest[@"pipeline_config"] = [self pipelineConfig];
    }];
}

- (void)togglePreset {
    _preset = [_preset isEqualToString:@"demo"] ? @"full" : @"demo";
    _optimizationConfig = [self configForPreset:_preset];
    [self persistPendingConfiguration];
    [self refresh];
}

- (void)showFrameTargetDialog {
    UIAlertController* alert = [UIAlertController
            alertControllerWithTitle:@"Frames to extract"
                             message:[NSString stringWithFormat:@"Enter %ld-%ld. Each target decodes two candidates "
                                                                @"and keeps the sharper frame.",
                                                                (long)kMinFrameTargetCount, (long)kMaxFrameTargetCount]
                      preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField* field) {
        field.keyboardType = UIKeyboardTypeNumberPad;
        field.text = [@(self->_frameTargetCount) stringValue];
    }];
    [alert addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"Apply" style:UIAlertActionStyleDefault handler:^(UIAlertAction*) {
        const NSInteger value = alert.textFields[0].text.integerValue;
        if (value < kMinFrameTargetCount || value > kMaxFrameTargetCount) {
            [self alert:@"Invalid" message:@"Frame count out of range"];
            return;
        }
        self->_frameTargetCount = value;
        [self persistPendingConfiguration];
        [self refresh];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)showOptimizationSettings {
    UIAlertController* alert = [UIAlertController
            alertControllerWithTitle:@"3DGS settings"
                             message:@"Iterations, max Gaussians, SH degree (0-3), density (mcmc/fixed). Large "
                                     @"capacities can exhaust device memory."
                      preferredStyle:UIAlertControllerStyleAlert];
    NSArray* placeholders = @[ @"Iterations", @"Max Gaussians", @"SH degree", @"Density strategy" ];
    NSArray* values = @[
        [_optimizationConfig[@"iterations"] stringValue], [_optimizationConfig[@"gaussian_capacity"] stringValue],
        [_optimizationConfig[@"sh_degree"] stringValue], _optimizationConfig[@"density_strategy"]
    ];
    for (NSUInteger index = 0; index < 4; index++) {
        [alert addTextFieldWithConfigurationHandler:^(UITextField* field) {
            field.placeholder = placeholders[index];
            field.text = values[index];
            field.keyboardType = index < 3 ? UIKeyboardTypeNumberPad : UIKeyboardTypeASCIICapable;
        }];
    }
    [alert addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"Apply" style:UIAlertActionStyleDefault handler:^(UIAlertAction*) {
        const NSInteger iterations = alert.textFields[0].text.integerValue;
        const NSInteger capacity = alert.textFields[1].text.integerValue;
        const NSInteger sh = alert.textFields[2].text.integerValue;
        NSString* density = alert.textFields[3].text.lowercaseString;
        if (iterations < kMinIterations || iterations > kMaxIterations || capacity < kMinCapacity ||
            capacity > kMaxCapacity || sh < 0 || sh > 3 ||
            !([density isEqualToString:@"mcmc"] || [density isEqualToString:@"fixed"])) {
            [self alert:@"Invalid" message:@"Iterations 1-100000, Gaussians 10000-2000000, SH 0-3, mcmc|fixed"];
            return;
        }
        self->_optimizationConfig = [@{
            @"iterations" : @(iterations),
            @"gaussian_capacity" : @(capacity),
            @"sh_degree" : @(sh),
            @"density_strategy" : density,
        } mutableCopy];
        [self persistPendingConfiguration];
        [self refresh];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

#pragma mark - Run selection

- (void)switchToFreshRun {
    [E2eTelemetry stop];
    NSString* runId = E2eUniqueId(@"run");
    E2eWorkspace* workspace = [E2eWorkspace openInRunsRoot:_runsRoot runId:runId];
    NSMutableDictionary* manifest =
            E2eNewManifest(runId, _preset, _frameTargetCount, _optimizationConfig, E2eEnvironment());
    manifest[@"pipeline_config"] = [self pipelineConfig];
    [workspace saveManifest:manifest];
    @synchronized(self) {
        _workspace = workspace;
        _manifest = manifest;
        [self openLogFile];
    }
    [E2eTelemetry startWithFile:workspace.telemetryFile stage:@"capture"];
    _viewerReady = false;
    _progress.progress = 0;
    _log.text = @"";
    _surface.image = nil;
}

- (BOOL)needsFreshRun {
    @synchronized(self) {
        return [_manifest[@"capture_id"] length] > 0 || ![_manifest[@"stage"] isEqualToString:@"capture"];
    }
}

- (void)recordScreenshot:(NSString*)stage path:(NSString*)path {
    NSString* relative = [_workspace relative:path];
    NSString* hash = E2eSha256(path);
    [self updateManifest:^(NSMutableDictionary* manifest) {
        NSMutableDictionary* shots = [manifest[@"screenshots"] mutableCopy];
        shots[stage] = relative;
        manifest[@"screenshots"] = shots;
        NSMutableDictionary* hashes = [manifest[@"artifact_sha256"] mutableCopy];
        hashes[relative] = hash;
        manifest[@"artifact_sha256"] = hashes;
    }];
}

- (void)recordFailure:(NSString*)message {
    [self updateManifest:^(NSMutableDictionary* manifest) {
        manifest[@"stage"] = @"failed";
        manifest[@"last_error"] = message;
    }];
    [self appendLog:[@"ERROR: " stringByAppendingString:message]];
}

#pragma mark - Capture

- (void)startCapture {
    if ([self refuseWhileRunning]) return;
    E2eCaptureRecord* capture = [_captureStore create];
    [E2eTelemetry setStage:@"capture"];
    __weak E2eViewController* weakSelf = self;
    E2eCaptureViewController* controller = [[E2eCaptureViewController alloc]
            initWithCaptureRoot:capture.root
                     completion:^(E2eCaptureResult* result, NSString* error) {
                         if (result == nil) {
                             if (error != nil) [weakSelf alert:@"Capture failed" message:error];
                             return;
                         }
                         [weakSelf beginRunFromCapture:capture source:@"on_device_arkit_capture" result:result];
                     }];
    [self presentViewController:controller animated:YES completion:nil];
}

- (void)chooseSavedCapture {
    if ([self refuseWhileRunning]) return;
    NSArray<E2eCaptureRecord*>* captures = [_captureStore listComplete];
    if (captures.count == 0) {
        [self alert:@"No captures" message:@"No completed captures"];
        return;
    }
    UIAlertController* sheet = [UIAlertController alertControllerWithTitle:@"Select immutable capture"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    for (E2eCaptureRecord* capture in captures) {
        [sheet addAction:[UIAlertAction actionWithTitle:capture.captureId style:UIAlertActionStyleDefault
                                                handler:^(UIAlertAction*) {
            [self beginRunFromCapture:capture source:@"saved_capture" result:nil];
        }]];
    }
    [sheet addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    sheet.popoverPresentationController.sourceView = _savedCaptureButton;
    [self presentViewController:sheet animated:YES completion:nil];
}

- (void)beginRunFromCapture:(E2eCaptureRecord*)capture source:(NSString*)source result:(nullable E2eCaptureResult*)result {
    NSString* sealError = [capture seal];
    if (sealError != nil) {
        [self alert:@"Capture unusable" message:sealError];
        return;
    }
    if ([self needsFreshRun]) [self switchToFreshRun];
    @try {
        [_workspace requireEmpty:_workspace.captureDir label:@"run capture input"];
    } @catch (NSException* exception) {
        [self alert:@"Run not empty" message:exception.reason];
        return;
    }
    NSFileManager* manager = NSFileManager.defaultManager;
    NSError* error = nil;
    for (NSString* name in @[ @"video.mp4", @"arkit_poses.txt", @"sensors.csv", @"capture.png", @"capture_manifest.json" ]) {
        if (![manager copyItemAtPath:[capture.root stringByAppendingPathComponent:name]
                              toPath:[_workspace.captureDir stringByAppendingPathComponent:name]
                               error:&error]) {
            [self recordFailure:[NSString stringWithFormat:@"failed to copy %@: %@", name, error.localizedDescription]];
            [self refresh];
            return;
        }
    }
    NSString* screenshot = [_workspace.screenshotsDir stringByAppendingPathComponent:@"capture.png"];
    [manager copyItemAtPath:capture.screenshotFile toPath:screenshot error:nil];
    [self recordScreenshot:@"capture" path:screenshot];
    [self showSurfaceImage:[UIImage imageWithContentsOfFile:screenshot]];
    [self updateManifest:^(NSMutableDictionary* manifest) {
        manifest[@"capture_id"] = capture.captureId;
        manifest[@"auto_run_to_end"] = @YES;
        manifest[@"stage"] = @"extract";
        manifest[@"capture_source"] = source;
        manifest[@"preset"] = self->_preset;
        manifest[@"optimization_config"] = [self->_optimizationConfig mutableCopy];
        manifest[@"frame_target_count"] = @(self->_frameTargetCount);
        manifest[@"pipeline_config"] = [self pipelineConfig];
        if (result != nil) {
            NSMutableDictionary* times = [manifest[@"stage_times_ms"] mutableCopy];
            times[@"capture"] = @(result.durationMs);
            manifest[@"stage_times_ms"] = times;
            manifest[@"capture_pose_samples"] = @(result.poseCount);
            manifest[@"capture_sensor_samples"] = @(result.sensorCount);
        }
        manifest[@"last_error"] = NSNull.null;
    }];
    [self appendLog:[NSString stringWithFormat:@"Run %@ from capture %@", _workspace.root.lastPathComponent,
                                               capture.captureId]];
    [self refresh];
    [self extractFrames];
}

#pragma mark - Stages

- (void)runStage:(void (^)(void))body {
    _activeStages++;
    dispatch_async(_stageQueue, ^{
        @autoreleasepool {
            body();
        }
        self->_activeStages--;
    });
}

- (void)extractFrames {
    if (![self.stage isEqualToString:@"extract"]) return;
    [E2eTelemetry setStage:@"extract"];
    [self setBusy:YES message:@"Preprocessing: Extracting sharp frames"];
    E2eWorkspace* workspace = _workspace;
    NSInteger target;
    @synchronized(self) {
        target = [_manifest[@"frame_target_count"] integerValue];
    }
    [self runStage:^{
        const long long started = NowMs();
        BOOL succeeded = NO;
        @try {
            [workspace requireEmpty:workspace.sourceImages label:@"frame extraction"];
            NSError* error = nil;
            NSString* poses = [NSFileManager.defaultManager fileExistsAtPath:workspace.rawPosesFile]
                                      ? workspace.rawPosesFile : nil;
            E2eExtractionResult* result = [E2eFrameExtractor
                    extractVideo:workspace.videoFile
                           toDir:workspace.sourceImages
                    targetFrames:target
                       posesPath:poses
                      priorsPath:workspace.framePriorsFile
                        progress:^(NSInteger done, NSInteger total) {
                            // The frame this window just picked (frame_%04ld.jpg of
                            // window done - 1), if it picked one.
                            NSString* picked = [workspace.sourceImages
                                    stringByAppendingPathComponent:[NSString stringWithFormat:@"frame_%04ld.jpg",
                                                                                              (long)done - 1]];
                            dispatch_async(dispatch_get_main_queue(), ^{
                                self->_progress.progress = (float)done / total;
                                self->_status.text = [NSString stringWithFormat:@"Preprocessing: Extracting sharp frames %ld/%ld",
                                                                                (long)done, (long)total];
                                UIImage* frame = [UIImage imageWithContentsOfFile:picked];
                                if (frame != nil) [self showSurfaceImage:frame];
                            });
                        }
                           error:&error];
            if (result == nil) [NSException raise:@"extract" format:@"%@", error.localizedDescription];
            if (result.written < 8) [NSException raise:@"extract" format:@"only %ld frames extracted", (long)result.written];
            [self updateManifest:^(NSMutableDictionary* manifest) {
                manifest[@"stage"] = @"sfm";
                manifest[@"extracted_frames"] = @(result.written);
                manifest[@"position_prior_count"] = @(result.positionPriors);
                manifest[@"gravity_prior_count"] = @(result.gravityPriors);
                manifest[@"intrinsics_prior_present"] = @(result.intrinsicsPrior);
                NSMutableDictionary* times = [manifest[@"stage_times_ms"] mutableCopy];
                times[@"extract"] = @(NowMs() - started);
                manifest[@"stage_times_ms"] = times;
            }];
            [self appendLog:[NSString stringWithFormat:@"Extracted %ld frames, %ld position priors, %ld gravity, intrinsics %@",
                                                       (long)result.written, (long)result.positionPriors,
                                                       (long)result.gravityPriors, result.intrinsicsPrior ? @"yes" : @"no"]];
            succeeded = YES;
        } @catch (NSException* exception) {
            [self recordFailure:exception.reason ?: exception.name];
        }
        [self afterStage:@"extract" succeeded:succeeded next:^{ [self runSfm]; }];
    }];
}

- (NSString*)middleSourceImage {
    NSArray* names = [[[NSFileManager.defaultManager contentsOfDirectoryAtPath:_workspace.sourceImages error:nil]
            filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"self ENDSWITH[c] '.jpg'"]]
            sortedArrayUsingSelector:@selector(compare:)];
    if (names.count == 0) return nil;
    return [_workspace.sourceImages stringByAppendingPathComponent:names[names.count / 2]];
}

// Records the stage screenshot, then continues to next when the run is
// auto-running and the stage succeeded.
- (void)afterStage:(NSString*)stage succeeded:(BOOL)succeeded next:(void (^)(void))next {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self refresh];
        if (!succeeded) {
            [self setBusy:NO message:nil];
            return;
        }
        BOOL autoRun;
        @synchronized(self) {
            autoRun = [self->_manifest[@"auto_run_to_end"] boolValue];
        }
        if ([stage isEqualToString:@"optimize"] && self->_trainingShot == kTrainingShotDone) {
            // Recorded during training (captureTrainingScreenshotAt:).
            if (autoRun) next(); else [self setBusy:NO message:nil];
            return;
        }
        // The screenshot records what the surface shows: the middle extracted
        // frame (extract), the last SfM preview frame (sfm), the trainer's
        // progress render (optimize). The files stand in only for an empty
        // surface.
        NSString* fallback = nil;
        if ([stage isEqualToString:@"extract"]) {
            // Extraction draws nothing, so the extract screen shows the middle
            // frame.
            fallback = [self middleSourceImage];
            if (fallback != nil) [self showSurfaceImage:[UIImage imageWithContentsOfFile:fallback]];
        } else if ([stage isEqualToString:@"sfm"]) {
            fallback = [self middleSourceImage];
        } else if ([stage isEqualToString:@"optimize"]) {
            fallback = [self->_workspace.output stringByAppendingPathComponent:@"render/view_000.png"];
        }
        // The status line stays as the stage last set it (extraction, the SfM
        // preview caption, the last progress line), so the screenshot shows
        // the stage at work rather than a bookkeeping message.
        [self setBusy:YES message:nil];
        [self captureStageScreenshot:stage surfaceImage:fallback completion:^(BOOL ok) {
            if (ok && autoRun) {
                next();
            } else {
                [self setBusy:NO message:nil];
            }
        }];
    });
}

// Main thread. Puts each caption in the corner of its panel, as the surface
// fits the image.
- (void)placeTrainingLabels {
    const CGSize bounds = _surface.bounds.size;
    if (_trainingLayout.width == 0 || bounds.width <= 0 || bounds.height <= 0) return;
    const CGFloat scale = MIN(bounds.width / _trainingLayout.width, bounds.height / _trainingLayout.height);
    const CGFloat left = (bounds.width - _trainingLayout.width * scale) / 2;
    const CGFloat top = (bounds.height - _trainingLayout.height * scale) / 2;
    const CGFloat margin = 4, paddingX = 6, paddingY = 2;
    for (NSUInteger index = 0; index < _trainingLabels.count; index++) {
        const auto& panel = _trainingLayout.panels[index];
        UILabel* label = _trainingLabels[index];
        const CGFloat room = panel.width * scale - 2 * (margin + paddingX);
        const CGSize text = [label sizeThatFits:CGSizeMake(room, CGFLOAT_MAX)];
        label.frame = CGRectMake(left + panel.x * scale + margin, top + panel.y * scale + margin,
                                 MIN(text.width, room) + 2 * paddingX, text.height + 2 * paddingY);
        label.hidden = NO;
    }
}

- (void)hideTrainingLabels {
    for (UILabel* label in _trainingLabels) label.hidden = YES;
}

// Main thread. The optimize screenshot shows training in progress: the first
// progress report past 90% of the updates, with the trainer's render of that
// report, its status line and the progress bar. A failed attempt is not a run
// failure; afterStage then takes the screenshot after training.
- (void)captureTrainingScreenshotAt:(int)iteration total:(int)total {
    if (_trainingShot != kTrainingShotNone || total <= 0 || iteration * 10LL < total * 9LL) return;
    _trainingShot = kTrainingShotPending;
    [self captureStageScreenshot:@"optimize" surfaceImage:nil required:NO completion:^(BOOL ok) {
        if (!ok) [self appendLog:@"Optimize screenshot during training failed; taking it after training"];
        self->_trainingShot = ok ? kTrainingShotDone : kTrainingShotFailed;
        void (^next)(void) = self->_afterTrainingShot;
        self->_afterTrainingShot = nil;
        if (next != nil) next();
    }];
}

- (void)captureStageScreenshot:(NSString*)stage
                  surfaceImage:(nullable NSString*)surfacePath
                    completion:(void (^)(BOOL ok))completion {
    [self captureStageScreenshot:stage surfaceImage:surfacePath required:YES completion:completion];
}

// Main thread. Full-window screenshot; the surface must show non-constant
// content (Android rejects black SurfaceView evidence the same way). A failed
// required screenshot fails the run.
- (void)captureStageScreenshot:(NSString*)stage
                  surfaceImage:(nullable NSString*)surfacePath
                      required:(BOOL)required
                    completion:(void (^)(BOOL ok))completion {
    if (surfacePath != nil && (_surface.image == nil || E2eSharpness(_surface.image.CGImage) <= 0.0)) {
        UIImage* image = [UIImage imageWithContentsOfFile:surfacePath];
        if (image != nil) _surface.image = image;
    }
    UIImage* surface = _surface.image;
    const BOOL requireSurface = surfacePath != nil || [stage isEqualToString:@"optimize"] ||
                                [stage isEqualToString:@"view"] || [OptionalScreenshotStages() containsObject:stage];
    if (requireSurface && (surface == nil || E2eSharpness(surface.CGImage) <= 0.0)) {
        if (required) {
            [self recordFailure:[NSString stringWithFormat:@"automatic screenshot has no surface content for stage=%@", stage]];
            [self refresh];
        }
        completion(NO);
        return;
    }
    // Let the new surface image and status reach the screen first.
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 250 * NSEC_PER_MSEC), dispatch_get_main_queue(), ^{
        UIView* root = self.view.window ?: self.view;
        UIGraphicsImageRendererFormat* format = [UIGraphicsImageRendererFormat preferredFormat];
        format.opaque = YES;
        UIImage* shot = [[[UIGraphicsImageRenderer alloc] initWithBounds:root.bounds format:format]
                imageWithActions:^(UIGraphicsImageRendererContext*) {
                    [root drawViewHierarchyInRect:root.bounds afterScreenUpdates:YES];
                }];
        NSString* target = [self->_workspace.screenshotsDir
                stringByAppendingPathComponent:[E2eUniqueId(stage) stringByAppendingString:@".png"]];
        const BOOL written = ![NSFileManager.defaultManager fileExistsAtPath:target] && E2eWritePng(shot.CGImage, target);
        if (!written) {
            if (required) {
                [self recordFailure:[NSString stringWithFormat:@"automatic screenshot failed for stage=%@", stage]];
                [self refresh];
            }
            completion(NO);
            return;
        }
        [self recordScreenshot:stage path:target];
        [self appendLog:[@"Automatic screenshot saved: " stringByAppendingString:[self->_workspace relative:target]]];
        completion(YES);
    });
}

- (void)onSfmLogLine:(NSString*)line {
    const long long now = NowMs();
    NSString* next = nil;
    if ([line containsString:@"=== Feature extraction ==="]) next = @"sfm_feature_extraction";
    else if ([line containsString:@"=== Feature matching"]) next = @"sfm_feature_matching";
    else if ([line containsString:@"=== Incremental mapping ==="] || [line containsString:@"=== Global mapping ==="])
        next = @"sfm_mapping";
    else if ([line containsString:@"=== Done:"]) next = @"done";
    if (next != nil) {
        [_sfmTimingLock lock];
        if (_sfmSubstage != nil) _sfmStageTimes[_sfmSubstage] = @(now - _sfmSubstageStartedMs);
        if ([next isEqualToString:@"done"]) {
            _sfmSubstage = nil;
        } else {
            _sfmSubstage = next;
            _sfmSubstageStartedMs = now;
        }
        [_sfmTimingLock unlock];
    }
    [self appendLog:line];
}

static void SfmLog(const char* line, void* user) {
    E2eViewController* controller = (__bridge E2eViewController*)user;
    [controller onSfmLogLine:[NSString stringWithUTF8String:line] ?: @""];
}

// Main thread. Saves the optional SfM screenshots: the SIFT keypoints and the
// pair matrix, each at the first preview frame at least half way through its
// step. A failed copy is only logged.
- (void)capturePreviewScreenshotIfDue:(int)stage caption:(NSString*)caption {
    NSString* shot = stage == CROSS3DGS_SFM_STAGE_FEATURES   ? kSiftScreenshotStage
                     : stage == CROSS3DGS_SFM_STAGE_MATCHING ? kMatchingScreenshotStage
                                                             : nil;
    if (shot == nil || [_previewShotsRequested containsObject:shot]) return;
    // The captions start "SIFT done/total" and "Matching done/total pairs",
    // with thousands separators (sfm_preview.cc).
    const char* plain = [caption stringByReplacingOccurrencesOfString:@"," withString:@""].UTF8String;
    long long done = 0, total = 0;
    const int parsed = stage == CROSS3DGS_SFM_STAGE_FEATURES ? sscanf(plain, "SIFT %lld/%lld", &done, &total)
                                                             : sscanf(plain, "Matching %lld/%lld", &done, &total);
    if (parsed != 2 || total <= 0 || done * 2 < total) return;
    [_previewShotsRequested addObject:shot];
    [self captureStageScreenshot:shot surfaceImage:nil required:NO completion:^(BOOL ok) {
        if (!ok) [self appendLog:[shot stringByAppendingString:@" screenshot failed; continuing"]];
    }];
}

// The SfM library's live preview (SIFT keypoints, pair matrix, growing sparse
// model); called on its preview thread. Display only: the preview never
// changes the SfM result, and the sfm screenshot records its last frame. The preview thread has no run loop, so without the
// pool every frame's autoreleased RGBA copy would live until SfM returns.
static void SfmPreview(const cross3dgs_sfm_preview* frame, void* user) {
    @autoreleasepool {
        E2eViewController* controller = (__bridge E2eViewController*)user;
        UIImage* image = ImageFromRgbaBytes(frame->rgba, static_cast<size_t>(frame->width),
                                            static_cast<size_t>(frame->height));
        NSString* caption = [NSString stringWithUTF8String:frame->caption] ?: @"";
        const int stage = frame->stage;
        dispatch_async(dispatch_get_main_queue(), ^{
            controller->_status.text = SfmStatus(caption);
            [controller showSurfaceImage:image];
            [controller capturePreviewScreenshotIfDue:stage caption:caption];
        });
    }
}

- (void)runSfm {
    if (![self.stage isEqualToString:@"sfm"]) return;
    [E2eTelemetry setStage:@"sfm"];
    [self setBusy:YES message:@"SfM: Running incremental reconstruction"];
    _previewShotsRequested = [NSMutableSet set];
    E2eWorkspace* workspace = _workspace;
    // Preview size: the surface in pixels, longest side capped like the viewer.
    const CGSize points = _surface.bounds.size;
    const CGFloat pixelScale = _surface.window.screen.scale ?: UIScreen.mainScreen.scale;
    const CGFloat longest = MAX(points.width, points.height) * pixelScale;
    const CGFloat previewScale = longest > 0 ? MIN(pixelScale, 960.0 / MAX(points.width, points.height)) : 0;
    const int previewWidth = longest > 0 ? (int)lround(points.width * previewScale) : 960;
    const int previewHeight = longest > 0 ? (int)lround(points.height * previewScale) : 540;
    // The run's frozen configuration decides, not this launch's arguments.
    BOOL preview = YES;
    @synchronized(self) {
        preview = ![_manifest[@"pipeline_config"][@"sfm_live_preview"] isEqual:@"disabled"];
    }
    [self runStage:^{
        const long long started = NowMs();
        [self->_sfmTimingLock lock];
        [self->_sfmStageTimes removeAllObjects];
        self->_sfmSubstage = nil;
        [self->_sfmTimingLock unlock];
        BOOL succeeded = NO;
        @try {
            if (!E2eDirectoryHasFiles(workspace.sourceImages)) [NSException raise:@"sfm" format:@"no extracted source images"];
            [workspace requireEmpty:workspace.sfmWorkspace label:@"SfM"];
            [NSFileManager.defaultManager createDirectoryAtPath:workspace.sfmSparse withIntermediateDirectories:YES
                                                     attributes:nil error:nil];
            NSString* priors = [NSFileManager.defaultManager fileExistsAtPath:workspace.framePriorsFile]
                                       ? workspace.framePriorsFile : nil;
            [self appendLog:priors ? @"Running incremental COLMAP with ARKit priors" : @"Running incremental COLMAP"];
            // Without an AR intrinsics prior COLMAP would pick SIMPLE_RADIAL,
            // which the optimizer rejects; ask for PINHOLE instead.
            if (preview) {
                cross3dgs_sfm_set_preview(SfmPreview, (__bridge void*)self, previewWidth, previewHeight);
            }
            const int exit = cross3dgs_sfm_run(workspace.sourceImages.fileSystemRepresentation,
                                               workspace.sfmWorkspace.fileSystemRepresentation,
                                               priors ? priors.fileSystemRepresentation : nullptr,
                                               priors ? nullptr : "PINHOLE", 0, 1, SfmLog,
                                               (__bridge void*)self);
            // No preview frame arrives after the run returns.
            cross3dgs_sfm_set_preview(nullptr, nullptr, 0, 0);
            if (exit == CROSS3DGS_SFM_CANCELED) [NSException raise:@"sfm" format:@"SfM canceled"];
            if (exit != CROSS3DGS_SFM_OK) [NSException raise:@"sfm" format:@"COLMAP failed with exit code %d", exit];
            NSString* selected = [self selectLargestSparseModel:workspace.sfmSparse];
            const NSInteger registered = [self prepareTrainingScene:selected workspace:workspace];
            const auto meanError = cross3dgs_ios::ColmapMeanReprojectionError(
                    [selected stringByAppendingPathComponent:@"points3D.bin"].fileSystemRepresentation);
            [self->_sfmTimingLock lock];
            NSDictionary* substages = [self->_sfmStageTimes copy];
            [self->_sfmTimingLock unlock];
            [self updateManifest:^(NSMutableDictionary* manifest) {
                manifest[@"stage"] = @"optimize";
                manifest[@"selected_sfm_model"] = selected;
                manifest[@"registered_images"] = @(registered);
                manifest[@"mean_reprojection_error"] = meanError ? (id)@(*meanError) : NSNull.null;
                NSMutableDictionary* times = [manifest[@"stage_times_ms"] mutableCopy];
                [times addEntriesFromDictionary:substages];
                times[@"sfm_total"] = @(NowMs() - started);
                manifest[@"stage_times_ms"] = times;
                manifest[@"last_error"] = NSNull.null;
            }];
            [self appendLog:[NSString stringWithFormat:@"SfM registered %ld images, mean reprojection error %@",
                                                       (long)registered, meanError ? @(*meanError) : @"n/a"]];
            succeeded = YES;
        } @catch (NSException* exception) {
            [self recordFailure:exception.reason ?: exception.name];
        }
        [self afterStage:@"sfm" succeeded:succeeded next:^{ [self runOptimization]; }];
    }];
}

- (NSString*)selectLargestSparseModel:(NSString*)root {
    NSString* best = nil;
    unsigned long long bestSize = 0;
    for (NSString* name in [NSFileManager.defaultManager contentsOfDirectoryAtPath:root error:nil]) {
        NSString* images = [[root stringByAppendingPathComponent:name] stringByAppendingPathComponent:@"images.bin"];
        const unsigned long long size = [[NSFileManager.defaultManager attributesOfItemAtPath:images error:nil] fileSize];
        if (size > bestSize) {
            bestSize = size;
            best = [root stringByAppendingPathComponent:name];
        }
    }
    if (best == nil) [NSException raise:@"sfm" format:@"COLMAP produced no sparse model"];
    return best;
}

- (NSInteger)prepareTrainingScene:(NSString*)selected workspace:(E2eWorkspace*)workspace {
    std::vector<std::string> names;
    try {
        names = cross3dgs_ios::ColmapRegisteredImageNames(
                [selected stringByAppendingPathComponent:@"images.bin"].fileSystemRepresentation);
    } catch (const std::exception& error) {
        [NSException raise:@"sfm" format:@"%s", error.what()];
    }
    if (names.size() < 8) [NSException raise:@"sfm" format:@"SfM registered only %zu images", names.size()];
    [workspace requireEmpty:workspace.trainingRoot label:@"training scene"];
    NSFileManager* manager = NSFileManager.defaultManager;
    [manager createDirectoryAtPath:workspace.trainingSparse withIntermediateDirectories:YES attributes:nil error:nil];
    [manager createDirectoryAtPath:workspace.trainingImages withIntermediateDirectories:YES attributes:nil error:nil];
    NSError* error = nil;
    for (NSString* file in @[ @"cameras.bin", @"images.bin", @"points3D.bin" ]) {
        if (![manager copyItemAtPath:[selected stringByAppendingPathComponent:file]
                              toPath:[workspace.trainingSparse stringByAppendingPathComponent:file]
                               error:&error]) {
            [NSException raise:@"sfm" format:@"%@", error.localizedDescription];
        }
    }
    for (const auto& name : names) {
        NSString* imageName = [NSString stringWithUTF8String:name.c_str()];
        NSString* source = [workspace.sourceImages stringByAppendingPathComponent:imageName];
        NSString* target = [workspace.trainingImages stringByAppendingPathComponent:imageName];
        [manager createDirectoryAtPath:target.stringByDeletingLastPathComponent withIntermediateDirectories:YES
                            attributes:nil error:nil];
        if (![manager copyItemAtPath:source toPath:target error:&error]) {
            [NSException raise:@"sfm" format:@"registered image missing: %@", imageName];
        }
    }
    return static_cast<NSInteger>(names.size());
}

- (void)runOptimization {
    if (![self.stage isEqualToString:@"optimize"]) return;
    [E2eTelemetry setStage:@"optimize"];
    @try {
        [_workspace requireEmpty:_workspace.output label:@"optimization"];
    } @catch (NSException* exception) {
        [self recordFailure:exception.reason];
        [self refresh];
        return;
    }
    [self updateManifest:^(NSMutableDictionary* manifest) {
        manifest[@"preset"] = self->_preset;
        manifest[@"optimization_config"] = [self->_optimizationConfig mutableCopy];
        manifest[@"pipeline_config"] = [self pipelineConfig];
        manifest[@"last_error"] = NSNull.null;
    }];
    auto& engine = cross3dgs_ios::Engine::instance();
    if (!engine.setTileBinCapacity(kTileBinCapacity)) {
        [self recordFailure:@"failed to set the tile-Gaussian pair budget"];
        [self refresh];
        return;
    }
    cross3dgs_ios::TrainingRequest request;
    request.scene_dir = Std(_workspace.root);
    request.output_dir = Std(_workspace.output);
    request.iterations = static_cast<uint32_t>([_optimizationConfig[@"iterations"] integerValue]);
    request.gaussian_capacity = static_cast<uint32_t>([_optimizationConfig[@"gaussian_capacity"] integerValue]);
    request.sh_degree = static_cast<uint32_t>([_optimizationConfig[@"sh_degree"] integerValue]);
    request.density_strategy = Std(_optimizationConfig[@"density_strategy"]);
    request.full_image_ssim = E2ePreset.fullImageSsim;
    request.mcmc_clone_offset = E2ePreset.mcmcCloneOffset;
    request.mcmc_gradient_sampling = E2ePreset.mcmcGradientSampling;
    _optimizationStartedMs = NowMs();
    _trainingShot = kTrainingShotNone;
    _afterTrainingShot = nil;
    _progress.progress = 0;
    if (!engine.startTraining(request)) {
        [self recordFailure:@"optimizer already running or request invalid"];
        [self refresh];
        return;
    }
    _activeStages++;
    [self setBusy:YES message:[NSString stringWithFormat:@"3DGS optimization: %u updates / %u max Gaussians",
                                                         request.iterations, request.gaussian_capacity]];
}

- (void)installEngineCallbacks {
    __weak E2eViewController* weakSelf = self;
    cross3dgs_ios::EngineCallbacks callbacks;
    callbacks.log = [weakSelf](const std::string& line) {
        [weakSelf appendLog:[NSString stringWithUTF8String:line.c_str()] ?: @""];
    };
    callbacks.progress = [weakSelf](int iteration, int total, float loss, double seconds) {
        dispatch_async(dispatch_get_main_queue(), ^{
            E2eViewController* strongSelf = weakSelf;
            if (strongSelf == nil) return;
            strongSelf->_progress.progress = total > 0 ? (float)iteration / total : 0;
            strongSelf->_status.text = [NSString stringWithFormat:@"3DGS optimization: %d/%d | loss %.6f | %.1f s", iteration, total, loss, seconds];
            [strongSelf placeTrainingLabels];
            [strongSelf captureTrainingScreenshotAt:iteration total:total];
        });
    };
    callbacks.frame = [weakSelf](const cross3dgs_ios::RgbaImage& image) {
        [weakSelf showSurfaceImage:ImageFromRgba(image)];
    };
    callbacks.training_preview_layout = [weakSelf](const cross3dgs::TrainingPreviewLayout& layout) {
        dispatch_async(dispatch_get_main_queue(), ^{
            E2eViewController* strongSelf = weakSelf;
            if (strongSelf != nil) strongSelf->_trainingLayout = layout;
        });
    };
    callbacks.finished = [weakSelf](int exitCode, const std::string& message, const std::string&) {
        NSString* text = [NSString stringWithUTF8String:message.c_str()] ?: @"";
        dispatch_async(dispatch_get_main_queue(), ^{
            if (exitCode != 0) [weakSelf hideTrainingLabels];
            [weakSelf onTrainingFinished:exitCode message:text];
        });
    };
    cross3dgs_ios::Engine::instance().setCallbacks(std::move(callbacks));
}

// Dropped tile-Gaussian pairs from bench.json, or -1 when unreadable.
- (long long)droppedPairs {
    NSData* data = [NSData dataWithContentsOfFile:[_workspace.output stringByAppendingPathComponent:@"bench.json"]];
    if (data == nil) return -1;
    NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    NSNumber* overflow = json[@"overflow_count"];
    return [overflow isKindOfClass:NSNumber.class] ? overflow.longLongValue : -1;
}

- (void)onTrainingFinished:(int)exitCode message:(NSString*)message {
    _activeStages--;
    const long long dropped = exitCode == 0 ? [self droppedPairs] : 0;
    NSString* verdict = message;
    if (dropped > 0) {
        verdict = [NSString stringWithFormat:@"%@; run rejected: the rasterizer dropped %lld tile-Gaussian pairs, so the "
                                             @"reconstruction is incomplete. Retry with a smaller Gaussian capacity.",
                                             message, dropped];
    } else if (dropped < 0) {
        verdict = [message stringByAppendingString:@"; run rejected: bench.json carries no overflow_count"];
    }
    const int code = dropped != 0 ? 1 : exitCode;
    if (code == 0) {
        [self updateManifest:^(NSMutableDictionary* manifest) {
            manifest[@"stage"] = @"view";
            NSMutableDictionary* times = [manifest[@"stage_times_ms"] mutableCopy];
            times[@"optimize_total"] = @(NowMs() - self->_optimizationStartedMs);
            manifest[@"stage_times_ms"] = times;
            manifest[@"last_error"] = NSNull.null;
        }];
        [self appendLog:verdict];
        void (^after)(void) = ^{
            [self afterStage:@"optimize" succeeded:YES next:^{ [self prepareViewerAndRender]; }];
        };
        if (_trainingShot == kTrainingShotPending) {
            _afterTrainingShot = after;
        } else {
            after();
        }
    } else {
        [self recordFailure:verdict];
        [self setBusy:NO message:nil];
    }
}

- (void)cancelStage {
    cross3dgs_ios::Engine::instance().requestStop();
    cross3dgs_sfm_cancel();
    [self appendLog:@"Cancellation requested"];
}

#pragma mark - Viewer

- (void)prepareViewerAndRender {
    [self hideTrainingLabels];
    NSString* stage = self.stage;
    if (!([stage isEqualToString:@"view"] || [stage isEqualToString:@"complete"])) return;
    const BOOL requiresEvidence = [stage isEqualToString:@"view"];
    [self setBusy:YES message:@"Viewer: Preparing free-view renderer"];
    if (requiresEvidence) [E2eTelemetry setStage:@"view"];
    _viewerStartedMs = NowMs();
    E2eWorkspace* workspace = _workspace;
    NSDictionary* manifest;
    @synchronized(self) {
        manifest = [_manifest copy];
    }
    NSString* reference = [NSUserDefaults.standardUserDefaults stringForKey:@"viewer_reference_image"] ?: @"";
    dispatch_async(_viewerQueue, ^{
        const bool ready = cross3dgs_ios::Engine::instance().prepareViewer(
                Std(workspace.root), Std(manifest[@"image_directory"]), Std(manifest[@"sparse_directory"]),
                Std([workspace.output stringByAppendingPathComponent:@"checkpoint.d3gs"]), Std(reference),
                [NSFileManager.defaultManager fileExistsAtPath:workspace.framePriorsFile]
                        ? Std(workspace.framePriorsFile) : std::string(),
                // Only the first view: a finished run is sealed.
                requiresEvidence
                        ? Std([workspace.output stringByAppendingPathComponent:@"splat_gravity_up.ply"])
                        : std::string(),
                static_cast<uint32_t>([manifest[@"optimization_config"][@"sh_degree"] integerValue]));
        dispatch_async(dispatch_get_main_queue(), ^{
            self->_viewerReady = ready;
            if (!ready) {
                if (requiresEvidence) [self recordFailure:@"viewer preparation failed"];
                [self setBusy:NO message:@"Viewer preparation failed"];
                return;
            }
            float min_pitch = 0, max_pitch = 0;
            cross3dgs_ios::Engine::instance().orbitPitchRange(&min_pitch, &max_pitch);
            [self->_orbitLock lock];
            self->_orbitMinPitch = min_pitch;
            self->_orbitMaxPitch = max_pitch;
            self->_orbitPitch = std::clamp(self->_orbitPitch, min_pitch, max_pitch);
            [self->_orbitLock unlock];
            [self appendLog:@"Viewer ready: drag=rotate, pinch=zoom, two fingers=pan"];
            if (requiresEvidence) {
                [self setStageTime:@"view_prepare" ms:NowMs() - self->_viewerStartedMs];
                [self renderOrbitSync:^(BOOL rendered) {
                    if (!rendered) {
                        [self recordFailure:@"first viewer render failed"];
                        [self setBusy:NO message:nil];
                        return;
                    }
                    self->_status.text = @"Viewer: Free-view rendering\nDrag = rotate · pinch = zoom · 2 fingers = pan";
                    [self captureStageScreenshot:@"view" surfaceImage:nil completion:^(BOOL ok) {
                        if (!ok) {
                            [self setBusy:NO message:nil];
                            return;
                        }
                        [self setStageTime:@"view_total" ms:NowMs() - self->_viewerStartedMs];
                        [self measureViewerFrames:^{ [self finalizeCompletedRun]; }];
                    }];
                }];
            } else {
                [self setBusy:NO message:nil];
                [self requestOrbitRender];
            }
        });
    });
}

- (void)renderOrbitSync:(void (^)(BOOL rendered))completion {
    const CGSize size = _surfaceSize;
    std::array<float, 5> values{};
    [_orbitLock lock];
    values[0] = _orbitYaw; values[1] = _orbitPitch; values[2] = _orbitZoom; values[3] = _orbitPanX; values[4] = _orbitPanY;
    [_orbitLock unlock];
    dispatch_async(_viewerQueue, ^{
        const bool rendered = cross3dgs_ios::Engine::instance().renderOrbit(
                values[0], values[1], values[2], values[3], values[4], (uint32_t)size.width, (uint32_t)size.height);
        dispatch_async(dispatch_get_main_queue(), ^{
            completion(rendered);
        });
    });
}

- (void)requestOrbitRender {
    if (!_viewerReady || _surfaceSize.width <= 0 || _surfaceSize.height <= 0) return;
    _viewerDirty = true;
    bool expected = false;
    if (!_viewerPending.compare_exchange_strong(expected, true)) return;
    const CGSize size = _surfaceSize;
    dispatch_async(_viewerQueue, ^{
        while (self->_viewerDirty.exchange(false)) {
            float values[5];
            [self->_orbitLock lock];
            values[0] = self->_orbitYaw; values[1] = self->_orbitPitch; values[2] = self->_orbitZoom;
            values[3] = self->_orbitPanX; values[4] = self->_orbitPanY;
            [self->_orbitLock unlock];
            cross3dgs_ios::Engine::instance().renderOrbit(values[0], values[1], values[2], values[3], values[4],
                                                          (uint32_t)size.width, (uint32_t)size.height);
        }
        self->_viewerPending = false;
        if (self->_viewerDirty.load()) {
            dispatch_async(dispatch_get_main_queue(), ^{ [self requestOrbitRender]; });
        }
    });
}

- (BOOL)gestureRecognizer:(UIGestureRecognizer*)gesture
        shouldRecognizeSimultaneouslyWithGestureRecognizer:(UIGestureRecognizer*)other {
    return YES;
}

- (void)handlePan:(UIPanGestureRecognizer*)pan {
    if (!_viewerReady) return;
    const CGPoint location = [pan locationInView:_surface];
    const NSInteger touches = pan.numberOfTouches;
    if (pan.state == UIGestureRecognizerStateBegan || touches != _lastPanTouches) {
        // A finger landing or lifting re-bases the gesture so the view does
        // not jump by the centroid shift.
        _lastPan = location;
        _lastPanTouches = touches;
        return;
    }
    if (pan.state != UIGestureRecognizerStateChanged) return;
    const CGFloat dx = location.x - _lastPan.x;
    const CGFloat dy = location.y - _lastPan.y;
    _lastPan = location;
    const CGFloat width = MAX(_surface.bounds.size.width, 1);
    const CGFloat height = MAX(_surface.bounds.size.height, 1);
    [_orbitLock lock];
    if (touches <= 1) {
        _orbitYaw += dx / width * kOrbitTurnPerSweep;
        _orbitPitch = std::clamp<float>(_orbitPitch - dy / height * kOrbitTurnPerSweep, _orbitMinPitch,
                                        _orbitMaxPitch);
    } else {
        _orbitPanX -= dx / width * _orbitZoom;
        _orbitPanY -= dy / height * _orbitZoom;
    }
    [_orbitLock unlock];
    [self requestOrbitRender];
}

- (void)handlePinch:(UIPinchGestureRecognizer*)pinch {
    if (!_viewerReady) return;
    [_orbitLock lock];
    if (pinch.state == UIGestureRecognizerStateBegan) _pinchStartZoom = _orbitZoom;
    if (pinch.scale > 0.01) _orbitZoom = std::clamp<float>(_pinchStartZoom / pinch.scale, 0.2f, 5.0f);
    [_orbitLock unlock];
    [self requestOrbitRender];
}

- (void)resetView {
    [_orbitLock lock];
    _orbitYaw = _orbitPitch = _orbitPanX = _orbitPanY = 0;
    _orbitZoom = 1;
    [_orbitLock unlock];
    [self requestOrbitRender];
}

// Deterministic bounded yaw sweep timing one renderOrbit host call per frame
// (Android measureViewerFrames), then restores the evidence pose.
- (void)measureViewerFrames:(void (^)(void))done {
    [self setBusy:YES message:@"Viewer: Measuring frame time"];
    const CGSize size = _surfaceSize;
    dispatch_async(_viewerQueue, ^{
        NSMutableArray<NSNumber*>* samples = [NSMutableArray array];
        auto& engine = cross3dgs_ios::Engine::instance();
        for (int frame = 0; frame < kViewerSweepFrames; frame++) {
            const float yaw = kViewerSweepYawSpan * (static_cast<float>(frame) / (kViewerSweepFrames - 1) - 0.5f);
            const CFTimeInterval started = CACurrentMediaTime();
            if (engine.renderOrbit(yaw, 0, 1, 0, 0, (uint32_t)size.width, (uint32_t)size.height)) {
                [samples addObject:@((long long)((CACurrentMediaTime() - started) * 1e6))];
            }
        }
        engine.renderOrbit(0, 0, 1, 0, 0, (uint32_t)size.width, (uint32_t)size.height);
        dispatch_async(dispatch_get_main_queue(), ^{
            if (samples.count > 0) {
                NSArray* sorted = [samples sortedArrayUsingSelector:@selector(compare:)];
                const double median = [sorted[sorted.count / 2] doubleValue] / 1000.0;
                const double p90 = [sorted[(NSUInteger)(0.9 * (sorted.count - 1))] doubleValue] / 1000.0;
                [self updateManifest:^(NSMutableDictionary* manifest) {
                    manifest[@"viewer_frame_sweep"] = @{
                        @"frame_times_us" : samples,
                        @"frames" : @(samples.count),
                        @"requested_width" : @((int)size.width),
                        @"requested_height" : @((int)size.height),
                        @"yaw_span_radians" : @(kViewerSweepYawSpan),
                        @"median_frame_ms" : @(median),
                        @"p90_frame_ms" : @(p90),
                        @"min_frame_ms" : @([sorted.firstObject doubleValue] / 1000.0),
                        @"max_frame_ms" : @([sorted.lastObject doubleValue] / 1000.0),
                        @"fps_from_median" : @(median > 0 ? 1000.0 / median : 0.0),
                        @"scope" : @"render_execute_download_color_and_rgba8_conversion_host_call",
                        @"semantics" : @"no_present_completion_guarantee_deterministic_yaw_sweep",
                    };
                }];
                [self appendLog:[NSString stringWithFormat:@"Viewer frame sweep: %lu frames, median %.1f ms",
                                                           (unsigned long)samples.count, median]];
            } else {
                [self appendLog:@"Viewer frame sweep recorded no frames"];
            }
            done();
        });
    });
}

- (void)finalizeCompletedRun {
    [self appendLog:@"Initial viewer render and screenshot verified; finalizing inventory"];
    [E2eTelemetry stop];
    [self setBusy:YES message:@"Finalizing evidence inventory"];
    E2eWorkspace* workspace = _workspace;
    dispatch_async(_stageQueue, ^{
        NSString* failure = nil;
        @synchronized(self) {
            [self->_logFile synchronizeFile];
        }
        NSDictionary* screenshots;
        @synchronized(self) {
            screenshots = [self->_manifest[@"screenshots"] copy];
        }
        // Every stage the run executed must have left a screenshot; an
        // image-folder run has no capture stage and so no capture image.
        NSMutableArray<NSString*>* required = [RequiredScreenshotStages() mutableCopy];
        if (!E2eDirectoryHasFiles(workspace.captureDir)) [required removeObject:@"capture"];
        NSSet* recorded = [NSSet setWithArray:screenshots.allKeys];
        NSSet* allowed = [NSSet setWithArray:[required arrayByAddingObjectsFromArray:OptionalScreenshotStages()]];
        if (![[NSSet setWithArray:required] isSubsetOfSet:recorded] || ![recorded isSubsetOfSet:allowed]) {
            failure = [NSString stringWithFormat:@"complete run requires screenshots for %@ and allows only %@ besides",
                                                 [required componentsJoinedByString:@", "],
                                                 [OptionalScreenshotStages() componentsJoinedByString:@", "]];
        }
        NSDictionary* inventory = failure == nil ? [workspace inventory] : nil;
        for (NSString* stage in required) {
            if (failure == nil && inventory[screenshots[stage]] == nil) {
                failure = [NSString stringWithFormat:@"%@ screenshot missing from final inventory", stage];
            }
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            if (failure != nil) {
                [self recordFailure:failure];
            } else {
                [self updateManifest:^(NSMutableDictionary* manifest) {
                    manifest[@"stage"] = @"complete";
                    manifest[@"artifact_sha256"] = inventory;
                    manifest[@"last_error"] = NSNull.null;
                }];
                [self appendLog:[NSString stringWithFormat:@"Run complete: %lu artifacts hashed",
                                                           (unsigned long)inventory.count]];
            }
            [self setBusy:NO message:nil];
        });
    });
}

#pragma mark - Finished runs / replay / share

- (NSArray<E2eWorkspace*>*)runsMatching:(BOOL (^)(E2eWorkspace*, NSDictionary*))predicate {
    NSMutableArray* runs = [NSMutableArray array];
    NSArray* names = [[NSFileManager.defaultManager contentsOfDirectoryAtPath:_runsRoot error:nil]
            sortedArrayUsingSelector:@selector(compare:)];
    for (NSString* name in names.reverseObjectEnumerator) {
        E2eWorkspace* candidate = [E2eWorkspace inspect:[_runsRoot stringByAppendingPathComponent:name]];
        if (candidate == nil || [candidate.root isEqualToString:_workspace.root]) continue;
        NSDictionary* manifest = [candidate loadManifest];
        if (manifest != nil && predicate(candidate, manifest)) [runs addObject:candidate];
    }
    return runs;
}

- (void)chooseFinishedRun {
    if ([self refuseWhileRunning]) return;
    NSArray<E2eWorkspace*>* runs = [self runsMatching:^BOOL(E2eWorkspace* run, NSDictionary* manifest) {
        return [manifest[@"stage"] isEqualToString:@"complete"] &&
               [NSFileManager.defaultManager fileExistsAtPath:[run.output stringByAppendingPathComponent:@"checkpoint.d3gs"]];
    }];
    if (runs.count == 0) {
        [self alert:@"No runs" message:@"No finished runs"];
        return;
    }
    UIAlertController* sheet = [UIAlertController alertControllerWithTitle:@"Open finished run" message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    for (E2eWorkspace* run in runs) {
        NSDictionary* manifest = [run loadManifest];
        NSString* title = [NSString stringWithFormat:@"%@ · %@ · %@ it", run.root.lastPathComponent,
                                                     [manifest[@"preset"] uppercaseString],
                                                     manifest[@"optimization_config"][@"iterations"]];
        [sheet addAction:[UIAlertAction actionWithTitle:title style:UIAlertActionStyleDefault handler:^(UIAlertAction*) {
            [self openFinishedRun:run];
        }]];
    }
    [sheet addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    sheet.popoverPresentationController.sourceView = _openFinishedButton;
    [self presentViewController:sheet animated:YES completion:nil];
}

- (void)openFinishedRun:(E2eWorkspace*)run {
    NSMutableDictionary* manifest = [run loadManifest];
    if (manifest == nil) return;
    [E2eTelemetry stop];
    _viewerReady = false;
    [self resetView];
    @synchronized(self) {
        _workspace = run;
        _manifest = manifest;
        [self openLogFile];
    }
    _preset = manifest[@"preset"];
    _frameTargetCount = [manifest[@"frame_target_count"] integerValue];
    _optimizationConfig = [manifest[@"optimization_config"] mutableCopy];
    _log.text = @"";
    [self refresh];
    [self prepareViewerAndRender];
}

- (void)chooseFrozenReplay {
    if ([self refuseWhileRunning]) return;
    NSArray<E2eWorkspace*>* runs = [self runsMatching:^BOOL(E2eWorkspace* run, NSDictionary* manifest) {
        return [manifest[@"extracted_frames"] integerValue] >= 8 && E2eDirectoryHasFiles(run.sourceImages) &&
               [manifest[@"screenshots"][@"extract"] length] > 0;
    }];
    if (runs.count == 0) {
        [self alert:@"No inputs" message:@"No frozen SfM inputs"];
        return;
    }
    UIAlertController* sheet = [UIAlertController alertControllerWithTitle:@"Select SfM input to replay" message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    for (E2eWorkspace* run in runs) {
        [sheet addAction:[UIAlertAction actionWithTitle:run.root.lastPathComponent style:UIAlertActionStyleDefault
                                                handler:^(UIAlertAction*) {
            [self beginRunFromFrozenReplay:run];
        }]];
    }
    [sheet addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    sheet.popoverPresentationController.sourceView = _frozenReplayButton;
    [self presentViewController:sheet animated:YES completion:nil];
}

// Re-runs SfM onward from another run's selected JPEGs and priors (Android
// beginRunFromFrozenReplay); the capture/extract evidence is carried over
// after verifying its recorded hashes.
- (void)beginRunFromFrozenReplay:(E2eWorkspace*)source {
    NSDictionary* sourceManifest = [source loadManifest];
    NSMutableDictionary* carried = [NSMutableDictionary dictionary];
    for (NSString* stage in @[ @"capture", @"extract" ]) {
        NSString* relative = sourceManifest[@"screenshots"][stage];
        NSString* expected = sourceManifest[@"artifact_sha256"][relative];
        if (relative == nil) continue;  // image-folder runs have no capture screenshot
        NSString* path = [source.root stringByAppendingPathComponent:relative];
        if (expected == nil || ![NSFileManager.defaultManager fileExistsAtPath:path] ||
            ![E2eSha256(path) isEqualToString:expected]) {
            [self alert:@"Replay refused" message:[NSString stringWithFormat:@"source %@ screenshot hash mismatch", stage]];
            return;
        }
        carried[stage] = path;
    }
    if ([self needsFreshRun]) [self switchToFreshRun];
    NSError* error = nil;
    if (!E2eCopyTreeNoOverwrite(source.captureDir, _workspace.captureDir, &error) ||
        !E2eCopyTreeNoOverwrite(source.sourceImages, _workspace.sourceImages, &error)) {
        [self recordFailure:error.localizedDescription ?: @"copy failed"];
        [self refresh];
        return;
    }
    if ([NSFileManager.defaultManager fileExistsAtPath:source.framePriorsFile]) {
        [NSFileManager.defaultManager copyItemAtPath:source.framePriorsFile toPath:_workspace.framePriorsFile error:nil];
    }
    for (NSString* stage in carried) {
        NSString* path = carried[stage];
        NSString* target = [_workspace.screenshotsDir stringByAppendingPathComponent:
                [stage isEqualToString:@"capture"] ? @"capture.png" : path.lastPathComponent];
        [NSFileManager.defaultManager copyItemAtPath:path toPath:target error:nil];
        [self recordScreenshot:stage path:target];
    }
    [self updateManifest:^(NSMutableDictionary* manifest) {
        manifest[@"capture_id"] = sourceManifest[@"capture_id"] ?: @"";
        manifest[@"auto_run_to_end"] = @YES;
        manifest[@"stage"] = @"sfm";
        manifest[@"preset"] = self->_preset;
        manifest[@"optimization_config"] = [self->_optimizationConfig mutableCopy];
        manifest[@"frame_target_count"] = sourceManifest[@"frame_target_count"];
        manifest[@"capture_source"] = @"saved_run_replay";
        manifest[@"extracted_frames"] = sourceManifest[@"extracted_frames"];
        manifest[@"position_prior_count"] = sourceManifest[@"position_prior_count"];
        manifest[@"gravity_prior_count"] = sourceManifest[@"gravity_prior_count"];
        manifest[@"intrinsics_prior_present"] = sourceManifest[@"intrinsics_prior_present"];
        NSMutableDictionary* config = [[self pipelineConfig] mutableCopy];
        config[@"sfm_input_replayed_from_run"] = source.root.lastPathComponent;
        manifest[@"pipeline_config"] = config;
        manifest[@"last_error"] = NSNull.null;
    }];
    [self refresh];
    [self runSfm];
}

// Development/test entry: a run whose SfM input is a folder of JPEGs instead
// of an AR capture (no priors). Recorded as capture_source
// "imported_image_folder"; the extract screenshot shows a source image.
- (void)beginRunFromImageFolder:(NSString*)folder {
    NSArray* images = [[[NSFileManager.defaultManager contentsOfDirectoryAtPath:folder error:nil]
            filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"self ENDSWITH[c] '.jpg' OR self ENDSWITH[c] '.jpeg' OR self ENDSWITH[c] '.png'"]]
            sortedArrayUsingSelector:@selector(compare:)];
    if (images.count < 8) {
        [self alert:@"Image folder" message:[NSString stringWithFormat:@"%@ holds %lu images (need 8)", folder,
                                                                       (unsigned long)images.count]];
        return;
    }
    if ([self needsFreshRun]) [self switchToFreshRun];
    for (NSString* name in images) {
        [NSFileManager.defaultManager copyItemAtPath:[folder stringByAppendingPathComponent:name]
                                              toPath:[_workspace.sourceImages stringByAppendingPathComponent:name]
                                               error:nil];
    }
    [self updateManifest:^(NSMutableDictionary* manifest) {
        manifest[@"capture_id"] = [@"image_folder:" stringByAppendingString:folder.lastPathComponent];
        manifest[@"auto_run_to_end"] = @YES;
        manifest[@"stage"] = @"sfm";
        manifest[@"capture_source"] = @"imported_image_folder";
        manifest[@"extracted_frames"] = @(images.count);
        manifest[@"preset"] = self->_preset;
        manifest[@"optimization_config"] = [self->_optimizationConfig mutableCopy];
        manifest[@"pipeline_config"] = [self pipelineConfig];
    }];
    [self appendLog:[NSString stringWithFormat:@"Run %@ from image folder %@ (%lu images)",
                                               _workspace.root.lastPathComponent, folder, (unsigned long)images.count]];
    [self refresh];
    [self setBusy:YES message:[NSString stringWithFormat:@"Preprocessing: Imported %lu images", (unsigned long)images.count]];
    [self captureStageScreenshot:@"extract" surfaceImage:[self middleSourceImage] completion:^(BOOL ok) {
        if (ok) [self runSfm]; else [self setBusy:NO message:nil];
    }];
}

- (void)sharePly {
    NSString* ply = [_workspace.output stringByAppendingPathComponent:@"splat_gravity_up.ply"];
    if (![NSFileManager.defaultManager fileExistsAtPath:ply]) {
        ply = [_workspace.output stringByAppendingPathComponent:@"splat.ply"];
    }
    if (![NSFileManager.defaultManager fileExistsAtPath:ply]) return;
    NSArray* items = @[ [NSURL fileURLWithPath:ply] ];
    NSString* bench = [_workspace.output stringByAppendingPathComponent:@"bench.json"];
    if ([NSFileManager.defaultManager fileExistsAtPath:bench]) items = [items arrayByAddingObject:[NSURL fileURLWithPath:bench]];
    UIActivityViewController* share = [[UIActivityViewController alloc] initWithActivityItems:items applicationActivities:nil];
    share.popoverPresentationController.sourceView = _shareButton;
    [self presentViewController:share animated:YES completion:nil];
}

#pragma mark - Automation

- (void)configureAutomation {
    NSUserDefaults* args = NSUserDefaults.standardUserDefaults;
    NSString* documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
    NSString* folder = [args stringForKey:@"image_folder"];
    if (folder.length > 0) {
        if (!folder.isAbsolutePath) folder = [documents stringByAppendingPathComponent:folder];
        [self beginRunFromImageFolder:folder];
        return;
    }
    NSString* capture = [args stringForKey:@"saved_capture"];
    if (capture.length > 0) {
        for (E2eCaptureRecord* record in [_captureStore listComplete]) {
            if ([record.captureId isEqualToString:capture]) {
                [self beginRunFromCapture:record source:@"saved_capture" result:nil];
                return;
            }
        }
        [self alert:@"Saved capture" message:[@"No sealed capture " stringByAppendingString:capture]];
        return;
    }
    if (![args boolForKey:@"auto_start"]) return;
    [self updateManifest:^(NSMutableDictionary* manifest) {
        manifest[@"auto_run_to_end"] = @YES;
    }];
    NSString* stage = self.stage;
    if ([stage isEqualToString:@"extract"]) [self extractFrames];
    else if ([stage isEqualToString:@"sfm"]) [self runSfm];
    else if ([stage isEqualToString:@"optimize"]) [self runOptimization];
    else if ([stage isEqualToString:@"view"] || [stage isEqualToString:@"complete"]) [self prepareViewerAndRender];
}

@end
