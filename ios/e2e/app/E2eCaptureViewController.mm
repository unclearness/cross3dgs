// ARKit capture for the iOS capture-to-view app (counterpart of
// android/e2e/.../ArCaptureActivity.kt).
//
// The ARKit camera stream (ARFrame.capturedImage, sensor orientation, so the
// file carries no rotation and the logged intrinsics match the decoded
// frames) is encoded to H.264 MP4 with AVAssetWriter while the per-frame VIO
// pose and intrinsics are logged in the same TUM layout as arcore_poses.txt.
// ARKit's camera convention (+X right, +Y up, -Z forward, world_from_camera)
// and gravity-aligned Y-up world equal ARCore's, so the downstream prior
// derivation is unchanged. CoreMotion samples share the ARFrame clock
// (seconds since boot), recorded as nanoseconds.
#import "E2eCaptureViewController.h"

#import <ARKit/ARKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMotion/CoreMotion.h>
#import <SceneKit/SceneKit.h>

#include <atomic>

#import "E2eFrameExtractor.h"

@implementation E2eCaptureResult
@end

namespace {
// Android SensorManager type ids, kept so sensors.csv stays one schema.
constexpr int kTypeAccelerometer = 1;
constexpr int kTypeMagneticField = 2;
constexpr int kTypeGyroscope = 4;
constexpr int kTypeGravity = 9;
constexpr int kTypeRotationVector = 11;
constexpr double kStandardGravity = 9.80665;
constexpr NSInteger kMinPoses = 10;
// Recorded frames before the live capture image (about half a second).
constexpr NSInteger kLiveEvidenceFrame = 15;
enum LiveEvidence : NSInteger { kLiveNone, kLivePending, kLiveDone, kLiveFailed };

long long Ns(NSTimeInterval seconds) { return llround(seconds * 1e9); }
}  // namespace

@interface E2eCaptureViewController () <ARSessionDelegate>
@end

@implementation E2eCaptureViewController {
    NSString* _captureRoot;
    void (^_completion)(E2eCaptureResult* _Nullable, NSString* _Nullable);
    NSString* _initError;
    NSString* _lastStatus;
    ARSCNView* _sceneView;
    UILabel* _status;
    UIButton* _record;
    UIButton* _cancel;
    dispatch_queue_t _frameQueue;

    // Frame-queue state.
    BOOL _recording;
    BOOL _finishing;
    AVAssetWriter* _writer;
    AVAssetWriterInput* _writerInput;
    AVAssetWriterInputPixelBufferAdaptor* _adaptor;
    long long _videoStartNs;
    NSInteger _videoFrames;
    NSInteger _droppedFrames;
    NSMutableString* _poseLines;
    NSInteger _poseCount;
    NSString* _intrinsicsLine;
    NSString* _formatLine;
    NSDate* _recordingStarted;
    NSInteger _statusCounter;
    BOOL _liveEvidenceRequested;

    // capture.png is the screen shortly after recording starts: the scene
    // view's own snapshot with the status and buttons drawn over it. Main
    // thread.
    NSInteger _liveEvidenceState;
    BOOL _liveEvidenceAbandoned;
    void (^_onLiveEvidence)(BOOL captured);

    CMMotionManager* _motion;
    NSOperationQueue* _motionQueue;
    NSFileHandle* _sensorFile;
    std::atomic<long long> _sensorCount;
    NSLock* _sensorLock;
}

- (instancetype)initWithCaptureRoot:(NSString*)captureRoot
                         completion:(void (^)(E2eCaptureResult* _Nullable, NSString* _Nullable))completion {
    if ((self = [super initWithNibName:nil bundle:nil])) {
        _captureRoot = [captureRoot copy];
        _completion = [completion copy];
        _frameQueue = dispatch_queue_create("cross3dgs.capture.frames", DISPATCH_QUEUE_SERIAL);
        _poseLines = [NSMutableString string];
        _sensorLock = [[NSLock alloc] init];
        self.modalPresentationStyle = UIModalPresentationFullScreen;
    }
    return self;
}

- (NSString*)path:(NSString*)name {
    return [_captureRoot stringByAppendingPathComponent:name];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.blackColor;
    NSArray* existing = [NSFileManager.defaultManager contentsOfDirectoryAtPath:_captureRoot error:nil];
    if (existing == nil || existing.count != 0) {
        _initError = @"capture output must be a new empty directory";
        return;
    }
    _sceneView = [[ARSCNView alloc] initWithFrame:self.view.bounds];
    _sceneView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    _sceneView.automaticallyUpdatesLighting = NO;
    _sceneView.session.delegate = self;
    _sceneView.session.delegateQueue = _frameQueue;
    [self.view addSubview:_sceneView];

    _status = [[UILabel alloc] init];
    _status.textColor = UIColor.whiteColor;
    _status.backgroundColor = [UIColor colorWithWhite:0 alpha:0.5];
    _status.font = [UIFont monospacedSystemFontOfSize:13 weight:UIFontWeightRegular];
    _status.numberOfLines = 0;
    _status.text = @"initializing";
    _status.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:_status];

    _record = [UIButton buttonWithType:UIButtonTypeSystem];
    UIButtonConfiguration* recordConfig = [UIButtonConfiguration filledButtonConfiguration];
    recordConfig.title = @"Start recording";
    _record.configuration = recordConfig;
    [_record addTarget:self action:@selector(toggleRecording) forControlEvents:UIControlEventTouchUpInside];
    _record.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:_record];

    _cancel = [UIButton buttonWithType:UIButtonTypeSystem];
    UIButtonConfiguration* cancelConfig = [UIButtonConfiguration grayButtonConfiguration];
    cancelConfig.title = @"Cancel";
    _cancel.configuration = cancelConfig;
    [_cancel addTarget:self action:@selector(cancelCapture) forControlEvents:UIControlEventTouchUpInside];
    _cancel.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:_cancel];

    UILayoutGuide* safe = self.view.safeAreaLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [_status.topAnchor constraintEqualToAnchor:safe.topAnchor constant:12],
        [_status.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:12],
        [_status.trailingAnchor constraintLessThanOrEqualToAnchor:safe.trailingAnchor constant:-12],
        [_record.bottomAnchor constraintEqualToAnchor:safe.bottomAnchor constant:-24],
        [_record.centerXAnchor constraintEqualToAnchor:safe.centerXAnchor constant:50],
        [_cancel.centerYAnchor constraintEqualToAnchor:_record.centerYAnchor],
        [_cancel.trailingAnchor constraintEqualToAnchor:_record.leadingAnchor constant:-16],
    ]];

    if (!ARWorldTrackingConfiguration.isSupported) {
        _initError = @"ARKit world tracking is not supported on this device";
    }
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    if (_initError != nil) [self finishWith:nil message:_initError];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    if (_initError != nil) return;
    UIApplication.sharedApplication.idleTimerDisabled = YES;
    ARWorldTrackingConfiguration* config = [[ARWorldTrackingConfiguration alloc] init];
    // The recorded stream is the camera image itself. Like the Android
    // capture (largest CPU image at 30 fps, typically 1920x1080), pick the
    // largest format whose long side is at most the 1920-pixel extraction
    // cap, preferring 30 fps among equal sizes.
    ARVideoFormat* best = nil;
    for (ARVideoFormat* format in ARWorldTrackingConfiguration.supportedVideoFormats) {
        const CGSize size = format.imageResolution;
        if (MAX(size.width, size.height) > 1920) continue;
        if (best == nil) { best = format; continue; }
        const double area = size.width * size.height;
        const double bestArea = best.imageResolution.width * best.imageResolution.height;
        if (area > bestArea ||
            (area == bestArea && format.framesPerSecond == 30 && best.framesPerSecond != 30)) {
            best = format;
        }
    }
    if (best != nil) config.videoFormat = best;
    config.autoFocusEnabled = YES;
    config.worldAlignment = ARWorldAlignmentGravity;
    _formatLine = [NSString stringWithFormat:
            @"# camera_config image %.0fx%.0f fps %ld autofocus_enabled true world_alignment gravity",
            config.videoFormat.imageResolution.width, config.videoFormat.imageResolution.height,
            (long)config.videoFormat.framesPerSecond];
    [_sceneView.session runWithConfiguration:config];
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [_sceneView.session pause];
}

- (void)cancelCapture {
    dispatch_async(_frameQueue, ^{
        if (self->_finishing) return;
        self->_finishing = YES;
        self->_recording = NO;
        [self stopSensors];
        [self->_writer cancelWriting];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self finishWith:nil message:nil];
        });
    });
}

- (void)toggleRecording {
    dispatch_async(_frameQueue, ^{
        if (self->_finishing) return;
        if (!self->_recording) {
            [self startRecording];
        } else {
            [self stopRecording];
        }
    });
}

// Frame queue.
- (void)startRecording {
    NSFileManager* manager = NSFileManager.defaultManager;
    for (NSString* name in @[ @"video.mp4", @"arkit_poses.txt", @"sensors.csv", @"capture.png" ]) {
        if ([manager fileExistsAtPath:[self path:name]]) {
            [self showMessage:@"capture output already exists; refusing to overwrite"];
            return;
        }
    }
    _videoStartNs = -1;
    _videoFrames = 0;
    _droppedFrames = 0;
    _poseCount = 0;
    _intrinsicsLine = nil;
    [_poseLines setString:@""];
    _recordingStarted = [NSDate date];
    _liveEvidenceRequested = NO;
    [self startSensors];
    _recording = YES;
    dispatch_async(dispatch_get_main_queue(), ^{
        self->_liveEvidenceState = kLiveNone;
        self->_liveEvidenceAbandoned = NO;
        self->_onLiveEvidence = nil;
        UIButtonConfiguration* config = self->_record.configuration;
        config.title = @"Stop recording";
        config.baseBackgroundColor = UIColor.systemRedColor;
        self->_record.configuration = config;
    });
}

// Frame queue; the writer is created lazily from the first frame's size.
- (BOOL)ensureWriterForBuffer:(CVPixelBufferRef)buffer {
    if (_writer != nil) return YES;
    NSError* error = nil;
    _writer = [[AVAssetWriter alloc] initWithURL:[NSURL fileURLWithPath:[self path:@"video.mp4"]]
                                        fileType:AVFileTypeMPEG4
                                           error:&error];
    if (_writer == nil) return NO;
    const size_t width = CVPixelBufferGetWidth(buffer);
    const size_t height = CVPixelBufferGetHeight(buffer);
    NSDictionary* settings = @{
        AVVideoCodecKey : AVVideoCodecTypeH264,
        AVVideoWidthKey : @(width),
        AVVideoHeightKey : @(height),
        AVVideoCompressionPropertiesKey : @{
            // High bit rate: the frames feed SIFT, so compression artifacts
            // cost matches.
            AVVideoAverageBitRateKey : @(24'000'000),
            AVVideoProfileLevelKey : AVVideoProfileLevelH264HighAutoLevel,
        },
    };
    _writerInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                      outputSettings:settings];
    _writerInput.expectsMediaDataInRealTime = YES;
    _adaptor = [AVAssetWriterInputPixelBufferAdaptor
            assetWriterInputPixelBufferAdaptorWithAssetWriterInput:_writerInput
                                       sourcePixelBufferAttributes:nil];
    if (![_writer canAddInput:_writerInput]) return NO;
    [_writer addInput:_writerInput];
    if (![_writer startWriting]) return NO;
    [_writer startSessionAtSourceTime:kCMTimeZero];
    return YES;
}

- (void)session:(ARSession*)session didUpdateFrame:(ARFrame*)frame {
    ARCamera* camera = frame.camera;
    const BOOL tracking = camera.trackingState == ARTrackingStateNormal;
    if (_recording && !_finishing) {
        const long long timestamp = Ns(frame.timestamp);
        if (_videoStartNs < 0) {
            // MP4 pts 0 is the first recorded frame, whether or not it was
            // tracked (the Android capture anchors the same way).
            if (![self ensureWriterForBuffer:frame.capturedImage]) {
                _recording = NO;
                [self showMessage:[NSString stringWithFormat:@"Could not start recording: %@",
                                                             _writer.error.localizedDescription]];
                return;
            }
            _videoStartNs = timestamp;
        }
        if (_writerInput.readyForMoreMediaData) {
            const CMTime time = CMTimeMake(timestamp - _videoStartNs, 1'000'000'000);
            if ([_adaptor appendPixelBuffer:frame.capturedImage withPresentationTime:time]) {
                _videoFrames++;
            } else {
                _droppedFrames++;
            }
        } else {
            _droppedFrames++;
        }
        if (tracking) {
            if (_intrinsicsLine == nil) {
                // Intrinsics of capturedImage, i.e. of the recorded stream.
                const simd_float3x3 k = camera.intrinsics;
                _intrinsicsLine = [NSString stringWithFormat:@"# intrinsics %.9g %.9g %.9g %.9g %.0f %.0f",
                                   k.columns[0][0], k.columns[1][1], k.columns[2][0], k.columns[2][1],
                                   camera.imageResolution.width, camera.imageResolution.height];
            }
            const simd_float4x4 transform = camera.transform;
            const simd_quatf q = simd_quaternion(transform);
            [_poseLines appendFormat:@"%lld %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", timestamp,
                                     transform.columns[3][0], transform.columns[3][1],
                                     transform.columns[3][2], q.vector[0], q.vector[1],
                                     q.vector[2], q.vector[3]];
            _poseCount++;
        }
    }
    // The live capture image is taken right after the status below reaches the
    // label, so its frame and pose counts are this frame's.
    const BOOL liveEvidenceDue = _recording && !_finishing && !_liveEvidenceRequested &&
                                 _videoFrames >= kLiveEvidenceFrame;
    if (liveEvidenceDue) _liveEvidenceRequested = YES;
    if (++_statusCounter % 6 != 0 && !liveEvidenceDue) return;
    NSString* state = @"TRACKING";
    if (!tracking) {
        switch (camera.trackingState) {
            case ARTrackingStateNotAvailable: state = @"NOT AVAILABLE"; break;
            case ARTrackingStateLimited:
                switch (camera.trackingStateReason) {
                    case ARTrackingStateReasonInitializing: state = @"LIMITED (initializing)"; break;
                    case ARTrackingStateReasonExcessiveMotion: state = @"LIMITED (moving too fast)"; break;
                    case ARTrackingStateReasonInsufficientFeatures: state = @"LIMITED (low texture)"; break;
                    case ARTrackingStateReasonRelocalizing: state = @"LIMITED (relocalizing)"; break;
                    default: state = @"LIMITED"; break;
                }
                break;
            default: break;
        }
    }
    NSMutableString* text = [NSMutableString
            stringWithString:_recording ? [NSString stringWithFormat:@"Capture: Recording video and ARKit poses · %ld poses",
                                                                     (long)_poseCount]
                                        : @"Capture: Ready to record"];
    [text appendFormat:@"\n%@  %.0fx%.0f  AF AUTO", state, camera.imageResolution.width, camera.imageResolution.height];
    if (_recording) {
        [text appendFormat:@"  %ld frames", (long)_videoFrames];
        if (_droppedFrames > 0) [text appendFormat:@"  %ld dropped", (long)_droppedFrames];
    }
    _lastStatus = [text copy];
    dispatch_async(dispatch_get_main_queue(), ^{
        self->_status.text = text;
        if (liveEvidenceDue) [self saveLiveEvidence];
    });
}

- (void)session:(ARSession*)session didFailWithError:(NSError*)error {
    dispatch_async(dispatch_get_main_queue(), ^{
        self->_status.text = [@"ARKit error: " stringByAppendingString:error.localizedDescription];
    });
}

// Frame queue.
- (void)stopRecording {
    _recording = NO;
    _finishing = YES;
    [self stopSensors];
    const long long durationMs = llround(-[_recordingStarted timeIntervalSinceNow] * 1000.0);
    if (_poseCount < kMinPoses || _videoStartNs < 0 || _writer == nil) {
        [_writer cancelWriting];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self finishWith:nil message:@"Too few tracked frames"];
        });
        return;
    }
    NSMutableString* header = [NSMutableString stringWithString:@"# cross3dgs ARKit capture v1\n"];
    [header appendFormat:@"# video_start_ns %lld\n", _videoStartNs];
    if (_intrinsicsLine != nil) [header appendFormat:@"%@\n", _intrinsicsLine];
    if (_formatLine != nil) [header appendFormat:@"%@\n", _formatLine];
    [header appendFormat:@"# video_frames %ld dropped_frames %ld\n", (long)_videoFrames, (long)_droppedFrames];
    [header appendString:@"# gravity_world 0 -1 0\n"];
    [header appendString:@"# format: timestamp_ns tx ty tz qx qy qz qw (world_from_camera)\n"];
    NSString* poses = [header stringByAppendingString:_poseLines];
    NSError* error = nil;
    if (![poses writeToFile:[self path:@"arkit_poses.txt"] atomically:YES encoding:NSUTF8StringEncoding error:&error]) {
        [_writer cancelWriting];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self finishWith:nil message:@"Could not save poses"];
        });
        return;
    }
    E2eCaptureResult* result = [[E2eCaptureResult alloc] init];
    result.durationMs = durationMs;
    result.poseCount = _poseCount;
    result.sensorCount = _sensorCount.load();
    NSString* label = [[_lastStatus stringByReplacingOccurrencesOfString:@"Recording" withString:@"Recorded"]
            stringByReplacingOccurrencesOfString:@"\n" withString:@"  "] ?: @"";
    [_writerInput markAsFinished];
    dispatch_async(dispatch_get_main_queue(), ^{
        UIButtonConfiguration* config = self->_record.configuration;
        config.title = @"Saving...";
        self->_record.configuration = config;
        self->_record.enabled = NO;
        self->_cancel.enabled = NO;
    });
    [_writer finishWritingWithCompletionHandler:^{
        const BOOL written = self->_writer.status == AVAssetWriterStatusCompleted;
        dispatch_async(dispatch_get_main_queue(), ^{
            if (!written) {
                [self finishWith:nil message:@"Could not finalize the video"];
                return;
            }
            // Fallback when the live screen image failed: capture.png from the
            // finalized MP4.
            void (^fromVideo)(void) = ^{
                dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                    [NSFileManager.defaultManager removeItemAtPath:[self path:@"capture.png"] error:nil];
                    const BOOL evidence = [self saveCaptureEvidence:label];
                    dispatch_async(dispatch_get_main_queue(), ^{
                        if (evidence) {
                            [self finishWith:result message:nil];
                        } else {
                            [self finishWith:nil message:@"Could not save the capture image"];
                        }
                    });
                });
            };
            if (self->_liveEvidenceState == kLiveDone) {
                [self finishWith:result message:nil];
            } else if (self->_liveEvidenceState == kLivePending) {
                self->_onLiveEvidence = ^(BOOL captured) {
                    if (captured) {
                        [self finishWith:result message:nil];
                    } else {
                        fromVideo();
                    }
                };
            } else {
                fromVideo();
            }
        });
    }];
}

// Main thread. ARSCNView draws through Metal, which drawViewHierarchy does not
// reliably capture, so the scene comes from its own snapshot (the frame on
// screen) and the overlays are drawn over it, after the status text just set
// has been laid out.
- (void)saveLiveEvidence {
    if (_liveEvidenceAbandoned || _completion == nil) {
        [self finishLiveEvidence:NO];
        return;
    }
    _liveEvidenceState = kLivePending;
    [self.view layoutIfNeeded];
    UIImage* scene = [_sceneView snapshot];
    UIView* root = self.view;
    UIGraphicsImageRendererFormat* format = [UIGraphicsImageRendererFormat preferredFormat];
    format.opaque = YES;
    UIImage* shot = [[[UIGraphicsImageRenderer alloc] initWithBounds:root.bounds format:format]
            imageWithActions:^(UIGraphicsImageRendererContext*) {
                [UIColor.blackColor setFill];
                UIRectFill(root.bounds);
                [scene drawInRect:self->_sceneView.frame];
                for (UIView* overlay in @[ self->_status, self->_record, self->_cancel ]) {
                    if (!overlay.hidden) [overlay drawViewHierarchyInRect:overlay.frame afterScreenUpdates:YES];
                }
            }];
    CGImageRef image = CGImageRetain(shot.CGImage);
    NSString* path = [self path:@"capture.png"];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        const BOOL written = image != nullptr && E2eSharpness(image) > 0.0 &&
                             ![NSFileManager.defaultManager fileExistsAtPath:path] &&
                             E2eWritePng(image, path);
        CGImageRelease(image);
        dispatch_async(dispatch_get_main_queue(), ^{
            [self finishLiveEvidence:written];
        });
    });
}

// Main thread.
- (void)finishLiveEvidence:(BOOL)captured {
    if (_liveEvidenceAbandoned) {
        [NSFileManager.defaultManager removeItemAtPath:[self path:@"capture.png"] error:nil];
        _liveEvidenceState = kLiveFailed;
        return;
    }
    _liveEvidenceState = captured ? kLiveDone : kLiveFailed;
    void (^next)(BOOL) = _onLiveEvidence;
    _onLiveEvidence = nil;
    if (next != nil) next(captured);
}

// capture.png from the finalized MP4 (Android saveVideoFrameEvidence): the
// sharpest of five decoded candidates, letterboxed into the screen extent
// with the capture status drawn over it. A constant image is rejected.
- (BOOL)saveCaptureEvidence:(NSString*)label {
    CGImageRef frame = [E2eFrameExtractor copySharpestPreviewFrame:[self path:@"video.mp4"]];
    if (frame == nullptr) return NO;
    if (E2eSharpness(frame) <= 0.0) {
        CGImageRelease(frame);
        return NO;
    }
    __block CGSize screen;
    dispatch_sync(dispatch_get_main_queue(), ^{
        screen = self.view.bounds.size;
    });
    const CGFloat scale = UIScreen.mainScreen.scale;
    const CGSize output = CGSizeMake(MAX(screen.width * scale, CGImageGetWidth(frame)),
                                     MAX(screen.height * scale, CGImageGetHeight(frame)));
    UIGraphicsImageRendererFormat* format = [UIGraphicsImageRendererFormat preferredFormat];
    format.scale = 1;
    format.opaque = YES;
    UIImage* composed = [[[UIGraphicsImageRenderer alloc] initWithSize:output format:format]
            imageWithActions:^(UIGraphicsImageRendererContext* context) {
                [UIColor.blackColor setFill];
                UIRectFill(CGRectMake(0, 0, output.width, output.height));
                const CGFloat fit = MIN(output.width / CGImageGetWidth(frame),
                                        output.height / CGImageGetHeight(frame));
                const CGSize draw = CGSizeMake(CGImageGetWidth(frame) * fit, CGImageGetHeight(frame) * fit);
                [[UIImage imageWithCGImage:frame]
                        drawInRect:CGRectMake((output.width - draw.width) / 2,
                                              (output.height - draw.height) / 2, draw.width, draw.height)];
                NSDictionary* attributes = @{
                    NSFontAttributeName : [UIFont monospacedSystemFontOfSize:13 * scale weight:UIFontWeightRegular],
                    NSForegroundColorAttributeName : UIColor.whiteColor,
                    NSBackgroundColorAttributeName : [UIColor colorWithWhite:0 alpha:0.5],
                };
                [label drawAtPoint:CGPointMake(16 * scale, 60 * scale) withAttributes:attributes];
            }];
    CGImageRelease(frame);
    return E2eWritePng(composed.CGImage, [self path:@"capture.png"]);
}

- (void)startSensors {
    NSString* path = [self path:@"sensors.csv"];
    [@"timestamp_ns,sensor_type,accuracy,v0,v1,v2,v3,v4\n" writeToFile:path
                                                             atomically:NO
                                                               encoding:NSUTF8StringEncoding
                                                                  error:nil];
    _sensorFile = [NSFileHandle fileHandleForWritingAtPath:path];
    [_sensorFile seekToEndOfFile];
    _sensorCount = 0;
    _motion = [[CMMotionManager alloc] init];
    _motionQueue = [[NSOperationQueue alloc] init];
    _motionQueue.maxConcurrentOperationCount = 1;
    // ~50 Hz, matching Android SENSOR_DELAY_GAME. Accelerometer and gravity
    // are converted from g to m/s^2 with Android's sign (+9.81 on +Z when
    // the phone lies face up); CoreMotion reports the opposite sign.
    const NSTimeInterval interval = 1.0 / 50.0;
    __weak E2eCaptureViewController* weakSelf = self;
    if (_motion.accelerometerAvailable) {
        _motion.accelerometerUpdateInterval = interval;
        [_motion startAccelerometerUpdatesToQueue:_motionQueue withHandler:^(CMAccelerometerData* data, NSError*) {
            if (data == nil) return;
            [weakSelf writeSensor:kTypeAccelerometer time:data.timestamp accuracy:3 values:@[
                @(-data.acceleration.x * kStandardGravity), @(-data.acceleration.y * kStandardGravity),
                @(-data.acceleration.z * kStandardGravity) ]];
        }];
    }
    if (_motion.gyroAvailable) {
        _motion.gyroUpdateInterval = interval;
        [_motion startGyroUpdatesToQueue:_motionQueue withHandler:^(CMGyroData* data, NSError*) {
            if (data == nil) return;
            [weakSelf writeSensor:kTypeGyroscope time:data.timestamp accuracy:3 values:@[
                @(data.rotationRate.x), @(data.rotationRate.y), @(data.rotationRate.z) ]];
        }];
    }
    if (_motion.magnetometerAvailable) {
        _motion.magnetometerUpdateInterval = interval;
        [_motion startMagnetometerUpdatesToQueue:_motionQueue withHandler:^(CMMagnetometerData* data, NSError*) {
            if (data == nil) return;
            [weakSelf writeSensor:kTypeMagneticField time:data.timestamp accuracy:0 values:@[
                @(data.magneticField.x), @(data.magneticField.y), @(data.magneticField.z) ]];
        }];
    }
    if (_motion.deviceMotionAvailable) {
        _motion.deviceMotionUpdateInterval = interval;
        [_motion startDeviceMotionUpdatesUsingReferenceFrame:CMAttitudeReferenceFrameXArbitraryCorrectedZVertical
                                                     toQueue:_motionQueue
                                                 withHandler:^(CMDeviceMotion* data, NSError*) {
            if (data == nil) return;
            [weakSelf writeSensor:kTypeGravity time:data.timestamp accuracy:3 values:@[
                @(-data.gravity.x * kStandardGravity), @(-data.gravity.y * kStandardGravity),
                @(-data.gravity.z * kStandardGravity) ]];
            const CMQuaternion q = data.attitude.quaternion;
            [weakSelf writeSensor:kTypeRotationVector time:data.timestamp accuracy:3 values:@[
                @(q.x), @(q.y), @(q.z), @(q.w) ]];
        }];
    }
}

- (void)writeSensor:(int)type time:(NSTimeInterval)time accuracy:(int)accuracy values:(NSArray<NSNumber*>*)values {
    NSMutableString* row = [NSMutableString stringWithFormat:@"%lld,%d,%d", Ns(time), type, accuracy];
    for (NSUInteger index = 0; index < 5; index++) {
        if (index < values.count) {
            [row appendFormat:@",%.9g", values[index].doubleValue];
        } else {
            [row appendString:@","];
        }
    }
    [row appendString:@"\n"];
    [_sensorLock lock];
    if (_sensorFile != nil) {
        [_sensorFile writeData:[row dataUsingEncoding:NSUTF8StringEncoding]];
        _sensorCount++;
    }
    [_sensorLock unlock];
}

- (void)stopSensors {
    [_motion stopAccelerometerUpdates];
    [_motion stopGyroUpdates];
    [_motion stopMagnetometerUpdates];
    [_motion stopDeviceMotionUpdates];
    [_motionQueue waitUntilAllOperationsAreFinished];
    [_sensorLock lock];
    [_sensorFile synchronizeFile];
    [_sensorFile closeFile];
    _sensorFile = nil;
    [_sensorLock unlock];
}

- (void)showMessage:(NSString*)message {
    dispatch_async(dispatch_get_main_queue(), ^{
        self->_status.text = message;
    });
}

- (void)finishWith:(nullable E2eCaptureResult*)result message:(nullable NSString*)message {
    if (result == nil) {
        // A live capture image of a discarded recording is not evidence.
        _liveEvidenceAbandoned = YES;
        _onLiveEvidence = nil;
        [NSFileManager.defaultManager removeItemAtPath:[self path:@"capture.png"] error:nil];
    }
    UIApplication.sharedApplication.idleTimerDisabled = NO;
    void (^completion)(E2eCaptureResult* _Nullable, NSString* _Nullable) = _completion;
    _completion = nil;
    if (completion == nil) return;
    [self.presentingViewController dismissViewControllerAnimated:YES completion:^{
        completion(result, message);
    }];
}

@end
