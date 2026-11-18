#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface E2eCaptureResult : NSObject
@property(nonatomic) long long durationMs;
@property(nonatomic) NSInteger poseCount;
@property(nonatomic) NSInteger sensorCount;
@end

// ARKit capture into a newly created empty capture directory (iOS
// counterpart of ArCaptureActivity). Writes video.mp4, arkit_poses.txt,
// sensors.csv and capture.png once; result is nil when the capture was
// canceled (error nil) or unusable (error set).
@interface E2eCaptureViewController : UIViewController
- (instancetype)initWithCaptureRoot:(NSString*)captureRoot
                         completion:(void (^)(E2eCaptureResult* _Nullable result,
                                              NSString* _Nullable error))completion;
@end

NS_ASSUME_NONNULL_END
