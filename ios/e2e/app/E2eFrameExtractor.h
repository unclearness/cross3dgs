#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Variance of the horizontal+vertical luma gradient on a 160-pixel-wide
// bilinear downscale (VideoFrameExtractor.sharpness on Android).
double E2eSharpness(CGImageRef image);
BOOL E2eWritePng(CGImageRef image, NSString* path);
BOOL E2eWriteJpeg(CGImageRef image, NSString* path, double quality);

@interface E2eExtractionResult : NSObject
@property(nonatomic) NSInteger written;
@property(nonatomic) NSInteger positionPriors;
@property(nonatomic) NSInteger gravityPriors;
@property(nonatomic) BOOL intrinsicsPrior;
@end

// Port of VideoFrameExtractor.extract: the video is split into targetFrames
// equal windows, two candidates per window are decoded at exact times
// (longest side capped at 1920) and the sharper one is written as
// frame_%04d.jpg (quality 92). When posesPath exists, a frame prior is
// interpolated at each kept frame's capture time and priorsPath is written
// with the intrinsics scaled to the decoded size.
@interface E2eFrameExtractor : NSObject
+ (nullable E2eExtractionResult*)extractVideo:(NSString*)videoPath
                                        toDir:(NSString*)outDir
                                 targetFrames:(NSInteger)targetFrames
                                    posesPath:(nullable NSString*)posesPath
                                   priorsPath:(nullable NSString*)priorsPath
                                     progress:(void (^)(NSInteger done, NSInteger total))progress
                                        error:(NSError**)error;
// Sharpest of five temporal candidates (20..80% of the duration), used for
// the capture evidence image.
+ (nullable CGImageRef)copySharpestPreviewFrame:(NSString*)videoPath CF_RETURNS_RETAINED;
@end

NS_ASSUME_NONNULL_END
