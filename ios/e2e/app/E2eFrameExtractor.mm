#import "E2eFrameExtractor.h"

#import <AVFoundation/AVFoundation.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <cmath>
#include <optional>
#include <sstream>
#include <vector>

#include "E2ePipeline.h"

namespace {
constexpr int kCandidatesPerFrame = 2;
constexpr int kMaxDecodeDimension = 1920;

NSError* MakeError(NSString* message) {
    return [NSError errorWithDomain:@"cross3dgs.extract"
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey : message}];
}

BOOL WriteImage(CGImageRef image, NSString* path, CFStringRef type,
                NSDictionary* options) {
    NSURL* url = [NSURL fileURLWithPath:path];
    CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url, type, 1, nullptr);
    if (destination == nullptr) return NO;
    CGImageDestinationAddImage(destination, image,
                               (__bridge CFDictionaryRef)options);
    const bool ok = CGImageDestinationFinalize(destination);
    CFRelease(destination);
    return ok;
}

AVAssetImageGenerator* ExactGenerator(AVAsset* asset, CGSize maximumSize) {
    AVAssetImageGenerator* generator =
            [[AVAssetImageGenerator alloc] initWithAsset:asset];
    generator.appliesPreferredTrackTransform = YES;
    generator.requestedTimeToleranceBefore = kCMTimeZero;
    generator.requestedTimeToleranceAfter = kCMTimeZero;
    generator.maximumSize = maximumSize;
    return generator;
}

CGImageRef CopyFrame(AVAssetImageGenerator* generator, double seconds) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return [generator copyCGImageAtTime:CMTimeMakeWithSeconds(seconds, 600000)
                             actualTime:nullptr
                                  error:nil];
#pragma clang diagnostic pop
}
}  // namespace

double E2eSharpness(CGImageRef image) {
    const size_t width = 160;
    const size_t height = std::max<size_t>(
            1, CGImageGetHeight(image) * width /
                       std::max<size_t>(1, CGImageGetWidth(image)));
    std::vector<uint8_t> pixels(width * height * 4);
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
            pixels.data(), width, height, 8, width * 4, space,
            kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(space);
    if (context == nullptr) return 0.0;
    CGContextSetInterpolationQuality(context, kCGInterpolationMedium);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    std::vector<double> luma(width * height);
    for (size_t index = 0; index < luma.size(); index++) {
        luma[index] = 0.299 * pixels[index * 4] + 0.587 * pixels[index * 4 + 1] +
                      0.114 * pixels[index * 4 + 2];
    }
    double sum = 0.0, sum_squared = 0.0;
    size_t count = 0;
    for (size_t y = 1; y + 1 < height; y++) {
        for (size_t x = 1; x + 1 < width; x++) {
            const double gradient =
                    std::abs(luma[y * width + x + 1] - luma[y * width + x - 1]) +
                    std::abs(luma[(y + 1) * width + x] - luma[(y - 1) * width + x]);
            sum += gradient;
            sum_squared += gradient * gradient;
            count++;
        }
    }
    if (count == 0) return 0.0;
    const double mean = sum / count;
    return sum_squared / count - mean * mean;
}

BOOL E2eWritePng(CGImageRef image, NSString* path) {
    return WriteImage(image, path, (__bridge CFStringRef)UTTypePNG.identifier, @{});
}

BOOL E2eWriteJpeg(CGImageRef image, NSString* path, double quality) {
    return WriteImage(image, path, (__bridge CFStringRef)UTTypeJPEG.identifier,
                      @{(__bridge NSString*)kCGImageDestinationLossyCompressionQuality : @(quality)});
}

@implementation E2eExtractionResult
@end

@implementation E2eFrameExtractor

+ (nullable E2eExtractionResult*)extractVideo:(NSString*)videoPath
                                        toDir:(NSString*)outDir
                                 targetFrames:(NSInteger)targetFrames
                                    posesPath:(nullable NSString*)posesPath
                                   priorsPath:(nullable NSString*)priorsPath
                                     progress:(void (^)(NSInteger, NSInteger))progress
                                        error:(NSError**)error {
    NSFileManager* manager = NSFileManager.defaultManager;
    if ([manager contentsOfDirectoryAtPath:outDir error:nil].count > 0) {
        if (error) *error = MakeError(@"frame output already exists; create a new run instead of overwriting");
        return nil;
    }
    if (priorsPath != nil && [manager fileExistsAtPath:priorsPath]) {
        if (error) *error = MakeError(@"frame-prior output already exists; refusing to overwrite");
        return nil;
    }
    [manager createDirectoryAtPath:outDir withIntermediateDirectories:YES attributes:nil error:nil];

    AVURLAsset* asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:videoPath]
                                            options:@{AVURLAssetPreferPreciseDurationAndTimingKey : @YES}];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    AVAssetTrack* track = [asset tracksWithMediaType:AVMediaTypeVideo].firstObject;
    const double duration = CMTimeGetSeconds(asset.duration);
#pragma clang diagnostic pop
    if (track == nil || !(duration > 0)) {
        if (error) *error = MakeError(@"video has no decodable track");
        return nil;
    }
    const CGSize natural = track.naturalSize;
    const double longest = std::max(natural.width, natural.height);
    const double scale = longest > kMaxDecodeDimension ? kMaxDecodeDimension / longest : 1.0;
    AVAssetImageGenerator* generator = ExactGenerator(
            asset, CGSizeMake(std::round(natural.width * scale), std::round(natural.height * scale)));

    std::optional<cross3dgs_ios::PosesFile> poses;
    if (posesPath != nil) poses = cross3dgs_ios::PosesFile::parse(posesPath.fileSystemRepresentation);

    std::ostringstream prior_lines;
    prior_lines.precision(17);
    E2eExtractionResult* result = [[E2eExtractionResult alloc] init];
    size_t decoded_width = 0, decoded_height = 0;
    const NSInteger slots = targetFrames * kCandidatesPerFrame;
    for (NSInteger i = 0; i < targetFrames; i++) {
        @autoreleasepool {
            CGImageRef best = nullptr;
            double best_score = -1.0;
            double best_time = 0.0;
            for (int j = 0; j < kCandidatesPerFrame; j++) {
                const NSInteger slot = i * kCandidatesPerFrame + j;
                // Same slot centers as Android: (slot + 0.5) / slots of the
                // duration, at microsecond resolution.
                const long long time_us =
                        static_cast<long long>((slot + 0.5) / slots * duration * 1e6);
                CGImageRef candidate = CopyFrame(generator, time_us / 1e6);
                if (candidate == nullptr) continue;
                const double score = E2eSharpness(candidate);
                if (score > best_score) {
                    if (best != nullptr) CGImageRelease(best);
                    best = candidate;
                    best_score = score;
                    best_time = time_us / 1e6;
                } else {
                    CGImageRelease(candidate);
                }
            }
            if (best != nullptr) {
                NSString* name = [NSString stringWithFormat:@"frame_%04ld.jpg", (long)i];
                const BOOL ok = E2eWriteJpeg(best, [outDir stringByAppendingPathComponent:name], 0.92);
                if (decoded_width == 0) {
                    decoded_width = CGImageGetWidth(best);
                    decoded_height = CGImageGetHeight(best);
                }
                CGImageRelease(best);
                if (!ok) {
                    if (error) *error = MakeError([@"failed to write " stringByAppendingString:name]);
                    return nil;
                }
                result.written++;
                if (poses) {
                    const int64_t t_abs = poses->videoStartNs() +
                                          static_cast<int64_t>(std::llround(best_time * 1e6)) * 1000;
                    if (const auto prior = poses->priorAt(t_abs)) {
                        prior_lines << name.UTF8String << ' ' << prior->position[0] << ' '
                                    << prior->position[1] << ' ' << prior->position[2] << ' '
                                    << prior->gravity_camera[0] << ' ' << prior->gravity_camera[1]
                                    << ' ' << prior->gravity_camera[2] << ' ' << prior->sigma << '\n';
                        result.positionPriors++;
                        const auto& g = prior->gravity_camera;
                        if (std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) > 0.5) {
                            result.gravityPriors++;
                        }
                    }
                }
            }
            progress(i + 1, targetFrames);
        }
    }

    if (poses && priorsPath != nil && result.positionPriors > 0) {
        std::ostringstream header;
        header.precision(17);
        if (poses->hasIntrinsics() && decoded_width > 0) {
            // Scale capture-resolution intrinsics to the decoded size.
            const auto& k = poses->intrinsics();
            const auto& size = poses->intrinsicsSize();
            const double sx = static_cast<double>(decoded_width) / size[0];
            const double sy = static_cast<double>(decoded_height) / size[1];
            header << "# intrinsics " << k[0] * sx << ' ' << k[1] * sy << ' ' << k[2] * sx << ' '
                   << k[3] * sy << ' ' << decoded_width << ' ' << decoded_height << '\n';
            result.intrinsicsPrior = YES;
        }
        NSString* text = [NSString stringWithUTF8String:(header.str() + prior_lines.str()).c_str()];
        if (![text writeToFile:priorsPath atomically:YES encoding:NSUTF8StringEncoding error:error]) {
            return nil;
        }
    }
    return result;
}

+ (nullable CGImageRef)copySharpestPreviewFrame:(NSString*)videoPath {
    AVURLAsset* asset = [AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:videoPath] options:nil];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    const double duration = CMTimeGetSeconds(asset.duration);
#pragma clang diagnostic pop
    if (!(duration > 0)) return nullptr;
    AVAssetImageGenerator* generator = ExactGenerator(asset, CGSizeZero);
    CGImageRef best = nullptr;
    double best_score = -INFINITY;
    for (double fraction : {0.2, 0.35, 0.5, 0.65, 0.8}) {
        CGImageRef candidate = CopyFrame(generator, duration * fraction);
        if (candidate == nullptr) continue;
        const double score = E2eSharpness(candidate);
        if (score > best_score) {
            if (best != nullptr) CGImageRelease(best);
            best = candidate;
            best_score = score;
        } else {
            CGImageRelease(candidate);
        }
    }
    return best;
}

@end
