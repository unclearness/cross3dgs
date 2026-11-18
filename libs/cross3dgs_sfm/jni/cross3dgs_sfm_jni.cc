// JNI wrapper over the shared C SfM API for the Android capture-to-view app.
//
// Kotlin side: org.unclearness.cross3dgs.SfmNativeBridge. The SfM itself is
// src/cross3dgs_sfm.cc, the same code the iOS app links; this file only
// converts arguments and forwards COLMAP log lines (to logcat and Kotlin)
// and live preview frames (to Kotlin) from the library's threads.

#include <jni.h>

#include <android/log.h>

#include <string>

#include "cross3dgs_sfm.h"

namespace {

constexpr const char* kTag = "cross3dgs_sfm";
constexpr const char* kBridgeClass = "org/unclearness/cross3dgs/SfmNativeBridge";

JavaVM* g_vm = nullptr;
jclass g_bridge_class = nullptr;
jmethodID g_on_log = nullptr;
jmethodID g_on_preview = nullptr;

// Called on COLMAP worker threads.
void ForwardLog(const char* line, void* /*user_data*/) {
  __android_log_write(ANDROID_LOG_INFO, kTag, line);
  if (g_vm == nullptr || g_bridge_class == nullptr || g_on_log == nullptr) {
    return;
  }
  JNIEnv* env = nullptr;
  bool attached = false;
  if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) ==
      JNI_EDETACHED) {
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
      return;
    }
    attached = true;
  }
  jstring jline = env->NewStringUTF(line);
  if (jline != nullptr) {
    env->CallStaticVoidMethod(g_bridge_class, g_on_log, jline);
    env->DeleteLocalRef(jline);
  }
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  if (attached) {
    g_vm->DetachCurrentThread();
  }
}

// Called on the library's preview thread.
void ForwardPreview(const cross3dgs_sfm_preview* frame, void* /*user_data*/) {
  if (g_vm == nullptr || g_bridge_class == nullptr || g_on_preview == nullptr) {
    return;
  }
  JNIEnv* env = nullptr;
  bool attached = false;
  if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) ==
      JNI_EDETACHED) {
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
      return;
    }
    attached = true;
  }
  const jsize bytes = static_cast<jsize>(frame->width) * frame->height * 4;
  jbyteArray rgba = env->NewByteArray(bytes);
  jstring caption = env->NewStringUTF(frame->caption);
  if (rgba != nullptr && caption != nullptr) {
    env->SetByteArrayRegion(rgba, 0, bytes,
                            reinterpret_cast<const jbyte*>(frame->rgba));
    env->CallStaticVoidMethod(g_bridge_class, g_on_preview, rgba,
                              static_cast<jint>(frame->width),
                              static_cast<jint>(frame->height),
                              static_cast<jint>(frame->stage), caption);
  }
  if (rgba != nullptr) env->DeleteLocalRef(rgba);
  if (caption != nullptr) env->DeleteLocalRef(caption);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  if (attached) {
    g_vm->DetachCurrentThread();
  }
}

std::string ToStdString(JNIEnv* env, jstring s) {
  if (s == nullptr) return "";
  const char* chars = env->GetStringUTFChars(s, nullptr);
  std::string result(chars == nullptr ? "" : chars);
  env->ReleaseStringUTFChars(s, chars);
  return result;
}

}  // namespace

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
  g_vm = vm;
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }
  jclass local = env->FindClass(kBridgeClass);
  if (local != nullptr) {
    g_bridge_class = static_cast<jclass>(env->NewGlobalRef(local));
    g_on_log = env->GetStaticMethodID(g_bridge_class, "onLog",
                                      "(Ljava/lang/String;)V");
    g_on_preview = env->GetStaticMethodID(g_bridge_class, "onPreview",
                                          "([BIIILjava/lang/String;)V");
    env->DeleteLocalRef(local);
  }
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jint JNICALL
Java_org_unclearness_cross3dgs_SfmNativeBridge_nativeRun(
    JNIEnv* env,
    jobject /*thiz*/,
    jstring j_image_path,
    jstring j_workspace,
    jstring j_priors_path,
    jboolean use_global,
    jboolean single_camera) {
  const std::string image_path = ToStdString(env, j_image_path);
  const std::string workspace = ToStdString(env, j_workspace);
  const std::string priors_path = ToStdString(env, j_priors_path);
  return cross3dgs_sfm_run(image_path.c_str(), workspace.c_str(),
                           priors_path.c_str(), /*camera_model=*/nullptr,
                           use_global == JNI_TRUE ? 1 : 0,
                           single_camera == JNI_TRUE ? 1 : 0, ForwardLog,
                           nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_org_unclearness_cross3dgs_SfmNativeBridge_nativeCancel(
    JNIEnv* /*env*/, jobject /*thiz*/) {
  cross3dgs_sfm_cancel();
}

extern "C" JNIEXPORT void JNICALL
Java_org_unclearness_cross3dgs_SfmNativeBridge_nativeSetPreview(
    JNIEnv* /*env*/, jobject /*thiz*/, jint width, jint height) {
  // width <= 0 disables the preview.
  cross3dgs_sfm_set_preview(width > 0 ? ForwardPreview : nullptr, nullptr,
                            width, height);
}

extern "C" JNIEXPORT void JNICALL
Java_org_unclearness_cross3dgs_SfmNativeBridge_nativeSetThreads(
    JNIEnv* /*env*/, jobject /*thiz*/, jint extraction, jint matching,
    jint mapping) {
  cross3dgs_sfm_set_threads(extraction, matching, mapping);
}

extern "C" JNIEXPORT void JNICALL
Java_org_unclearness_cross3dgs_SfmNativeBridge_nativeSetCpuMask(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong mask) {
  cross3dgs_sfm_set_cpu_mask(static_cast<unsigned long long>(mask));
}
