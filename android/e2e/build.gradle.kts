plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.unclearness.cross3dgs.e2e"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        // -Pcross3dgsApplicationId / -Pcross3dgsAppLabel install the same build
        // under another package name and launcher label (clock-policy
        // diagnostics); the namespace is unchanged.
        applicationId = project.findProperty("cross3dgsApplicationId") as String?
            ?: "org.unclearness.cross3dgs.e2e"
        manifestPlaceholders["appLabel"] =
            project.findProperty("cross3dgsAppLabel") as String?
                ?: "Cross3DGS Capture-to-View"
        minSdk = 29
        targetSdk = 36
        versionCode = 8
        versionName = "0.9-e2e-launch-preset-budget"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_static") }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("../app/src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
            isDebuggable = true
            externalNativeBuild {
                cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" }
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    // libcross3dgs_sfm_jni.so from scripts/build_android_sfm.ps1 (CPU COLMAP
    // SfM, libs/cross3dgs_sfm); built outside Gradle because it needs the
    // vcpkg toolchain.
    providers.gradleProperty("sfmJniDir").orNull?.let { directory ->
        sourceSets["main"].jniLibs.srcDir(file(directory))
    }
    lint { disable += "ChromeOsAbiSupport" }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("com.google.android.material:material:1.12.0")
    implementation("com.google.ar:core:1.49.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}

val verifySfmJni by tasks.registering {
    doLast {
        // Unconditional: the app loads libcross3dgs_sfm_jni.so for every SfM
        // run, so an APK without it installs fine and then dies on the first
        // reconstruction. There is no valid e2e build without the library,
        // so there is no opt-out.
        val root = providers.gradleProperty("sfmJniDir")
            .orNull?.let(::file)
        check(root != null &&
            root.resolve("arm64-v8a/libcross3dgs_sfm_jni.so").isFile) {
            "E2E builds require -PsfmJniDir=<jniLibs root> containing " +
                "arm64-v8a/libcross3dgs_sfm_jni.so (scripts/build_android_sfm.ps1; " +
                "the app loads it for SfM and crashes on first reconstruction " +
                "without it)"
        }
    }
}
tasks.named("preBuild") { dependsOn(verifySfmJni) }
