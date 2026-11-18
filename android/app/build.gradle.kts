plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.unclearness.cross3dgs"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        // -Pcross3dgsApplicationId=<id> installs the same build under another
        // package name (clock-policy diagnostics); the namespace is unchanged.
        applicationId = project.findProperty("cross3dgsApplicationId") as String?
            ?: "org.unclearness.cross3dgs"
        // -Pcross3dgsAppLabel=<label> tells such installs apart on the device.
        manifestPlaceholders["appLabel"] =
            project.findProperty("cross3dgsAppLabel") as String? ?: "Cross 3DGS"
        minSdk = 29
        targetSdk = 36
        versionCode = 3
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
        // -Pcross3dgsShReorderNarrowSubgroups=ON|OFF overrides the CMake
        // switch that gives subgroups narrower than 32 (Mali) the reordered
        // SH-rest layout; without the property the CMake default applies.
        val shReorderNarrow =
            project.findProperty("cross3dgsShReorderNarrowSubgroups") as String?
        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_static")
                if (shReorderNarrow != null) {
                    arguments +=
                        "-DCROSS3DGS_SH_REORDER_NARROW_SUBGROUPS=$shReorderNarrow"
                }
            }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }
    buildFeatures { buildConfig = true }
    buildTypes {
        debug {
            externalNativeBuild {
                cmake { arguments += "-DCMAKE_BUILD_TYPE=RelWithDebInfo" }
            }
        }
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
    sourceSets["main"].assets.srcDir(
        layout.buildDirectory.dir("generated/cross3dgsAssets"))
    lint {
        // This Vulkan training build intentionally targets physical arm64
        // phones/tablets only; ChromeOS x86 translation is out of scope.
        disable += "ChromeOsAbiSupport"
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

val sceneRoot = file("../../data/360_v2")
val bundledScenes = listOf(
    "bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump")
val sparseFiles = listOf("cameras.bin", "images.bin", "points3D.bin")

val syncCross3dgsAssets by tasks.registering(Sync::class) {
    into(layout.buildDirectory.dir("generated/cross3dgsAssets"))
    bundledScenes.forEach { scene ->
        from(sceneRoot.resolve("$scene/sparse/0")) {
            include(sparseFiles)
            into("data/$scene/sparse/0")
        }
        from(sceneRoot.resolve("$scene/images_4")) {
            into("data/$scene/images_4")
        }
    }
    doFirst {
        val missing = bundledScenes.flatMap { scene ->
            sparseFiles.map { sceneRoot.resolve("$scene/sparse/0/$it") } +
                listOf(sceneRoot.resolve("$scene/images_4"))
        }.filterNot { it.exists() }
        check(missing.isEmpty()) {
            "Missing 360_v2 dataset files:\n" +
                missing.joinToString("\n") { "  $it" }
        }
    }
}
tasks.named("preBuild") { dependsOn(syncCross3dgsAssets) }
