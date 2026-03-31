# Mobile OS -- Comprehensive Guide to iOS and Android Architecture and Design Principles

> **Prerequisites**: Basic knowledge of process management, memory management, and file systems
> **Estimated Study Time**: Approximately 6 hours
> **Difficulty**: ★★★☆☆ (Intermediate)

Mobile operating systems have undergone unique evolution to achieve processing power rivaling desktop OSes and deliver the best user experience, all within the severe constraints of battery-powered operation, touch-based interaction, and a wide variety of sensors. This guide systematically covers the kernel structure, process model, memory management, power management, security model, and application lifecycle of mobile OSes, focusing on iOS and Android.

---

## What You Will Learn in This Chapter

- [ ] Explain the differences between iOS (XNU) and Android (Linux) kernel architectures with diagrams
- [ ] Understand the process lifecycle and priority models unique to mobile OSes
- [ ] Explain the security design based on sandboxing and permission models
- [ ] Understand the mechanisms of power management (DVFS, Doze, Background App Refresh)
- [ ] Grasp the design principles of push notifications, sensor integration, and IPC
- [ ] Compare Android's HAL / HIDL with iOS's IOKit driver model
- [ ] Build mobile app build, test, and distribution pipelines


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## Table of Contents

1. [History and Evolution of Mobile OS](#1-history-and-evolution-of-mobile-os)
2. [Kernel Architecture Comparison](#2-kernel-architecture-comparison)
3. [Process Model and Application Lifecycle](#3-process-model-and-application-lifecycle)
4. [Memory Management and Virtual Memory](#4-memory-management-and-virtual-memory)
5. [Power Management and Scheduling](#5-power-management-and-scheduling)
6. [Security Model and Sandboxing](#6-security-model-and-sandboxing)
7. [Inter-Process Communication and Push Notifications](#7-inter-process-communication-and-push-notifications)
8. [Sensors and Hardware Abstraction Layer](#8-sensors-and-hardware-abstraction-layer)
9. [App Development and Build Pipeline](#9-app-development-and-build-pipeline)
10. [Anti-Patterns and Design Pitfalls](#10-anti-patterns-and-design-pitfalls)
11. [Tiered Exercises](#11-tiered-exercises)
12. [FAQ -- Frequently Asked Questions](#12-faq----frequently-asked-questions)
13. [Summary and Next Steps](#13-summary-and-next-steps)
14. [References](#14-references)

---

## 1. History and Evolution of Mobile OS

### 1.1 Lineage of Mobile OS

The history of mobile OS dates back to PDA operating systems in the 1990s. Palm OS and Windows CE were pioneers, tackling the challenge of providing a GUI in severely constrained embedded environments.

```
Mobile OS Evolution Timeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1996 ──── Palm OS 1.0
           │  Stylus operation, single-tasking
           │  16MHz Motorola 68000, 128KB RAM
           ▼
2000 ──── Symbian OS (Nokia)
           │  Multitasking, C++ based
           │  First full-featured OS for mobile phones
           ▼
2002 ──── BlackBerry OS
           │  Push email, enterprise security
           │  Optimized for QWERTY keyboard
           ▼
2005 ──── Windows Mobile 5.0
           │  .NET Compact Framework
           │  Unified PocketPC and Smartphone
           ▼
2007 ──── iPhone OS 1.0 (later iOS)  ← Revolutionary turning point
           │  Multi-touch, Safari, Visual Voicemail
           │  No third-party apps (Web Apps only)
           ▼
2008 ──── Android 1.0 (HTC Dream)
           │  Open source (AOSP)
           │  Google services integration, Android Market
           ▼
2008 ──── iPhone OS 2.0 + App Store
           │  Native SDK released
           │  Third-party app ecosystem begins
           ▼
2010 ──── Android 2.2 (Froyo)   / iOS 4
           │  JIT compilation         Multitasking support
           ▼
2014 ──── Android 5.0 (Lollipop) / iOS 8
           │  ART (AOT compilation)   Extensions, Metal API
           │  Material Design         HealthKit, HomeKit
           ▼
2017 ──── Android 8.0 (Oreo)     / iOS 11
           │  Project Treble (HAL     ARKit, Core ML
           │  separation)             HEIF/HEVC standardization
           │  Background restrictions
           ▼
2020 ──── Android 11              / iOS 14
           │  Scoped Storage enforced App Clips, Widgets
           │  One-time permissions    App Library
           ▼
2023 ──── Android 14              / iOS 17
           │  Large screen support    Interactive Widgets
           │  Privacy Sandbox         StandBy mode
           ▼
2025 ──── Android 16              / iOS 19
           On-device AI integration, cutting edge of privacy enhancement
```

### 1.2 Why Mobile OS is Special

The fundamental difference between desktop and mobile OSes lies in the "severity of constraints."

```
Desktop OS vs Mobile OS -- Design Constraint Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                Desktop OS              Mobile OS
              ┌──────────────┐       ┌──────────────┐
  Power       │ AC power     │       │ Battery      │
              │ (unlimited)  │       │ (3000-5000mAh)│
              ├──────────────┤       ├──────────────┤
  Cooling     │ Fan/liquid   │       │ Passive      │
              │ Hundreds of W│       │ Few W limit  │
              ├──────────────┤       ├──────────────┤
  Memory      │ 16-128 GB   │       │ 4-16 GB      │
              │ Free swapping│       │ Limited swap │
              ├──────────────┤       ├──────────────┤
  Input       │ Keyboard     │       │ Touch        │
              │ Mouse        │       │ Gestures     │
              ├──────────────┤       ├──────────────┤
  Network     │ Wired/WiFi   │       │ Cellular/WiFi│
              │ Always-on    │       │ Intermittent │
              ├──────────────┤       ├──────────────┤
  Security    │ User trust   │       │ Zero trust   │
              │ model        │       │ App isolation│
              └──────────────┘       └──────────────┘

  Mobile OS Design Principles:
  1. Power efficiency first -- Every feature is evaluated from a power consumption perspective
  2. Responsiveness guarantee -- Blocking the UI thread is absolutely unacceptable
  3. Privacy by design -- Apps operate with minimum privileges
  4. Intermittent connectivity assumption -- Basic functions maintained even offline
  5. Defense in depth security -- Consistent protection from hardware to applications
```

### 1.3 Market Share and Influence

As of 2025, the mobile OS market is effectively a two-platform system of iOS and Android.

| Metric | iOS | Android | Others |
|--------|-----|---------|--------|
| Global share | ~27% | ~72% | ~1% |
| North America share | ~55% | ~44% | ~1% |
| Japan share | ~65% | ~34% | ~1% |
| Active devices | ~2 billion | ~3.5 billion | - |
| Annual app revenue | ~$85 billion | ~$48 billion | - |
| Number of developers | ~34 million | ~40 million | - |

> **Note**: While Android dominates in device count, iOS leads in app revenue. This is due to differences in user purchasing power and iOS's unified ecosystem.

---

## 2. Kernel Architecture Comparison

### 2.1 iOS: XNU Kernel

iOS is based on Apple's Darwin OS, and its core XNU (X is Not Unix) kernel is a hybrid design that fuses the Mach microkernel with FreeBSD's monolithic kernel.

```
XNU Kernel Architecture Detailed Diagram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────┐
  │              User Space                          │
  │                                                 │
  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
  │  │ UIKit /  │ │ Cocoa    │ │ System Daemons   ││
  │  │ SwiftUI  │ │ Touch    │ │ (launchd,        ││
  │  │ Apps     │ │Framework │ │  configd, etc.)  ││
  │  └────┬─────┘ └────┬─────┘ └────────┬─────────┘│
  │       │            │                │           │
  │  ─────┴────────────┴────────────────┴───────────│
  │       │  libSystem (libc, libdispatch, etc.)    │
  ├───────┼─────────────────────────────────────────┤
  │       │        Kernel Space                      │
  │       ▼                                         │
  │  ┌──────────────────────────────────────────┐   │
  │  │            BSD Layer                      │   │
  │  │  ┌────────────┬──────────┬─────────────┐ │   │
  │  │  │ POSIX API  │ VFS     │ Networking  │ │   │
  │  │  │ (syscall)  │(Virtual │ (TCP/IP)    │ │   │
  │  │  │ Process    │  FS)    │ BSD Socket  │ │   │
  │  │  │ Signals    │ HFS+    │             │ │   │
  │  │  │            │ APFS    │             │ │   │
  │  │  └────────────┴──────────┴─────────────┘ │   │
  │  ├──────────────────────────────────────────┤   │
  │  │            Mach Layer                     │   │
  │  │  ┌────────────┬──────────┬─────────────┐ │   │
  │  │  │ Tasks      │ IPC     │ Scheduler   │ │   │
  │  │  │ (Processes)│ (Mach   │ (Priority   │ │   │
  │  │  │ Threads    │  Port)  │  based)     │ │   │
  │  │  │ Virtual    │ MIG     │ Real-time   │ │   │
  │  │  │ Memory     │         │             │ │   │
  │  │  └────────────┴──────────┴─────────────┘ │   │
  │  ├──────────────────────────────────────────┤   │
  │  │            IOKit (Driver Framework)       │   │
  │  │  C++ based, object-oriented driver model  │   │
  │  │  Power management, hot-plug, device tree  │   │
  │  └──────────────────────────────────────────┘   │
  │                                                 │
  │  ┌──────────────────────────────────────────┐   │
  │  │   Secure Enclave Processor (SEP)          │   │
  │  │   Independent processor / dedicated OS    │   │
  │  │   (sepOS)                                 │   │
  │  │   Cryptographic key management /          │   │
  │  │   biometric data protection               │   │
  │  └──────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────┘
```

**Key Design Features of XNU:**

- **Mach Microkernel**: Handles task management, thread management, virtual memory management, and IPC. Mach port message passing is the foundation of IPC
- **BSD Layer**: Provides POSIX-compatible APIs, file system (APFS), networking stack (TCP/IP), and user/group management
- **IOKit**: Object-oriented driver framework written in C++. Supports power management and device hot-plugging
- **Secure Enclave**: A dedicated chip physically separated from the main processor that manages cryptographic keys and biometric authentication data

### 2.2 Android: Modified Linux Kernel

Android is based on the Linux kernel, but incorporates significant modifications that differ greatly from standard Linux.

```
Android System Architecture Detailed Diagram
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────┐
  │          Application Layer (Applications)        │
  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ │
  │  │ Phone  │ │ Chrome │ │Camera  │ │Third-party││
  │  │        │ │        │ │        │ │ Apps      ││
  │  └───┬────┘ └───┬────┘ └───┬────┘ └────┬─────┘ │
  ├──────┴──────────┴──────────┴────────────┴───────┤
  │        Android Framework (Java/Kotlin API)      │
  │  ┌────────────┬──────────────┬────────────────┐ │
  │  │Activity    │Content       │PackageManager  │ │
  │  │Manager     │Provider      │(App management)│ │
  │  ├────────────┼──────────────┼────────────────┤ │
  │  │Window      │Notification  │Telephony       │ │
  │  │Manager     │Manager       │Manager         │ │
  │  ├────────────┼──────────────┼────────────────┤ │
  │  │Resource    │Location      │Sensor          │ │
  │  │Manager     │Manager       │Manager         │ │
  │  └────────────┴──────────────┴────────────────┘ │
  ├─────────────────────────────────────────────────┤
  │     Android Runtime (ART) / Native Libraries     │
  │  ┌─────────────────┐  ┌──────────────────────┐ │
  │  │  ART             │  │ Native Libraries     │ │
  │  │  - AOT compile   │  │ - libc (Bionic)     │ │
  │  │  - GC (CC)       │  │ - OpenGL ES / Vulkan│ │
  │  │  - JNI           │  │ - Media Framework   │ │
  │  │  - Profile       │  │ - SQLite            │ │
  │  │    Guided Compile│  │ - SSL (BoringSSL)   │ │
  │  └─────────────────┘  └──────────────────────┘ │
  ├─────────────────────────────────────────────────┤
  │        Hardware Abstraction Layer (HAL)          │
  │  ┌────────┬────────┬────────┬────────┬────────┐ │
  │  │Audio   │Camera  │Sensor  │Graphics│Power   │ │
  │  │HAL     │HAL     │HAL     │HAL     │HAL     │ │
  │  └────────┴────────┴────────┴────────┴────────┘ │
  │        HIDL/AIDL (HAL Interface Definition Lang) │
  ├─────────────────────────────────────────────────┤
  │              Linux Kernel (Modified)              │
  │  ┌──────────┬──────────┬──────────┬───────────┐ │
  │  │ Binder   │ Ashmem / │ wakelocks│ Low Memory│ │
  │  │ IPC      │ ION      │ (power)  │ Killer    │ │
  │  │ Driver   │ Memory   │          │(Enhanced  │ │
  │  │          │          │          │ OOM)      │ │
  │  ├──────────┼──────────┼──────────┼───────────┤ │
  │  │ SELinux  │ cgroups  │ Network  │ File      │ │
  │  │(Mandatory│(Resource │ Drivers  │ System    │ │
  │  │ Access   │ Control) │          │ (ext4/    │ │
  │  │ Control) │          │          │  f2fs)    │ │
  │  └──────────┴──────────┴──────────┴───────────┘ │
  └─────────────────────────────────────────────────┘
```

**Android-Specific Linux Kernel Modifications:**

| Feature | Standard Linux | Android Modified Version |
|---------|---------------|------------------------|
| IPC | SysV IPC, Unix Socket, D-Bus | Binder (high-speed IPC) |
| Shared Memory | POSIX shm, tmpfs | Ashmem -> memfd (Android 10+) |
| GPU Memory | DRM/GEM | ION -> DMA-BUF heaps (Android 12+) |
| OOM Handling | oom_killer (simple) | LowMemoryKiller (multi-stage) |
| Power Management | runtime PM | wakelocks (user-space controllable) |
| Security | DAC + AppArmor/SELinux | SELinux enforcing + seccomp |
| Logging | syslog / journald | logd (ring buffer) |

### 2.3 Detailed Kernel Comparison

```
XNU vs Linux (Android) -- Differences in Design Philosophy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────┬───────────────────────┬───────────────────────┐
│     Aspect     │   XNU (iOS)           │   Linux (Android)     │
├────────────────┼───────────────────────┼───────────────────────┤
│ Kernel Type    │ Hybrid                │ Monolithic (modular)  │
│ Design Origin  │ Mach + BSD fusion     │ Linus Torvalds design │
│ IPC Foundation │ Mach Port (msg pass)  │ Binder (Android-spec) │
│ Drivers        │ IOKit (C++ OOP)       │ Kernel modules (C)    │
│ Scheduler      │ Mach priority-based   │ CFS (Completely Fair) │
│ File System    │ APFS                  │ ext4 / f2fs           │
│ Memory Mgmt    │ Mach VM + jetsam      │ Linux VM + LMK        │
│ License        │ APSL + closed parts   │ GPL v2                │
│ Lines of Code  │ ~12 million           │ ~30+ million          │
│ Release Cycle  │ Annual (iOS major)    │ Annual + monthly sec  │
│ Customizability│ Apple only            │ Any vendor            │
└────────────────┴───────────────────────┴───────────────────────┘
```

### 2.4 Code Example: Retrieving Kernel Information

**Code Example 1: iOS (Swift) -- Retrieving System Information**

```swift
import UIKit
import Darwin

/// Utility for retrieving iOS device system information
struct SystemInfo {

    /// Retrieve kernel version
    static func kernelVersion() -> String {
        var size = 0
        // Retrieve XNU kernel version via sysctl
        sysctlbyname("kern.osrelease", nil, &size, nil, 0)
        var version = CChar
        sysctlbyname("kern.osrelease", &version, &size, nil, 0)
        return String(cString: version)
    }

    /// Retrieve physical memory amount
    static func physicalMemory() -> UInt64 {
        return ProcessInfo.processInfo.physicalMemory
    }

    /// Retrieve active processor count
    static func processorCount() -> Int {
        return ProcessInfo.processInfo.activeProcessorCount
    }

    /// Monitor thermal state
    static func thermalState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:   return "Normal"
        case .fair:      return "Slightly warm"
        case .serious:   return "Hot - Performance throttled"
        case .critical:  return "Critical - Emergency throttling"
        @unknown default: return "Unknown"
        }
    }

    /// Device information summary
    static func summary() -> String {
        """
        ===== iOS Device Information =====
        Kernel: Darwin \(kernelVersion())
        Memory: \(physicalMemory() / 1_073_741_824) GB
        Processor Count: \(processorCount())
        Thermal State: \(thermalState())
        OS: \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)
        Model: \(UIDevice.current.model)
        ================================
        """
    }
}

// Usage example
// print(SystemInfo.summary())
// Output example:
// ===== iOS Device Information =====
// Kernel: Darwin 24.1.0
// Memory: 6 GB
// Processor Count: 6
// Thermal State: Normal
// OS: iOS 18.2
// Model: iPhone
// ================================
```

**Code Example 2: Android (Kotlin) -- Retrieving System Information**

```kotlin
import android.os.Build
import android.app.ActivityManager
import android.content.Context
import java.io.File

/**
 * Utility for retrieving Android device system information
 *
 * On Android, Linux kernel information can be accessed
 * directly through the /proc file system.
 */
object SystemInfoUtil {

    /** Retrieve Linux kernel version */
    fun getKernelVersion(): String {
        return try {
            File("/proc/version").readText().trim()
        } catch (e: Exception) {
            "Retrieval failed: ${e.message}"
        }
    }

    /** Retrieve CPU information */
    fun getCpuInfo(): Map<String, String> {
        val info = mutableMapOf<String, String>()
        try {
            File("/proc/cpuinfo").readLines().forEach { line ->
                val parts = line.split(":")
                if (parts.size == 2) {
                    info[parts[0].trim()] = parts[1].trim()
                }
            }
        } catch (e: Exception) {
            info["error"] = e.message ?: "Unknown error"
        }
        return info
    }

    /** Retrieve memory information */
    fun getMemoryInfo(context: Context): String {
        val activityManager = context.getSystemService(
            Context.ACTIVITY_SERVICE
        ) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val totalMB = memInfo.totalMem / (1024 * 1024)
        val availMB = memInfo.availMem / (1024 * 1024)
        val usedMB = totalMB - availMB
        val usagePercent = (usedMB.toDouble() / totalMB * 100).toInt()

        return """
            |===== Android Memory Information =====
            |Total Memory: ${totalMB} MB
            |In Use:       ${usedMB} MB (${usagePercent}%)
            |Available:    ${availMB} MB
            |Low Memory:   ${memInfo.lowMemory}
            |Threshold:    ${memInfo.threshold / (1024 * 1024)} MB
            |==============================
        """.trimMargin()
    }

    /** Retrieve build information */
    fun getBuildInfo(): String {
        return """
            |===== Android Build Information =====
            |Device: ${Build.DEVICE}
            |Model: ${Build.MODEL}
            |Manufacturer: ${Build.MANUFACTURER}
            |Android Version: ${Build.VERSION.RELEASE}
            |SDK Level: ${Build.VERSION.SDK_INT}
            |Security Patch: ${Build.VERSION.SECURITY_PATCH}
            |Build Number: ${Build.DISPLAY}
            |ABI: ${Build.SUPPORTED_ABIS.joinToString(", ")}
            |===============================
        """.trimMargin()
    }

    /** Retrieve SELinux status */
    fun getSeLinuxStatus(): String {
        return try {
            val process = Runtime.getRuntime().exec("getenforce")
            val result = process.inputStream.bufferedReader().readText().trim()
            "SELinux: $result"  // Enforcing, Permissive, or Disabled
        } catch (e: Exception) {
            "SELinux: Retrieval failed"
        }
    }
}
```

---

## 3. Process Model and Application Lifecycle

### 3.1 iOS Process Model

In iOS, every app runs as an independent process within a strict sandbox. The app lifecycle is strictly managed by the OS, and background execution is extremely restricted.

```
iOS Application Lifecycle
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    ┌───────────────┐
                    │  Not Running  │
                    │               │
                    └───────┬───────┘
                            │ User taps / System launches
                            ▼
                    ┌───────────────┐
                    │   Inactive    │◄────── Phone call,
                    │               │        notification, etc.
                    └───────┬───────┘
                            │
                            ▼
         ┌──────────────────────────────────────┐
         │            Active                    │
         │                                      │
         │  - Running in foreground              │
         │  - Able to receive UI events          │
         │  - Full resource access               │
         └──────────┬───────────────────────────┘
                    │ Home button / App switch
                    ▼
         ┌──────────────────────────────────────┐
         │          Background                  │
         │                                      │
         │  - ~30 second grace period for tasks  │
         │  - Specific APIs: music playback,     │
         │    location, VoIP, Bluetooth,         │
         │    downloads                          │
         │  - Background App Refresh (optional)  │
         └──────────┬───────────────────────────┘
                    │ Resource shortage / time elapsed
                    ▼
         ┌──────────────────────────────────────┐
         │           Suspended                  │
         │                                      │
         │  - Remains in memory but not executed │
         │  - Zero CPU time                      │
         │  - OS auto-terminates on low memory   │
         │    (jetsam mechanism)                  │
         └──────────────────────────────────────┘

  * jetsam: iOS-specific memory reclamation mechanism
    Preferentially kills suspended apps when memory is low
    Immediately kills apps that exceed per-app memory limit (footprint limit)
```

### 3.2 Android Process Model

Android classifies apps into 5 importance levels and terminates low-importance processes first when memory is insufficient.

```
Android Process Priority Hierarchy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Importance: High
  ┌─────────────────────────────────────────────┐
  │  1. Foreground Process                       │
  │     - Activity the user is interacting with  │
  │     - BroadcastReceiver executing onReceive()│
  │     - Service executing onCreate/onStart     │
  │     -> Never killed unless last resort        │
  ├─────────────────────────────────────────────┤
  │  2. Visible Process                          │
  │     - Activity visible but not in focus       │
  │     - Service running as foregroundService    │
  │     -> Killed only to maintain foreground     │
  ├─────────────────────────────────────────────┤
  │  3. Service Process                          │
  │     - Service started via startService()      │
  │     - Music playback, data sync, etc.         │
  │     -> Demoted to cached after 30+ minutes    │
  ├─────────────────────────────────────────────┤
  │  4. Cached Process                           │
  │     - Hidden Activity (after onStop)          │
  │     - Managed via LRU list                    │
  │     -> Kill target on low memory              │
  ├─────────────────────────────────────────────┤
  │  5. Empty Process                            │
  │     - No active components                    │
  │     - Retained only for caching purposes      │
  │     -> Killed first                           │
  └─────────────────────────────────────────────┘
  Importance: Low

  LowMemoryKiller (LMK) behavior:
    adj value  Process type           Kill threshold (example)
    ─────────────────────────────────────────
    0          Foreground             Not killed
    100        Visible                Below 72 MB
    200        Service                Below 64 MB
    700        Cached (recent)        Below 56 MB
    900        Cached (old)           Below 48 MB
    906        Cached (empty)         Below 40 MB
```

### 3.3 Android's 4 Major Components

Android apps are composed of 4 basic components. Each component functions as an independent entry point.

| Component | Role | Lifecycle | Usage Example |
|-----------|------|-----------|---------------|
| **Activity** | Screen with UI | Created->Started->Resumed->Paused->Stopped->Destroyed | Main screen, settings screen |
| **Service** | Background processing | Created->Started->Destroyed / Created->Bound->Destroyed | Music playback, sync |
| **BroadcastReceiver** | System event receiver | onReceive() only | Low battery notification, network changes |
| **ContentProvider** | Data sharing | onCreate() -> CRUD operations | Contacts, media store |

### 3.4 Code Example: Lifecycle Management

**Code Example 3: Android (Kotlin) -- Activity Lifecycle Implementation**

```kotlin
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner

/**
 * Comprehensive Android Activity lifecycle implementation example
 *
 * Understanding when each callback is invoked is essential
 * for preventing memory leaks and crashes.
 */
class MainLifecycleActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "LifecycleDemo"
        private const val KEY_COUNTER = "counter"
    }

    private var counter = 0

    // =========================================================
    // Activity Lifecycle Callbacks
    // =========================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Restore state (when recreated due to screen rotation, etc.)
        savedInstanceState?.let {
            counter = it.getInt(KEY_COUNTER, 0)
            Log.d(TAG, "State restored: counter = $counter")
        }

        // Process-level lifecycle monitoring
        ProcessLifecycleOwner.get().lifecycle.addObserver(
            AppLifecycleObserver()
        )

        Log.d(TAG, "onCreate: Activity has been created")
    }

    override fun onStart() {
        super.onStart()
        // Just before Activity becomes visible on screen
        // Perform UI updates and sensor registration here
        Log.d(TAG, "onStart: Activity has become visible")
    }

    override fun onResume() {
        super.onResume()
        // Activity has come to the foreground
        // Ready to accept user input
        Log.d(TAG, "onResume: Activity has become active")
    }

    override fun onPause() {
        super.onPause()
        // Another Activity has come to the foreground (may be partially visible)
        // Do not perform heavy operations here (causes ANR)
        Log.d(TAG, "onPause: Activity has been paused")
    }

    override fun onStop() {
        super.onStop()
        // Activity has become completely hidden
        // Save data and release resources
        Log.d(TAG, "onStop: Activity has been stopped")
    }

    override fun onDestroy() {
        super.onDestroy()
        // Activity is being destroyed
        // Final cleanup
        Log.d(TAG, "onDestroy: Activity has been destroyed")
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // Save state before configuration changes (screen rotation, etc.)
        outState.putInt(KEY_COUNTER, counter)
        Log.d(TAG, "State saved: counter = $counter")
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // Memory release request from OS
        when (level) {
            TRIM_MEMORY_UI_HIDDEN ->
                Log.d(TAG, "UI hidden - Can release UI cache")
            TRIM_MEMORY_RUNNING_LOW ->
                Log.w(TAG, "Memory low - Should release unnecessary resources")
            TRIM_MEMORY_RUNNING_CRITICAL ->
                Log.e(TAG, "Memory critical - Release resources immediately")
            TRIM_MEMORY_COMPLETE ->
                Log.e(TAG, "Will be killed first in background")
        }
    }

    // =========================================================
    // Application-wide Lifecycle Monitoring
    // =========================================================

    /**
     * App-wide lifecycle monitoring using ProcessLifecycleOwner
     * Detects when the app transitions to foreground/background
     */
    inner class AppLifecycleObserver : LifecycleEventObserver {
        override fun onStateChanged(
            source: LifecycleOwner,
            event: Lifecycle.Event
        ) {
            when (event) {
                Lifecycle.Event.ON_START ->
                    Log.d(TAG, "App transitioned to foreground")
                Lifecycle.Event.ON_STOP ->
                    Log.d(TAG, "App transitioned to background")
                else -> { /* Other events */ }
            }
        }
    }
}
```

---

## 4. Memory Management and Virtual Memory

### 4.1 iOS Memory Management: Jetsam

iOS does not have a swap area in the traditional sense. Excessive writes to NAND flash shorten storage lifespan. Instead, it manages memory through a mechanism called **jetsam**.

```
iOS Jetsam Memory Management Overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Physical Memory (e.g., 6 GB)
  ┌─────────────────────────────────────────────────┐
  │████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░│
  │◄── Kernel+Daemons ───►◄── App Space ──────────►│
  │     (~1-2 GB fixed)      (~4-5 GB shared)      │
  └─────────────────────────────────────────────────┘

  App Memory Limits (footprint limit):
  ┌──────────────┬──────────────┬──────────────────┐
  │ Device       │ Physical RAM │ App Limit (approx)│
  ├──────────────┼──────────────┼──────────────────┤
  │ iPhone SE(3) │ 4 GB         │ ~1.4 GB          │
  │ iPhone 15    │ 6 GB         │ ~2.8 GB          │
  │ iPhone 15 Pro│ 8 GB         │ ~3.5 GB          │
  │ iPad Pro M4  │ 16 GB        │ ~5.0 GB          │
  └──────────────┴──────────────┴──────────────────┘

  Jetsam Operation Flow:

  Memory usage increases
       │
       ▼
  ┌─────────────┐    No     ┌──────────────────┐
  │ Compressible?│─────────►│ Footprint        │
  │(WK compress) │          │ exceeded?        │
  └──────┬──────┘          └───────┬──────────┘
         │ Yes                     │ Yes
         ▼                         ▼
  ┌─────────────┐          ┌──────────────────┐
  │ Compress    │          │ Immediately kill  │
  │ Memory      │          │ the app (SIGKILL) │
  │ (WKdm/LZ4) │          │ -> crash log      │
  └──────┬──────┘          └──────────────────┘
         │
         ▼
  ┌─────────────┐
  │ Still       │─── No ──► Resume normal operation
  │ insufficient?│
  └──────┬──────┘
         │ Yes
         ▼
  ┌───────────────────────────────────────────┐
  │ Kill suspended apps in priority order     │
  │ (based on jetsam priority band)           │
  │                                           │
  │ Band 0-10:  Background apps (first)       │
  │ Band 10-20: Mail, Calendar, etc.          │
  │ Band 20+:   System daemons (last)         │
  └───────────────────────────────────────────┘
```

### 4.2 Android Memory Management: LowMemoryKiller and zRAM

Android uses LowMemoryKiller, an extension of Linux's OOM Killer, and additionally leverages zRAM for memory compression.

```
Android Memory Management Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Physical Memory (e.g., 8 GB)
  ┌────────────────────────────────────────────────────────┐
  │███████│██████████│░░░░░░░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
  │Kernel │  zRAM    │   App Space       │   File         │
  │       │(Compress)│                   │   Cache        │
  │~1.5GB │ ~2 GB    │    ~3 GB          │   ~1.5 GB      │
  └────────────────────────────────────────────────────────┘

  zRAM: In-memory compression
  ┌──────────┐ compress ┌──────────┐
  │ Page A   │────────►│Compressed│  Typically compressed to 50-70%
  │ (4 KB)   │         │ A (~2KB) │  No disk I/O
  └──────────┘         └──────────┘  <- Protects NAND lifespan

  LowMemoryKiller (lmkd) daemon:
  ┌─────────────────────────────────────────────┐
  │ 1. Periodically monitors /proc/meminfo      │
  │ 2. When free memory drops below threshold:  │
  │    a. Identify processes with high           │
  │       oom_adj_score (low importance)         │
  │    b. Measure memory pressure via PSI        │
  │       (Pressure Stall Information, Android   │
  │       10+)                                   │
  │    c. Send SIGKILL to target process         │
  │ 3. Repeat until free memory recovers         │
  └─────────────────────────────────────────────┘
```

### 4.3 iOS vs Android Memory Management Comparison

| Aspect | iOS | Android |
|--------|-----|---------|
| Swap | None (compression only) | zRAM (in-memory compression) |
| OOM Countermeasure | jetsam (proactive kill) | LowMemoryKiller (reactive kill) |
| Compression Algorithm | WKdm + LZ4 | LZ4 / LZO |
| GC | ARC (compile-time) | Tracing GC (ART) |
| Memory Warning | didReceiveMemoryWarning | onTrimMemory / onLowMemory |
| Shared Memory | Mach VM (copy-on-write) | Ashmem / mmap |
| App Limit | footprint limit (strict) | Configurable (largeHeap) |
| Compression Ratio | ~40-60% | ~30-50% |

### 4.4 Code Example: Memory Monitoring

**Code Example 4: iOS (Swift) -- Memory Usage Monitoring**

```swift
import Foundation
import os.log

/// Utility for monitoring iOS app memory usage
///
/// To avoid forced termination by jetsam,
/// periodically monitor memory usage and perform
/// cache cleanup when thresholds are exceeded.
class MemoryMonitor {

    private let logger = Logger(
        subsystem: "com.example.app",
        category: "memory"
    )

    private var timer: Timer?
    private let warningThreshold: UInt64  // In bytes

    /// Initialize with warning threshold in MB
    init(warningThresholdMB: UInt64 = 200) {
        self.warningThreshold = warningThresholdMB * 1024 * 1024
    }

    /// Get current app memory usage (bytes)
    func currentMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size
        ) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(
                to: integer_t.self,
                capacity: Int(count)
            ) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        guard result == KERN_SUCCESS else {
            return 0
        }
        return info.resident_size
    }

    /// Return memory usage in MB
    func currentMemoryMB() -> Double {
        Double(currentMemoryUsage()) / (1024.0 * 1024.0)
    }

    /// Start periodic monitoring (check at specified intervals)
    func startMonitoring(intervalSeconds: TimeInterval = 5.0) {
        timer = Timer.scheduledTimer(
            withTimeInterval: intervalSeconds,
            repeats: true
        ) { [weak self] _ in
            guard let self = self else { return }
            let usage = self.currentMemoryUsage()
            let usageMB = Double(usage) / (1024.0 * 1024.0)

            self.logger.info("Memory usage: \(usageMB, format: .fixed(precision: 1)) MB")

            if usage > self.warningThreshold {
                self.logger.warning(
                    "Memory warning: \(usageMB, format: .fixed(precision: 1)) MB " +
                    "(threshold: \(Double(self.warningThreshold) / (1024.0 * 1024.0)) MB)"
                )
                // Perform cache cleanup or resource reduction
                self.handleMemoryPressure()
            }
        }
    }

    /// Stop monitoring
    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }

    /// Handle memory pressure
    private func handleMemoryPressure() {
        // Clear image cache
        URLCache.shared.removeAllCachedResponses()

        // Release custom cache
        NotificationCenter.default.post(
            name: .init("MemoryPressureWarning"),
            object: nil
        )

        logger.info("Memory pressure response: Cache cleared")
    }
}

// Usage example:
// let monitor = MemoryMonitor(warningThresholdMB: 500)
// monitor.startMonitoring(intervalSeconds: 3.0)
// print("Current memory: \(monitor.currentMemoryMB()) MB")
```

---

## 5. Power Management and Scheduling

### 5.1 Mobile CPU Power Optimization

The biggest constraint for mobile devices is battery life. SoC (System on a Chip) integrates CPU, GPU, NPU, modem, and ISP (Image Signal Processor) into a single chip, reducing inter-chip communication power.

```
Mobile SoC Architecture (e.g., Apple A17 Pro / Snapdragon 8 Gen 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────┐
  │                    SoC Package                       │
  │                                                     │
  │  ┌───────────────────────────────────────────────┐  │
  │  │              CPU Cluster                       │  │
  │  │  ┌─────────────┐  ┌────────────────────────┐  │  │
  │  │  │ Performance │  │ Efficiency Cores        │  │  │
  │  │  │ Cores       │  │ (E-core)               │  │  │
  │  │  │ (P-core)    │  │ 4-6 cores              │  │  │
  │  │  │ 2-4 cores   │  │ 1-2 GHz               │  │  │
  │  │  │ 3-4 GHz     │  │ Low power              │  │  │
  │  │  │ High IPC    │  │ consumption            │  │  │
  │  │  └─────────────┘  └────────────────────────┘  │  │
  │  └───────────────────────────────────────────────┘  │
  │                                                     │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
  │  │   GPU    │ │  NPU/    │ │  ISP     │            │
  │  │(Graphics)│ │  Neural  │ │ (Camera  │            │
  │  │          │ │  Engine  │ │  Proc.)  │            │
  │  └──────────┘ └──────────┘ └──────────┘            │
  │                                                     │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
  │  │ Modem   │ │ Secure   │ │ Memory   │            │
  │  │ (5G/LTE)│ │ Enclave  │ │Controller│            │
  │  └──────────┘ └──────────┘ └──────────┘            │
  └─────────────────────────────────────────────────────┘

  DVFS (Dynamic Voltage and Frequency Scaling):
  ─────────────────────────────────────────────
  Dynamically adjusts CPU voltage and frequency based on load

  Power consumption ∝ C x V^2 x f
    C = Capacitance (circuit complexity)
    V = Operating voltage
    f = Clock frequency

  -> Halving the voltage reduces power consumption to 1/4
  -> Light tasks use E-core + low frequency
  -> Only heavy tasks use P-core + high frequency
```

### 5.2 iOS Power Management

iOS implements the following multi-layered power management mechanisms.

| Feature | Description | Effect |
|---------|-------------|--------|
| **Background App Refresh** | Optimizes background update timing based on usage patterns | Eliminates unnecessary launches |
| **App Nap** | Suppresses timer and network activity for hidden apps | Minimizes CPU usage |
| **Coalesced Timer** | Batches timers from multiple apps | Reduces CPU wake-up count |
| **Discretionary Background Tasks** | OS determines optimal execution timing | Executes during charging / WiFi connection |
| **Low Power Mode** | User-activated power saving mode | Stops background tasks, reduces frame rate, disables 5G |
| **Thermal Throttling** | Graduated performance limiting based on thermal sensors | Prevents overheating |

### 5.3 Android Power Management: Doze and App Standby

Doze mode, introduced in Android 6.0, progressively strengthens restrictions when the device is stationary, screen-off, and not charging.

```
Android Doze Mode Progressive Restrictions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Screen ON                  Screen OFF + Stationary + Not charging
  ──────►                   ──────────────────────►
                                    Time elapsed
  ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │Normal│  │ Light    │  │ Deep     │  │ Deep     │
  │Opera-│->│ Doze    │->│ Doze    │->│ Doze    │
  │tion  │  │ (Light)  │  │ (Initial)│  │ (Full)   │
  └──────┘  └──────────┘  └──────────┘  └──────────┘

  Light Doze (after a few minutes):
  ├── Network access: Restricted
  ├── Jobs/sync: Deferred
  ├── Alarms: Deferred (setExactAndAllowWhileIdle allowed)
  └── GPS: Stopped

  Deep Doze (after ~30 minutes):
  ├── Network access: Stopped
  ├── WiFi scans: Stopped
  ├── wakelocks: Ignored
  ├── Alarms: Deferred
  └── Jobs/sync: All deferred

  Maintenance Windows:
  ┌───┐     ┌───┐          ┌───┐               ┌───┐
  │   │     │   │          │   │               │   │
  │Pro│     │Pro│          │Pro│               │Pro│
  │ces│     │ces│          │ces│               │ces│
  │   │     │   │          │   │               │   │
  └─┬─┘     └─┬─┘          └─┬─┘               └─┬─┘
    │  Doze    │    Doze      │      Doze          │
    ├─────────┤              │                    │
    │ Short   │              │                    │
    │ interval│              │                    │
    ├─────────┴──────────────┤                    │
    │  Intervals gradually   │                    │
    │  increase              │                    │
    ├────────────────────────┴────────────────────┤
    │       Maximum interval is several hours     │

  App Standby Buckets (Android 9+):
  ┌───────────┬──────────────┬──────────────────────┐
  │ Bucket    │ Job Limit    │ Condition             │
  ├───────────┼──────────────┼──────────────────────┤
  │ Active    │ No limit     │ Currently in use      │
  │ Working   │ Once per 2hr │ Frequently used       │
  │ Frequent  │ Once per 8hr │ Regularly used        │
  │ Rare      │ Once per 24hr│ Rarely used           │
  │ Restricted│ Once per day │ Almost never (And.12) │
  └───────────┴──────────────┴──────────────────────┘
```

### 5.4 CPU Scheduling Comparison

| Aspect | iOS (XNU) | Android (Linux) |
|--------|-----------|----------------|
| Scheduler | Mach priority scheduler | CFS (Completely Fair Scheduler) |
| Priority Levels | 128 levels (0-127) | nice value (-20 to 19) + RT priority |
| Real-time | FIFO / RR (Mach RT threads) | SCHED_FIFO / SCHED_RR |
| QoS Classification | UserInteractive, UserInitiated, Utility, Background | THREAD_PRIORITY_* |
| big.LITTLE Control | Fully OS controlled (proprietary algorithm) | EAS (Energy Aware Scheduling) |
| UI Priority | Main Thread highest priority | RenderThread priority (Android 11+) |

---

## 6. Security Model and Sandboxing

### 6.1 Defense in Depth Architecture

Mobile OS security is designed with consistent defense in depth from hardware to application layers.

```
Mobile OS Security Defense in Depth Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Attacker ─────────────────────────────────────────────►

  Layer 7: Application Layer
  ┌─────────────────────────────────────────────────┐
  │ - App Store / Play Store review                 │
  │ - Code signing (mandatory)                      │
  │ - Runtime permission checks                     │
  └─────────────────────────────┬───────────────────┘
                                │ If breached
  Layer 6: Sandbox Layer        ▼
  ┌─────────────────────────────────────────────────┐
  │ - iOS: App Sandbox (Seatbelt)                   │
  │ - Android: SELinux + UID isolation + seccomp-bpf│
  │ - Each app runs in an isolated environment       │
  └─────────────────────────────┬───────────────────┘
                                │ If breached
  Layer 5: Runtime Protection   ▼
  ┌─────────────────────────────────────────────────┐
  │ - ASLR (Address Space Layout Randomization)     │
  │ - Stack Canary / Stack Protector                │
  │ - PAC (Pointer Authentication, ARM v8.3+)       │
  │ - MTE (Memory Tagging Extension, Android 14+)   │
  │ - CFI (Control Flow Integrity)                  │
  └─────────────────────────────┬───────────────────┘
                                │ If breached
  Layer 4: Kernel Protection    ▼
  ┌─────────────────────────────────────────────────┐
  │ - KTRR / CTRR (iOS: kernel text read-only)      │
  │ - KASLR (Kernel ASLR)                           │
  │ - W^X (Write XOR Execute) policy                │
  │ - PPL (Page Protection Layer, iOS)              │
  └─────────────────────────────┬───────────────────┘
                                │ If breached
  Layer 3: Firmware Layer       ▼
  ┌─────────────────────────────────────────────────┐
  │ - Secure Boot Chain (chain of boot verification) │
  │ - iOS: iBoot -> Kernel -> kext all stages signed │
  │ - Android: Verified Boot (AVB 2.0)              │
  │ - Bootloader lock                               │
  └─────────────────────────────┬───────────────────┘
                                │ If breached
  Layer 2: Hardware Security    ▼
  ┌─────────────────────────────────────────────────┐
  │ - Secure Enclave (iOS) / Titan M (Pixel)        │
  │ - TrustZone (ARM)                               │
  │ - eFuse (one-time programmable)                  │
  │ - Physical attack countermeasures (tamper detect)│
  └─────────────────────────────┬───────────────────┘
                                │ If breached
  Layer 1: Cryptographic Foundation ▼
  ┌─────────────────────────────────────────────────┐
  │ - FBE (File-Based Encryption)                   │
  │ - AES-256-XTS (storage encryption)              │
  │ - Hardware-bound keys (non-extractable from SEP)│
  │ - Secure Element (eSIM, NFC payments)           │
  └─────────────────────────────────────────────────┘
```

### 6.2 Sandbox Implementation Details

**iOS Sandbox (Seatbelt / sandbox_init)**

In iOS, each app is assigned a dedicated container directory, and escaping from it is essentially impossible.

```
iOS App Sandbox Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━

  /var/mobile/Containers/
  ├── Bundle/
  │   └── Application/
  │       └── <UUID>/
  │           └── MyApp.app/     <- Read-only
  │               ├── MyApp      (executable binary)
  │               ├── Info.plist
  │               └── Assets/
  │
  └── Data/
      └── Application/
          └── <UUID>/            <- App-specific data area
              ├── Documents/     <- User data (iCloud sync capable)
              ├── Library/
              │   ├── Caches/    <- Cache (OS may delete)
              │   └── Preferences/ <- UserDefaults
              ├── tmp/           <- Temporary files (OS may delete)
              └── SystemData/    <- System-managed data

  Access Control:
  ┌────────────────────────┬──────────────────────┐
  │ Resource               │ Access               │
  ├────────────────────────┼──────────────────────┤
  │ Own app container      │ Read/write           │
  │ Other app containers   │ No access            │
  │ System files           │ No access            │
  │ Camera/Microphone      │ User permission req. │
  │ Location               │ User permission req. │
  │ Contacts/Calendar      │ User permission req. │
  │ Network                │ Permission (iOS 14+) │
  │ Bluetooth              │ User permission req. │
  │ Keychain               │ Same Team ID only    │
  └────────────────────────┴──────────────────────┘
```

**Android Security Boundaries**

Android combines Linux UID-based isolation with SELinux mandatory access control.

```
Android App Isolation Model
━━━━━━━━━━━━━━━━━━━━━━━━

  At install time:
  ┌──────────────────────────────────────────┐
  │ PackageManager assigns a unique          │
  │ Linux UID/GID to each app               │
  │                                          │
  │ com.example.app1 -> UID 10045            │
  │ com.example.app2 -> UID 10046            │
  │ com.example.app3 -> UID 10047            │
  └──────────────────────────────────────────┘

  Data directories:
  /data/data/com.example.app1/   (accessible only by UID 10045)
  ├── databases/                 (SQLite databases)
  ├── shared_prefs/              (SharedPreferences XML)
  ├── files/                     (internal storage)
  └── cache/                     (cache)

  SELinux Policy (Android 5.0+):
  ┌────────────────────────────────────────────────┐
  │ All apps are assigned SELinux contexts          │
  │                                                │
  │ untrusted_app : Normal third-party apps         │
  │ platform_app  : System partition apps           │
  │ priv_app      : Privileged apps                 │
  │ isolated_app  : WebView renderers, etc. (min)   │
  │                                                │
  │ Policy example:                                │
  │ allow untrusted_app app_data_file:file          │
  │   { read write create };                       │
  │ neverallow untrusted_app system_data_file:file  │
  │   { read write };                              │
  └────────────────────────────────────────────────┘

  Scoped Storage (Android 10+):
  ┌────────────────────────────────────────────────┐
  │ Restricts external storage access to           │
  │ MediaStore, preventing direct access to        │
  │ other apps' files                              │
  │                                                │
  │ /storage/emulated/0/                           │
  │ ├── DCIM/     <- via MediaStore.Images         │
  │ ├── Music/    <- via MediaStore.Audio           │
  │ ├── Download/ <- SAF (Storage Access Framework) │
  │ └── Android/data/com.example.app/              │
  │               <- App-exclusive (no other access)│
  └────────────────────────────────────────────────┘
```

### 6.3 Evolution of Permission Models

| Version | iOS | Android |
|---------|-----|---------|
| Initial | All permissions at install | All permissions at install |
| Transitional | iOS 6: Runtime permission for some | Android 6.0: Runtime for dangerous |
| Current | iOS 14+: Approximate location, limited photo access, App Tracking Transparency | Android 12+: Approximate location, nearby device permission, photo picker |
| Latest | iOS 17+: Permission reset, communication safety | Android 14+: Partial photo/video access, health data permission |

---

## 7. Inter-Process Communication and Push Notifications

### 7.1 IPC Mechanism Comparison

```
IPC Mechanisms in Mobile OS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  iOS IPC:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ┌─────────┐  Mach Port   ┌─────────┐             │
  │  │ App A   │─────────────►│ App B   │  Limited     │
  │  └─────────┘  (via kernel) └─────────┘             │
  │                                                    │
  │  Primary IPC mechanisms:                           │
  │  1. XPC Services    - Privilege-separated helpers   │
  │  2. URL Scheme      - Deep links between apps       │
  │  3. Universal Links - App launch from HTTP URL      │
  │  4. App Groups      - Data sharing within same dev  │
  │  5. UIPasteboard    - Via clipboard                 │
  │  6. Extensions      - Share, Today, Action          │
  │  7. App Intents     - Siri / Shortcuts integration  │
  │                                                    │
  │  Constraint: Direct communication between           │
  │             arbitrary apps is not possible           │
  │             OS mediates everything                   │
  └────────────────────────────────────────────────────┘

  Android IPC (Binder):
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ┌─────────┐   Binder    ┌─────────┐              │
  │  │ App A   │────────────►│ App B   │              │
  │  │ (Client)│  /dev/binder│(Server) │              │
  │  └─────────┘     │       └─────────┘              │
  │                  │                                 │
  │          ┌───────┴───────┐                         │
  │          │ Binder Driver │  Kernel space            │
  │          │ (single copy) │  copy_from_user ->       │
  │          │               │  target's mmap region    │
  │          └───────────────┘                         │
  │                                                    │
  │  Primary IPC mechanisms:                           │
  │  1. Intent         - Inter-component messaging      │
  │  2. AIDL           - Cross-process method calls     │
  │  3. ContentProvider- Structured data sharing        │
  │  4. Messenger      - Handler-based messaging        │
  │  5. BroadcastReceiver - System event notification   │
  │  6. FileProvider   - File sharing (via URI)         │
  │                                                    │
  │  Binder features:                                  │
  │  - Single memory copy transfer (send->kernel mmap) │
  │  - Auto-verifies caller UID/PID (anti-spoofing)    │
  │  - Reference count management (death notification)  │
  │  - 16MB transaction buffer limit                    │
  └────────────────────────────────────────────────────┘
```

### 7.2 Push Notification Architecture

Push notifications are a mechanism where, instead of each app maintaining its own persistent server connection, the OS centrally manages a single shared connection.

```
Push Notification Architecture Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  iOS (APNs: Apple Push Notification service):
  ┌──────────┐    HTTPS     ┌──────────┐   TLS persistent ┌──────────┐
  │ App      │─────────────►│  APNs    │───────────────►│  iOS     │
  │ Server   │  Certificate │  Server  │  Port 5223     │  Device  │
  └──────────┘  auth        └──────────┘                └──────────┘
       │                         │                          │
       │ 1. Get device token     │                          │
       │◄────────────────────────┤◄─────────────────────────┤
       │                         │ 2. Send notification      │
       │────────────────────────►│  payload                  │
       │                         │ 3. Deliver to device      │
       │                         │─────────────────────────►│
       │                         │                     4. Process in app
       │                         │                     UNUserNotification
       │                         │                     Center handles it

  Android (FCM: Firebase Cloud Messaging):
  ┌──────────┐    HTTPS     ┌──────────┐   XMPP/HTTPS   ┌──────────┐
  │ App      │─────────────►│  FCM     │───────────────►│ Android  │
  │ Server   │  API key auth│  Server  │  via GMS       │  Device  │
  └──────────┘              └──────────┘                └──────────┘
       │                         │                          │
       │ 1. Get registration     │                          │
       │    token                │                          │
       │◄────────────────────────┤◄─────────────────────────┤
       │                         │ 2. Send message          │
       │────────────────────────►│                          │
       │                         │ 3. Deliver via GCM       │
       │                         │    Connection Service     │
       │                         │─────────────────────────►│
       │                         │                     4. onMessageReceived
       │                         │                     FirebaseMessaging
       │                         │                     Service handles it

  Comparison:
  ┌─────────────────┬──────────────────┬──────────────────┐
  │ Item            │ APNs (iOS)       │ FCM (Android)    │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Protocol        │ Proprietary (TLS)│ XMPP / HTTP/2    │
  │ Payload limit   │ 4 KB             │ 4 KB (data)      │
  │ Priority        │ 5 (immediate)/1-4│ High / Normal    │
  │ Topic subscribe │ Yes              │ Yes              │
  │ Silent notif.   │ content-available│ data message     │
  │ Doze delivery   │ N/A              │ High only        │
  │ GMS dependency  │ None (OS built-in)│ Yes (GMS req.)  │
  │ Reliability     │ Best Effort      │ Best Effort      │
  └─────────────────┴──────────────────┴──────────────────┘
```

### 7.3 Code Example: Push Notification Implementation

**Code Example 5: Android (Kotlin) -- FCM Notification Reception and Processing**

```kotlin
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

/**
 * Firebase Cloud Messaging (FCM) notification receiver service
 *
 * This service handles the following two types of messages:
 * 1. Notification messages (notification) - Automatically displayed by OS
 * 2. Data messages (data) - App controls the processing
 *
 * Even during Doze mode, High Priority messages are delivered.
 */
class AppFirebaseMessagingService : FirebaseMessagingService() {

    companion object {
        private const val TAG = "FCMService"
        private const val CHANNEL_ID = "default_channel"
        private const val CHANNEL_NAME = "General Notifications"
    }

    /**
     * Called when the FCM token is refreshed
     * The new token must be sent to the app server
     */
    override fun onNewToken(token: String) {
        super.onNewToken(token)
        Log.d(TAG, "FCM token updated: $token")
        sendTokenToServer(token)
    }

    /**
     * Called when a message is received
     *
     * Note:
     * - Foreground: Both notification + data are processed here
     * - Background: OS displays notification, only data comes here
     */
    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        super.onMessageReceived(remoteMessage)

        Log.d(TAG, "From: ${remoteMessage.from}")
        Log.d(TAG, "Message ID: ${remoteMessage.messageId}")

        // Process data payload
        if (remoteMessage.data.isNotEmpty()) {
            Log.d(TAG, "Data: ${remoteMessage.data}")
            handleDataMessage(remoteMessage.data)
        }

        // Process notification payload (when in foreground)
        remoteMessage.notification?.let { notification ->
            Log.d(TAG, "Notification title: ${notification.title}")
            Log.d(TAG, "Notification body: ${notification.body}")
            showNotification(
                title = notification.title ?: "Notification",
                body = notification.body ?: "",
                data = remoteMessage.data
            )
        }
    }

    /**
     * Process data messages
     * Used for background sync and silent updates
     */
    private fun handleDataMessage(data: Map<String, String>) {
        val type = data["type"] ?: return
        when (type) {
            "sync" -> {
                // Schedule background data sync
                Log.d(TAG, "Sync request received")
            }
            "update" -> {
                // Update in-app data
                val payload = data["payload"]
                Log.d(TAG, "Update data: $payload")
            }
            "silent" -> {
                // Silent notification (no UI display)
                Log.d(TAG, "Silent processing executed")
            }
        }
    }

    /**
     * Display notification
     * NotificationChannel is required for Android 8.0+
     */
    private fun showNotification(
        title: String,
        body: String,
        data: Map<String, String>
    ) {
        val notificationManager = getSystemService(
            Context.NOTIFICATION_SERVICE
        ) as NotificationManager

        // Create notification channel for Android 8.0+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "General app notifications"
                enableVibration(true)
            }
            notificationManager.createNotificationChannel(channel)
        }

        // Set Activity to open on tap
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP
            data.forEach { (key, value) -> putExtra(key, value) }
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_ONE_SHOT or PendingIntent.FLAG_IMMUTABLE
        )

        // Build and display notification
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(body)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .build()

        notificationManager.notify(
            System.currentTimeMillis().toInt(),
            notification
        )
    }

    /** Send FCM token to app server */
    private fun sendTokenToServer(token: String) {
        // Implementation: Send via HTTPS POST to API server
        // Typically using Retrofit or OkHttp
        Log.d(TAG, "Sending token to server: $token")
    }
}
```

---

## 8. Sensors and Hardware Abstraction Layer

### 8.1 Sensor Array in Mobile Devices

Modern smartphones are equipped with more than 10 types of sensors. These sensors provide a unified API to applications through the OS's Hardware Abstraction Layer (HAL).

```
Mobile Device Sensor Inventory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────┐
  │              Smartphone Cross-Section             │
  │                                                  │
  │  [Front Camera] [Proximity] [Ambient] [Face ID]  │
  │  ┌────────────────────────────────────────────┐  │
  │  │                                            │  │
  │  │           Display                          │  │
  │  │         (Built-in touch sensor)            │  │
  │  │         (Under-display fingerprint)        │  │
  │  │                                            │  │
  │  ├────────────────────────────────────────────┤  │
  │  │  [Accel]  [Gyro]  [Mag]  [Barometer]      │  │
  │  │  [GPS/GNSS]  [UWB]  [WiFi RTT]           │  │
  │  │  [NFC]  [Temp]  [Mic x2-4]               │  │
  │  │  [Vibration Motor (Haptic Engine)]        │  │
  │  ├────────────────────────────────────────────┤  │
  │  │  [Rear Camera x2-4] [LiDAR] [ToF]        │  │
  │  └────────────────────────────────────────────┘  │
  └──────────────────────────────────────────────────┘

  Sensor Classification:
  ┌──────────────┬────────────────────┬───────────────────┐
  │ Category     │ Sensor             │ Usage              │
  ├──────────────┼────────────────────┼───────────────────┤
  │ Motion       │ Accelerometer      │ Step count, tilt   │
  │              │ Gyroscope          │ Rotation, AR       │
  │              │ Magnetometer       │ Compass            │
  ├──────────────┼────────────────────┼───────────────────┤
  │ Environment  │ Barometer          │ Altitude, weather  │
  │              │ Ambient light      │ Screen brightness  │
  │              │ Temperature sensor │ Battery/SoC temp   │
  │              │ Humidity sensor    │ Env. monitoring    │
  ├──────────────┼────────────────────┼───────────────────┤
  │ Location     │ GPS/GNSS           │ Positioning        │
  │              │ WiFi RTT           │ Indoor positioning │
  │              │ UWB                │ Close-range precise│
  ├──────────────┼────────────────────┼───────────────────┤
  │ Proximity/   │ Proximity sensor   │ Screen off on call │
  │ Depth        │ LiDAR/ToF          │ 3D spatial scan    │
  │              │ Structured light   │ Face authentication│
  ├──────────────┼────────────────────┼───────────────────┤
  │ Biometric    │ Fingerprint sensor │ Authentication     │
  │              │ TrueDepth camera   │ Face ID            │
  ├──────────────┼────────────────────┼───────────────────┤
  │ Communication│ NFC                │ Contactless payment│
  │              │ Bluetooth 5.x     │ Peripheral connect │
  │              │ WiFi 6E/7          │ High-speed comm.   │
  └──────────────┴────────────────────┴───────────────────┘
```

### 8.2 HAL (Hardware Abstraction Layer) Design

```
iOS IOKit vs Android HAL -- Driver Abstraction Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  iOS (IOKit):
  ┌─────────────────────────────────────────┐
  │ Application (Swift/ObjC)                │
  ├─────────────────────────────────────────┤
  │ Framework (CoreMotion, CoreLocation)    │
  ├─────────────────────────────────────────┤
  │ IOKit User Client (user-space driver)   │
  ├─────────────────────────────────────────┤
  │ IOKit Kernel (C++ object hierarchy)     │
  │  IOService -> IOHIDDevice               │
  │           -> IOAccelerator              │
  │           -> AppleSensorKit             │
  ├─────────────────────────────────────────┤
  │ Hardware                                │
  └─────────────────────────────────────────┘

  Android (Post-Project Treble HIDL/AIDL HAL):
  ┌─────────────────────────────────────────┐
  │ Application (Kotlin/Java)               │
  ├─────────────────────────────────────────┤
  │ Framework API (SensorManager, etc.)     │
  ├─────────────────────────────────────────┤
  │ System Server (system_server process)   │
  ├─────────────────────────────────────────┤
  │ HIDL/AIDL HAL Interface                 │
  │  (android.hardware.sensors@2.1)         │
  │  <- Boundary between vendor and framework│
  ├─────────────────────────────────────────┤
  │ Vendor HAL Implementation (vendor-      │
  │  provided: Qualcomm, Samsung, MediaTek) │
  ├─────────────────────────────────────────┤
  │ Kernel Driver                           │
  ├─────────────────────────────────────────┤
  │ Hardware                                │
  └─────────────────────────────────────────┘

  Significance of Project Treble:
  ┌──────────────────────────────────────────────┐
  │ Before Treble:                                │
  │  OS update = Google -> SoC -> OEM -> Carrier  │
  │  HAL modifications needed at each stage       │
  │  -> Updates took months to years              │
  │                                               │
  │ After Treble:                                 │
  │  HAL interface stabilized                      │
  │  No vendor HAL recompilation on OS update     │
  │  Testable with Generic System Image (GSI)     │
  │  -> Update speed significantly improved        │
  └──────────────────────────────────────────────┘
```

### 8.3 Low-Power Coprocessors

Dedicated chips that process sensor data without waking the main CPU dramatically improve power efficiency.

| Coprocessor | Platform | Function | Power Consumption |
|-------------|----------|----------|-------------------|
| Apple Motion Coprocessor (M-series) | iOS | Always-on monitoring of accelerometer, gyroscope, barometer; step counting | ~1/100 of main CPU |
| Sensor Hub (Qualcomm SSC) | Android (Snapdragon) | Always-on sensor processing, Activity Recognition | Few mW |
| Samsung CHUB | Android (Exynos) | Environmental sensor processing, gesture detection | Few mW |
| Google Tensor Context Hub | Android (Pixel) | Face detection, music recognition, environmental context inference | Few mW |

### 8.4 Code Example: Sensor Usage

**Code Example 6: iOS (Swift) -- Motion Detection Using CoreMotion**

```swift
import CoreMotion
import Foundation

/**
 * Device motion detection using CoreMotion
 *
 * The Motion Coprocessor enables low-power sensor data
 * collection even when the app is in the background.
 *
 * Used for step counting, stair climbing detection,
 * vehicle riding detection, etc.
 */
class MotionDetector {

    private let motionManager = CMMotionManager()
    private let pedometer = CMPedometer()
    private let activityManager = CMMotionActivityManager()
    private let altimeter = CMAltimeter()

    // ===================================================
    // Raw Accelerometer and Gyroscope Data
    // ===================================================

    /// Start accelerometer (for games, AR)
    func startAccelerometer(
        interval: TimeInterval = 0.1,
        handler: @escaping (CMAccelerometerData) -> Void
    ) {
        guard motionManager.isAccelerometerAvailable else {
            print("Accelerometer not available")
            return
        }

        motionManager.accelerometerUpdateInterval = interval
        motionManager.startAccelerometerUpdates(
            to: .main
        ) { data, error in
            guard let data = data, error == nil else { return }
            // data.acceleration.x/y/z includes gravity (-1.0 ~ +1.0 G)
            handler(data)
        }
    }

    /// Device motion (sensor fusion result)
    func startDeviceMotion(
        interval: TimeInterval = 0.05,
        handler: @escaping (CMDeviceMotion) -> Void
    ) {
        guard motionManager.isDeviceMotionAvailable else { return }

        motionManager.deviceMotionUpdateInterval = interval
        motionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: .main
        ) { motion, error in
            guard let motion = motion, error == nil else { return }
            // motion.attitude    : Orientation (roll, pitch, yaw)
            // motion.rotationRate: Angular velocity
            // motion.gravity     : Gravity component
            // motion.userAcceleration: User acceleration (gravity removed)
            handler(motion)
        }
    }

    // ===================================================
    // Step Counting (Using Motion Coprocessor)
    // ===================================================

    /// Start real-time step counting
    func startPedometer(
        handler: @escaping (Int, Double) -> Void
    ) {
        guard CMPedometer.isStepCountingAvailable() else {
            print("Step counting not available")
            return
        }

        pedometer.startUpdates(from: Date()) { data, error in
            guard let data = data, error == nil else { return }
            let steps = data.numberOfSteps.intValue
            let distance = data.distance?.doubleValue ?? 0
            handler(steps, distance)
        }
    }

    // ===================================================
    // Activity Recognition (Walking/Running/Driving/Cycling)
    // ===================================================

    /// Detect user activity type
    func startActivityRecognition(
        handler: @escaping (String) -> Void
    ) {
        guard CMMotionActivityManager.isActivityAvailable() else {
            return
        }

        activityManager.startActivityUpdates(
            to: .main
        ) { activity in
            guard let activity = activity else { return }
            var type = "Unknown"
            if activity.walking   { type = "Walking" }
            if activity.running   { type = "Running" }
            if activity.cycling   { type = "Cycling" }
            if activity.automotive{ type = "Driving" }
            if activity.stationary{ type = "Stationary" }
            handler(type)
        }
    }

    // ===================================================
    // Cleanup
    // ===================================================

    func stopAll() {
        motionManager.stopAccelerometerUpdates()
        motionManager.stopDeviceMotionUpdates()
        pedometer.stopUpdates()
        activityManager.stopActivityUpdates()
    }
}

// Usage example:
// let detector = MotionDetector()
// detector.startPedometer { steps, distance in
//     print("Steps: \(steps), Distance: \(distance)m")
// }
// detector.startActivityRecognition { type in
//     print("Current activity: \(type)")
// }
```

### 8.5 Code Example: Android Sensor Usage

**Code Example 7: Android (Kotlin) -- Acceleration Detection with SensorManager**

```kotlin
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import kotlin.math.sqrt

/**
 * Acceleration and shake detection using Android SensorManager
 *
 * Demonstrates unified access to vendor-specific sensor
 * hardware through the HAL.
 *
 * Always unregister when not needed for power efficiency.
 */
class ShakeDetector(
    private val context: Context,
    private val onShake: () -> Unit
) : SensorEventListener {

    companion object {
        private const val TAG = "ShakeDetector"
        private const val SHAKE_THRESHOLD = 12.0f  // m/s^2
        private const val MIN_TIME_BETWEEN_SHAKES = 1000L  // ms
    }

    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private var lastShakeTime = 0L

    /** Register and start sensor */
    fun start() {
        val accelerometer = sensorManager.getDefaultSensor(
            Sensor.TYPE_LINEAR_ACCELERATION  // Gravity component removed
        )
        if (accelerometer != null) {
            sensorManager.registerListener(
                this,
                accelerometer,
                SensorManager.SENSOR_DELAY_UI  // ~60ms interval
            )
            Log.d(TAG, "Accelerometer started")
        } else {
            Log.w(TAG, "LINEAR_ACCELERATION sensor not available")
        }
    }

    /** Unregister sensor */
    fun stop() {
        sensorManager.unregisterListener(this)
        Log.d(TAG, "Accelerometer stopped")
    }

    override fun onSensorChanged(event: SensorEvent) {
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        // Calculate composite acceleration from 3 axes
        val magnitude = sqrt(x * x + y * y + z * z)

        if (magnitude > SHAKE_THRESHOLD) {
            val now = System.currentTimeMillis()
            if (now - lastShakeTime > MIN_TIME_BETWEEN_SHAKES) {
                lastShakeTime = now
                Log.d(TAG, "Shake detected: acceleration = $magnitude m/s^2")
                onShake()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        Log.d(TAG, "Accuracy changed: ${sensor.name} -> $accuracy")
    }

    /** Enumerate all available sensors */
    fun listAllSensors(): List<String> {
        return sensorManager.getSensorList(Sensor.TYPE_ALL).map {
            "${it.name} (type=${it.type}, vendor=${it.vendor}, " +
            "power=${it.power}mA, resolution=${it.resolution})"
        }
    }
}

// Usage example (inside Activity):
// val shakeDetector = ShakeDetector(this) {
//     Toast.makeText(this, "Shake detected!", Toast.LENGTH_SHORT).show()
// }
// override fun onResume() { super.onResume(); shakeDetector.start() }
// override fun onPause() { super.onPause(); shakeDetector.stop() }
```

---

## 9. App Development and Build Pipeline

### 9.1 Development Environment Comparison

| Item | iOS | Android |
|------|-----|---------|
| Official IDE | Xcode | Android Studio |
| Build System | xcodebuild / Swift Package Manager | Gradle (Kotlin DSL) |
| Language | Swift (mainstream), Objective-C (legacy) | Kotlin (mainstream), Java (legacy) |
| UI Framework | SwiftUI (declarative), UIKit (imperative) | Jetpack Compose (declarative), View (imperative) |
| Testing | XCTest, XCUITest | JUnit, Espresso, Compose Testing |
| Profiler | Instruments | Android Profiler (CPU, Memory, Network) |
| Package Management | SPM, CocoaPods | Gradle dependencies, Maven Central |
| CI/CD | Xcode Cloud, Fastlane | GitHub Actions, Fastlane |
| Distribution | App Store Connect, TestFlight | Google Play Console, Firebase App Distribution |
| Code Signing | Required (Provisioning Profile + Certificate) | Required (APK/AAB signing) |
| Minimum Target | Typically iOS N-2 (currently iOS 16+) | minSdk 24+ (Android 7.0) is common |

### 9.2 Build Pipeline Overview

```
Mobile App CI/CD Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Developer
    │
    ├── git push
    │
    ▼
  ┌──────────────────────────────────────────────────────┐
  │                  CI Server                            │
  │                                                      │
  │  1. Fetch source code (git clone)                    │
  │     ▼                                                │
  │  2. Resolve dependencies                             │
  │     iOS:  swift package resolve / pod install        │
  │     Android: ./gradlew dependencies                  │
  │     ▼                                                │
  │  3. Static analysis                                  │
  │     iOS:  SwiftLint, SwiftFormat                     │
  │     Android: ktlint, detekt, Android Lint            │
  │     ▼                                                │
  │  4. Unit tests                                       │
  │     iOS:  xcodebuild test -scheme MyApp              │
  │     Android: ./gradlew testDebugUnitTest             │
  │     ▼                                                │
  │  5. Build                                            │
  │     iOS:  xcodebuild archive -> .xcarchive           │
  │     Android: ./gradlew assembleRelease -> .apk/.aab  │
  │     ▼                                                │
  │  6. Code signing                                     │
  │     iOS:  Provisioning Profile + Distribution Cert   │
  │     Android: Keystore-based APK/AAB signing          │
  │     ▼                                                │
  │  7. UI test / E2E test                               │
  │     iOS:  XCUITest (Simulator / Device Farm)         │
  │     Android: Espresso (Emulator / Firebase Test Lab) │
  │     ▼                                                │
  │  8. Distribution                                     │
  │     ┌──────────────────┬──────────────────────┐      │
  │     │ Test distribution │ Production           │      │
  │     │ TestFlight       │ App Store            │      │
  │     │ Firebase App Dist│ Google Play          │      │
  │     │ DeployGate       │ (staged rollout      │      │
  │     │                  │  10->100%)           │      │
  │     └──────────────────┴──────────────────────┘      │
  └──────────────────────────────────────────────────────┘
```

### 9.3 Cross-Platform Development

Approaches to developing for both iOS and Android from a single codebase are widely adopted.

| Framework | Language | Rendering | Performance | Adopters |
|-----------|----------|-----------|-------------|----------|
| **Flutter** | Dart | Custom engine (Skia/Impeller) | High (native-equivalent) | Google, BMW, Alibaba |
| **React Native** | JavaScript/TypeScript | Native UI (Bridge/JSI) | Medium-High | Meta, Microsoft, Shopify |
| **Kotlin Multiplatform** | Kotlin | Native UI (per platform) | High (shared logic) | Netflix, VMware, Philips |
| **MAUI (.NET)** | C# | Native UI (Handlers) | Medium | Microsoft, UPS |
| **Capacitor/Ionic** | Web (HTML/CSS/JS) | WebView | Low-Medium | Burger King, Sanvello |

### 9.4 Code Example: Gradle Build Configuration

**Code Example 8: Android (Kotlin DSL) -- build.gradle.kts Configuration**

```kotlin
// app/build.gradle.kts
// Android app build configuration

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.hilt)
    alias(libs.plugins.ksp)
}

android {
    namespace = "com.example.mobileapp"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.mobileapp"
        minSdk = 26          // Android 8.0 and above
        targetSdk = 35       // Latest API level
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner =
            "androidx.test.runner.AndroidJUnitRunner"

        // Embed build information in BuildConfig
        buildConfigField(
            "String", "BUILD_TIME",
            "\"${java.time.Instant.now()}\""
        )
    }

    // Build variants: debug / staging / release
    buildTypes {
        debug {
            isDebuggable = true
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
        }
        create("staging") {
            isDebuggable = false
            applicationIdSuffix = ".staging"
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        release {
            isMinifyEnabled = true    // Code shrinking with R8
            isShrinkResources = true  // Remove unused resources
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    // Jetpack Compose configuration
    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // Jetpack Compose
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.navigation)

    // Lifecycle
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)

    // DI (Hilt)
    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)

    // Network
    implementation(libs.retrofit)
    implementation(libs.okhttp)

    // Testing
    testImplementation(libs.junit)
    testImplementation(libs.mockk)
    androidTestImplementation(libs.compose.ui.test)
}
```

---

## 10. Anti-Patterns and Design Pitfalls

### 10.1 Anti-Pattern 1: Heavy Processing on the Main Thread

**Problem**: Performing network communication, disk I/O, or heavy computation on the UI thread (main thread) causes the UI to freeze (Application Not Responding = ANR).

```
Anti-Pattern: Main Thread Blocking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Timeline ────────────────────────────────────►

  Main Thread:
  ┌─────┐  ┌───────────────────────────┐  ┌─────┐
  │ UI  │  │  Network request (3 sec)  │  │ UI  │
  │Draw │  │  <- UI frozen during ->   │  │Draw │
  └─────┘  └───────────────────────────┘  └─────┘
  16ms      <- ANR triggered (Android: 5s / iOS: watchdog) ->

  Correct Pattern: Delegate to background thread
  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
  │ UI  │  │ UI  │  │ UI  │  │ UI  │  │ UI  │
  │Draw │  │Draw │  │Draw │  │Draw │  │Updt │
  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
  Main thread always responsive

  Background:
  ┌───────────────────────────┐
  │  Network request (3 sec)  │ -> Notify main thread on completion
  └───────────────────────────┘
```

**Bad Example (Android / Kotlin)**:
```kotlin
// NG: Network communication on main thread
// Android throws NetworkOnMainThreadException
fun loadDataBad() {
    val url = URL("https://api.example.com/data")
    val data = url.readText()  // Blocks main thread
    textView.text = data
}
```

**Good Example (Android / Kotlin)**:
```kotlin
// OK: Background processing with coroutines
fun loadDataGood() {
    viewModelScope.launch {
        val data = withContext(Dispatchers.IO) {
            // Network communication on IO dispatcher
            repository.fetchData()
        }
        // UI update on main thread (automatically Dispatchers.Main)
        _uiState.value = UiState.Success(data)
    }
}
```

### 10.2 Anti-Pattern 2: Ignoring Memory Leaks

**Problem**: When long-lived objects hold references to Activity or Context, GC cannot reclaim them, causing memory leaks. Since Activity is recreated on every screen rotation, memory is exhausted in a short time.

```
Memory Leak Patterns and Countermeasures
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Typical memory leak:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  static / companion object                      │
  │  ┌───────────────┐                              │
  │  │ Listener/     │── strong ref ──► Activity A  │
  │  │ Callback      │             (should be       │
  │  │ (long-lived)  │              destroyed on     │
  │  └───────────────┘              rotation)        │
  │                                                 │
  │  Activity A cannot be GC'd -> Memory leak       │
  │  New Activity created each rotation              │
  │  Old Activities accumulate                       │
  └─────────────────────────────────────────────────┘

  Countermeasures:

  1. Use WeakReference
     val activityRef = WeakReference(activity)

  2. Unregister listeners in sync with lifecycle
     override fun onDestroy() {
         listener.unregister()
         super.onDestroy()
     }

  3. Use ViewModel + LiveData / StateFlow
     ViewModel survives configuration changes (rotation),
     eliminating need for direct Activity references

  4. Detect with LeakCanary
     debugImplementation("com.squareup.leakcanary:leakcanary-android:2.x")
     -> Automatically detects and reports memory leaks at build time
```

### 10.3 Anti-Pattern 3: Excessive Background Processing

**Problem**: Overusing unnecessary background services and alarms rapidly drains battery. Since Android 8.0, background service launches are restricted, and violations result in system-forced termination.

**Countermeasure**: Use WorkManager to let the OS execute at optimal timing. For non-urgent tasks, specify `Constraints` to limit execution to charging and WiFi-connected states.

---

## 11. Tiered Exercises

### Exercise 1: Beginner -- System Information Collection and Display

**Goal**: Create an app for iOS or Android that collects and displays device system information.

**Requirements**:
1. Retrieve OS version, device model, memory amount, and CPU core count
2. Display battery level and charging status in real-time
3. Display storage usage and free space as a pie chart

**Verification Items**:
- [ ] Successfully retrieved information from sysctl (iOS) or /proc (Android)
- [ ] Battery state changes reflected in real-time
- [ ] Memory usage changes monitored successfully

**Hints**:
- iOS: Use `ProcessInfo`, `UIDevice`, `FileManager`
- Android: Use `Build`, `ActivityManager`, `BatteryManager`, `StatFs`

---

### Exercise 2: Intermediate -- Background Tasks and Notifications

**Goal**: Create an app that periodically fetches data in the background and sends local notifications based on conditions.

**Requirements**:
1. Execute background tasks at 15-minute intervals
2. Fetch weather API data and save to local DB
3. Notify the user with a local notification when rain is forecast
4. Design considering Doze mode / Background App Refresh

**Verification Items**:
- [ ] Background task correctly scheduled
- [ ] Task executed after Doze mode recovery (Android)
- [ ] Verified behavior when Background App Refresh is disabled (iOS)
- [ ] Notification permission request properly implemented
- [ ] Confirmed battery consumption is not excessive via profiler

**Hints**:
- iOS: Use `BGTaskScheduler`, `UNUserNotificationCenter`
- Android: Use `WorkManager` + `NotificationCompat`

---

### Exercise 3: Advanced -- Secure Data Storage and IPC

**Goal**: Implement secure storage protected by biometric authentication and share data with another app/Extension.

**Requirements**:
1. Implement biometric authentication (Face ID / fingerprint) unlock
2. Store sensitive data in Keychain (iOS) / EncryptedSharedPreferences (Android)
3. Share data with Widget / App Extension (App Groups / ContentProvider)
4. Implement app tampering detection (Jailbreak/Root detection)
5. Protect network communication with SSL Pinning

**Verification Items**:
- [ ] Fallback (passcode input) works on biometric authentication failure
- [ ] Data correctly stored in Keychain / EncryptedSharedPreferences
- [ ] Data readable from Widget / Extension
- [ ] Detection message displayed on Root / Jailbreak environment
- [ ] Confirmed SSL cannot be intercepted by MITM tools like Charles Proxy

**Hints**:
- iOS: `LAContext` (LocalAuthentication), `Keychain Services`, `App Groups`
- Android: `BiometricPrompt`, `EncryptedSharedPreferences`, `ContentProvider`

---

## 12. FAQ -- Frequently Asked Questions

### Q1: Why doesn't iOS use swap?

**A**: There are three main reasons iOS does not use swap (paging out to disk).

1. **NAND Flash Lifespan**: Massive writes from swapping rapidly consume NAND flash write cycles. Mobile device storage is non-replaceable, and reduced lifespan directly shortens device usable life.

2. **Latency**: NAND flash random read/write is 100-1000x slower than DRAM. Performance degradation from swap is incompatible with mobile responsiveness requirements.

3. **Power Consumption**: Storage I/O consumes more power than CPU idle state, making it unsuitable for battery-powered mobile devices.

As alternatives, iOS combines memory compression (WKdm/LZ4 algorithms) with process termination via jetsam. Compression effectively expands physical memory capacity by approximately 1.5-2x, and when that is still insufficient, it terminates lower-priority processes.

### Q2: How severe is Android's fragmentation problem?

**A**: Android fragmentation refers to the extreme diversity of OS versions, screen sizes, and hardware configurations among devices on the market.

**Key Challenges and Countermeasures:**

- **OS Version Fragmentation**: As of 2025, Android 10-15 each hold 5-20% market share. Google provides backward compatibility through Jetpack libraries, making new features available on older OS versions.

- **Screen Size Diversity**: Support needed from 4-inch smartphones to 13-inch tablets and foldable devices. Address with Jetpack Compose's `WindowSizeClass` and Material Design's Adaptive Layout.

- **Security Patch Delays**: Project Treble (Android 8.0) and Project Mainline (Android 10) allow Google to update security modules directly through the Play Store, partially resolving OEM patch delay issues.

- **Hardware Diversity**: Thousands of device types exist, but CTS (Compatibility Test Suite) ensures minimum compatibility. Automated testing on device farms like Firebase Test Lab is recommended.

### Q3: What are best practices for runtime permission design in mobile apps?

**A**: The timing and method of permission requests significantly impact user experience. Follow these best practices:

1. **Request only needed permissions when needed**: Do not request all permissions at once on app launch. Request camera permission when opening a camera feature, location permission when displaying a map.

2. **Show pre-explanation**: Before displaying the OS permission dialog, show an in-app screen explaining why the permission is needed. This is especially important on iOS where re-requesting after denial is not possible.

3. **Design to work without permissions**: Provide fallbacks such as manual address input when location is denied, or gallery selection when camera is denied.

4. **Operate with minimum precision**: Use approximate location when precise location is unnecessary. Continuous background location tracking should be limited to very specific use cases as it erodes user trust.

### Q4: How can app size be reduced for iOS and Android?

**A**: App size reduction is important as it directly impacts download rates and first-launch rates.

| Method | iOS | Android |
|--------|-----|---------|
| Code shrinking | Swift Compiler optimization (-Osize) | R8 / ProGuard (isMinifyEnabled) |
| Resource reduction | Asset Catalog (auto 1x/2x/3x selection) | isShrinkResources + WebP conversion |
| App splitting | App Thinning (Slicing + ODR) | App Bundle (.aab) + Dynamic Delivery |
| Dynamic delivery | On-Demand Resources | Dynamic Feature Modules |
| Image format | HEIF, WebP | WebP, AVIF |
| Native libraries | arm64 only (no Universal Binary) | ABI Split (arm64-v8a only) |

### Q5: Should I choose Kotlin Multiplatform (KMP) or Flutter?

**A**: Decide based on the following criteria:

- **KMP is suitable when**: You want to share business logic (data processing, API communication, local DB) while implementing UI natively for each platform. Suitable for teams with existing Kotlin expertise.

- **Flutter is suitable when**: You want fully unified UI across platforms. When high-performance custom UI is needed and you also want to expand to web and desktop. Suitable for teams that can accept the Dart learning cost.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important. Understanding deepens not just from theory, but from actually writing and running code.

### Q2: What common mistakes do beginners make?

Skipping fundamentals to jump into applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary and Next Steps

### 13.1 Comprehensive Comparison Table

| Item | iOS | Android |
|------|-----|---------|
| Kernel | XNU (Mach + BSD hybrid) | Linux (monolithic + modular) |
| IPC | Mach Port, XPC | Binder |
| Memory Management | Jetsam + memory compression | LMK + zRAM |
| Swap | None | zRAM (in-memory compression) |
| GC / Memory Release | ARC (compile-time reference counting) | Tracing GC (ART Concurrent Copying) |
| File System | APFS | ext4 / f2fs |
| Security | Sandbox + KTRR + SEP | SELinux + seccomp + TrustZone |
| Power Management | Background App Refresh, App Nap | Doze, App Standby Buckets |
| Push Notifications | APNs | FCM |
| Drivers | IOKit (C++) | HAL/HIDL/AIDL + Linux Driver |
| Update Method | All devices simultaneously (Apple managed) | Project Mainline + OEM dependent |
| Dev Language | Swift / Objective-C | Kotlin / Java |
| Declarative UI | SwiftUI | Jetpack Compose |
| App Distribution | App Store (with review) | Play Store + sideloading |
| Source Code | Closed (except Darwin) | AOSP (open source) |

### 13.2 Future Trends

1. **On-Device AI Integration**: Apple Intelligence and Google Gemini Nano are being integrated at the OS level, with text generation, image recognition, and speech understanding heading toward local completion. NPU (Neural Processing Unit) performance is rapidly improving, and cloud-independent AI experiences are becoming standard.

2. **Deepening Privacy Enhancement**: Following App Tracking Transparency (iOS) and Privacy Sandbox (Android), technologies are evolving to achieve personalization while preserving privacy, including advertising ID deprecation, IP address obfuscation, and on-device machine learning-based personalization.

3. **Spatial Computing**: Apple Vision Pro (visionOS) and Android XR are extending mobile OS concepts from 2D screens to 3D space. ARKit / ARCore technologies are evolving for head-mounted displays, introducing new input paradigms (eye tracking, hand gestures).

4. **Satellite Communication Integration**: With satellite SOS since iOS 14 and Android 15's satellite messaging support, basic communication becomes available outside cellular coverage. The OS-level communication stack design is being expanded accordingly.

5. **Security Evolution**: With expanding hardware support for MTE (Memory Tagging Extension) and PAC (Pointer Authentication Code) proliferation, memory safety is being strengthened at the OS level. Zero-day attack difficulty continues to increase year over year.

### 13.3 Learning Roadmap

```
Mobile OS Learning Roadmap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Beginner ─────────────────────────────────
  │ OS fundamentals (processes, memory, FS)
  │ Swift / Kotlin syntax
  │ Create app with official tutorials
  ▼
  Intermediate ─────────────────────────────
  │ Lifecycle management
  │ Background processing and power management
  │ Understanding security models
  │ Performance profiling
  ▼
  Advanced ─────────────────────────────────
  │ Kernel internals understanding
  │ IPC design and implementation
  │ Driver / HAL mechanisms
  │ Security research (vulnerability analysis)
  ▼
  Expert ────────────────────────────────
    OS customization (AOSP builds)
    Kernel module development
    Reverse engineering
    Security auditing
```

---

## Next Guides to Read


---

## 14. References

### Books

1. Levin, J. *"\*OS Internals, Volume I: User Mode"*. Technologeeks Press, 2017. -- The definitive systematic guide to iOS/macOS user-space architecture. Details the internal structure of frameworks built on the XNU kernel.

2. Levin, J. *"\*OS Internals, Volume II: Kernel Mode"*. Technologeeks Press, 2019. -- Advanced reference that deeply explores XNU kernel internals (Mach, BSD, IOKit). Includes implementation details of jetsam, sandbox, and code signing.

3. Levin, J. *"\*OS Internals, Volume III: Security & Insecurity"*. Technologeeks Press, 2020. -- Comprehensive coverage of iOS/macOS security architecture. Includes details on Secure Enclave, code signing chain, AMFI, and sandbox profiles.

4. Yaghmour, K. *"Embedded Android: Porting, Extending, and Customizing"*. O'Reilly Media, 2013. -- Practical guide to AOSP build system, HAL implementation, and kernel customization.

5. Gargenta, M. and Nakamura, M. *"Learning Android"*. O'Reilly Media, 2014. -- Systematic introduction from Android app development basics to using system services.

### Official Documentation

6. Apple Inc. *"Apple Platform Security Guide"*. 2024. https://support.apple.com/guide/security/ -- Official Apple documentation explaining iOS, iPadOS, and macOS security design. Covers hardware security, encryption, and authentication mechanisms.

7. Android Open Source Project. *"Android Architecture"*. https://source.android.com/docs/core/architecture -- Official AOSP architecture documentation. Includes design and implementation guidelines for HAL, Treble, and AIDL.

8. Google. *"Android Developer Documentation"*. https://developer.android.com/docs -- Official Android app development reference. Includes API guides, best practices, and codelabs.

9. Apple Inc. *"iOS App Dev Tutorials"*. https://developer.apple.com/tutorials/app-dev-training -- Official iOS app development tutorials using SwiftUI.

### Academic Papers and Technical Documents

10. Felt, A. P., et al. "Android Permissions Demystified". *ACM CCS*, 2011. -- Pioneering paper analyzing the design and actual usage of the Android permission model.

11. Enck, W., et al. "TaintDroid: An Information-Flow Tracking System for Realtime Privacy Monitoring on Smartphones". *OSDI*, 2010. -- Proposal for a system to track privacy information leakage in mobile apps.

12. Singh, A. *"Mac OS X Internals: A Systems Approach"*. Addison-Wesley, 2006. -- Classic reference for understanding the historical background and design principles of the XNU kernel. Details the BSD/Mach integration history of Darwin.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://ja.wikipedia.org/) - Technology concept overviews
