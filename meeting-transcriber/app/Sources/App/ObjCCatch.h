#import <Foundation/Foundation.h>
// proc_pid_rusage for the telemetry's sampler (SystemSampler.swift): declared in libproc.h, which this bridging header
// makes visible to Swift whether or not the SDK's Darwin module exports it.
#include <libproc.h>

NS_ASSUME_NONNULL_BEGIN

/// Runs `block` and returns the Objective-C exception it raised, or nil when it returned normally. Swift cannot catch
/// such an exception, and AVAudioEngine raises them (not errors) when a tap or a start finds the engine in a state it
/// does not accept; uncaught, one of them ends the app with SIGABRT (the crashes of 1.9.6 and 1.9.7). Compiled by
/// build_app.sh with clang and handed to swiftc as the bridging header plus an object file.
NSException * _Nullable MTCatchObjCException(void (NS_NOESCAPE ^block)(void));

NS_ASSUME_NONNULL_END
