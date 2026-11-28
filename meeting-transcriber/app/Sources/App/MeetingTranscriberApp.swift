import SwiftUI
import AppKit

@main
struct MeetingTranscriberApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var runner = Runner.shared
    @StateObject private var layout = LayoutPrefs.shared        // the View menu's two columns

    init() {
        // Command-line entry points for the build's smoke test.
        let arguments = CommandLine.arguments
        if arguments.contains("--version") {
            print(AppInfo.versionString(info: Bundle.main.infoDictionary))
            exit(0)
        }
        if arguments.contains("--self-test") {
            var ok = Runner.bundleIsComplete()
            print(ok ? "Engine files: bundled." : "FAIL: engine files missing from the bundle")
            if let runtime = Runner.bundledRuntime() {
                let p = Process()
                p.executableURL = runtime.appendingPathComponent("bin/python3")
                p.arguments = ["--version"]
                let pipe = Pipe()
                p.standardOutput = pipe
                p.standardError = pipe
                if (try? p.run()) != nil {
                    let out = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                    p.waitUntilExit()
                    let good = p.terminationStatus == 0 && out.hasPrefix("Python 3.")
                    print(good ? "Bundled runtime (\(Runner.machine)): \(out.trimmingCharacters(in: .whitespacesAndNewlines))"
                               : "FAIL: the bundled runtime does not run: \(out)")
                    ok = ok && good
                } else {
                    print("FAIL: the bundled runtime could not be started")
                    ok = false
                }
            } else {
                print("No bundled runtime for \(Runner.machine): a system Python 3.9+ will be used.")
            }
            print(ok ? "Self-test passed." : "Self-test FAILED.")
            exit(ok ? 0 : 1)
        }
    }

    var body: some Scene {
        WindowGroup("Meeting Transcriber") {
            ContentView()
                .environmentObject(runner)
        }
        .defaultSize(width: 1280, height: 860)               // three columns: Controls, Session, Telemetry (min 980 × 660)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            // View ▸ Show Telemetry (⌥⌘T) and Show Settings (⌥⌘S). They take the place of the toolbar's own items, whose
            // "Hide Toolbar" has ⌥⌘T too: two menu items with one shortcut would leave it to AppKit which one runs.
            CommandGroup(replacing: .toolbar) {
                Toggle("Show Telemetry", isOn: Binding(get: { layout.showTelemetry }, set: { layout.showTelemetry = $0 }))
                    .keyboardShortcut("t", modifiers: [.option, .command])
                Toggle("Show Settings", isOn: Binding(get: { layout.showControls }, set: { layout.showControls = $0 }))
                    .keyboardShortcut("s", modifiers: [.option, .command])
            }
            CommandGroup(after: .appInfo) {
                Button("Project Page…") {
                    if let url = URL(string: AppInfo.projectURL) { NSWorkspace.shared.open(url) }
                }
                Button("Set Up Speech Engine Again…") { runner.setUpEngine() }
                    .disabled(runner.isRunning)                       // the setup rebuilds the engine every running process uses
                    .help("Not while a recording or a job runs: the setup rebuilds the engine they run on.")
            }
        }
    }
}

/// An Objective-C exception nobody caught ends the app, and macOS's crash report leaves the exception's reason out
/// (both reports of 1.9.6 and 1.9.7 did). This handler runs just before the end and writes the name, the reason and
/// the stack to a file, so the next report can be read: ~/Library/Logs/Meeting Transcriber/uncaught-exception.txt.
enum UncaughtExceptionLog {
    static var file: URL {
        let logs = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Meeting Transcriber", isDirectory: true)
        return logs.appendingPathComponent("uncaught-exception.txt")
    }

    static func install() {
        NSSetUncaughtExceptionHandler { exception in UncaughtExceptionLog.write(exception) }
    }

    static func write(_ exception: NSException) {
        let text = "\(Date())\nMeeting Transcriber \(Runner.appVersion)\n\(exception.name.rawValue)\n\(exception.reason ?? "no reason given")\n\n"
            + exception.callStackSymbols.joined(separator: "\n") + "\n"
        let url = file
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        UncaughtExceptionLog.install()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard Runner.shared.isRunning else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "Work is still in progress."
        alert.informativeText = "Quitting cancels the recording, the running job and the jobs waiting behind it. Files that are already written are kept; a recording keeps its audio in the data folder, and a transcription continues from its checkpoint when it is run again."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Keep Working")
        alert.addButton(withTitle: "Cancel and Quit")
        if alert.runModal() == .alertFirstButtonReturn { return .terminateCancel }
        Runner.shared.cancel()
        // A recording needs a moment to close its files: wait until the job has ended (at most 5 s) before quitting.
        let deadline = Date().addingTimeInterval(5)
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
            if !Runner.shared.isRunning || Date() > deadline {
                timer.invalidate()
                NSApp.reply(toApplicationShouldTerminate: true)
            }
        }
        return .terminateLater
    }
}
