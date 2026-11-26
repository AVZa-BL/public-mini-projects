import Foundation

/// The speech engine: mt.py and its Python packages. The app bundles mt.py; the packages (faster-whisper …) are
/// installed once into a private virtual environment in Application Support, so the app itself stays small.
struct Engine: Equatable {
    /// ~/Library/Application Support/Meeting Transcriber
    let support: URL

    init(home: URL) {
        support = home.appendingPathComponent("Library/Application Support/Meeting Transcriber", isDirectory: true)
    }

    var venv: URL { return support.appendingPathComponent("venv", isDirectory: true) }
    var python: URL { return venv.appendingPathComponent("bin/python") }
    /// The app's own Python runtime, copied out of the bundle at setup (a stable path: a quarantined app may be moved).
    var runtime: URL { return support.appendingPathComponent("runtime", isDirectory: true) }
    var runtimePython: URL { return runtime.appendingPathComponent("bin/python3") }
    /// Transcripts and recordings (MT_HOME).
    var data: URL { return support.appendingPathComponent("data", isDirectory: true) }
    /// Written after a successful setup; holds the app version that did it.
    var marker: URL { return support.appendingPathComponent("engine-ready") }

    /// Pythons that can install faster-whisper (3.9+), tried when the bundle carries no runtime for this Mac.
    static let pythonCandidates = ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]

    /// The runtime the build bundles for this architecture: Contents/Resources/python/<arch>/bin/python3, or nil.
    static func bundledRuntime(resources: URL?, arch: String, fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) -> URL? {
        guard let resources = resources else { return nil }
        let folder = resources.appendingPathComponent("python/\(arch)", isDirectory: true)
        return fileExists(folder.appendingPathComponent("bin/python3").path) ? folder : nil
    }

    /// "Python 3.12.4" → [3, 12, 4]
    static func pythonVersion(_ output: String) -> [Int]? {
        let words = output.split(whereSeparator: { $0 == " " || $0 == "\n" })
        guard words.count >= 2, words[0] == "Python" else { return nil }
        let parts = words[1].split(separator: ".").compactMap { Int($0.prefix { $0.isNumber }) }
        return parts.count >= 2 ? parts : nil
    }

    static func isSupported(_ version: [Int]) -> Bool {
        return version.count >= 2 && (version[0] > 3 || (version[0] == 3 && version[1] >= 9))
    }

    /// The commands of the one-time setup, in order. Each is an executable followed by its arguments. With a bundled
    /// runtime it is copied into Application Support first and the environment is built from that copy.
    func setupSteps(basePython: String?, engineFolder: String, bundledRuntime: URL? = nil, mlx: Bool = false, pyannote: Bool = false) -> [[String]] {
        var steps: [[String]] = []
        let base: String
        if let bundled = bundledRuntime {
            steps.append(["/usr/bin/ditto", bundled.path, runtime.path])
            base = runtimePython.path
        } else {
            base = basePython ?? "/usr/bin/python3"
        }
        steps += [
            [base, "-m", "venv", venv.path],
            [python.path, "-m", "pip", "install", "--upgrade", "pip"],
            [python.path, "-m", "pip", "install", "-r", engineFolder + "/requirements.txt"],
        ]
        if mlx { steps.append([python.path, "-m", "pip", "install", "-r", engineFolder + "/requirements-mlx.txt"]) }
        if pyannote { steps.append([python.path, "-m", "pip", "install", "-r", engineFolder + "/requirements-pyannote.txt"]) }
        steps.append([python.path, engineFolder + "/mt.py", "prepare"])
        return steps
    }

    /// The Apple-GPU engine's packages may not install (an Intel Mac, an older macOS), nor pyannote's (2 GB, Python 3.10+):
    /// those steps are allowed to fail.
    static func isOptionalStep(_ command: [String]) -> Bool {
        return command.contains { $0.hasSuffix("requirements-mlx.txt") || $0.hasSuffix("requirements-pyannote.txt") }
    }

    /// Whether the setup was completed by this version of the app.
    func isReady(appVersion: String, fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) },
                 readMarker: () -> String? = { nil }) -> Bool {
        guard fileExists(python.path), fileExists(marker.path) else { return false }
        return readMarker()?.trimmingCharacters(in: .whitespacesAndNewlines) == appVersion
    }

    /// Whether ffmpeg and ffprobe (which decode every audio and video format) can be found.
    static func ffmpegAvailable(fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) -> Bool {
        let folders = extraPath.split(separator: ":").map(String.init)
        return ["ffmpeg", "ffprobe"].allSatisfy { tool in folders.contains { fileExists($0 + "/" + tool) } }
    }

    /// Folders where Homebrew puts ffmpeg; a Finder-started app does not have them in PATH.
    static let extraPath = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
}

enum AppInfo {
    static let projectURL = "https://github.com/AVZa-BL/public-mini-projects/tree/main/meeting-transcriber"

    static func versionString(info: [String: Any]?) -> String {
        let version = (info?["CFBundleShortVersionString"] as? String) ?? "0.0.0"
        let build = (info?["CFBundleVersion"] as? String) ?? ""
        let commit = (info?["MeetingTranscriberGitCommit"] as? String) ?? ""
        var text = "Meeting Transcriber \(version)"
        if !build.isEmpty && build != version { text += " (build \(build)" + (commit.isEmpty ? ")" : ", commit \(commit))") }
        else if !commit.isEmpty { text += " (commit \(commit))" }
        return text
    }
}
