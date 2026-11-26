import Foundation

/// Turns what the user picked or dropped (files and folders) into the list of files to work on.
enum MediaFiles {
    /// Audio and video containers ffmpeg reads. Used only to pick files out of a folder: a file the user
    /// chooses or drops by itself is always taken, whatever its extension.
    static let mediaExtensions: Set<String> = [
        "mp3", "m4a", "m4b", "aac", "wav", "flac", "ogg", "oga", "opus", "wma", "aif", "aiff", "aifc", "caf", "amr",
        "ac3", "mka", "weba", "mp4", "m4v", "mov", "mkv", "webm", "avi", "flv", "wmv", "mpg", "mpeg", "ts", "mts",
        "m2ts", "3gp", "3g2", "ogv",
    ]
    static let textExtensions: Set<String> = ["txt", "text", "md"]

    /// More than this many files from folders are not added (a folder like the home folder would otherwise
    /// queue the whole disk); the caller says so.
    static let limit = 1000
    /// A scan gives up after looking at this many entries, so a huge tree or a slow volume cannot keep it busy for ever.
    static let visitLimit = 200_000

    struct Result: Equatable {
        var files: [String] = []
        var truncated = false
    }

    static func matches(_ path: String, translate: Bool) -> Bool {
        let ext = URL(fileURLWithPath: path).pathExtension.lowercased()
        return (translate ? textExtensions : mediaExtensions).contains(ext)
    }

    /// Files as they are; folders replaced by the matching files inside them, at any depth, hidden items and
    /// package contents skipped, sorted by path. `translate` picks text files instead of audio and video.
    /// The scan stops as soon as `limit` files are found or `visitLimit` entries were looked at, and says so
    /// in `truncated`; then the files returned are a sorted selection, not necessarily the first in path order.
    /// It reads the disk, so call it off the main thread.
    static func expand(_ paths: [String], translate: Bool, fileManager: FileManager = .default,
                       visitLimit: Int = MediaFiles.visitLimit) -> Result {
        var result = Result()
        for path in paths {
            var isDir: ObjCBool = false
            guard fileManager.fileExists(atPath: path, isDirectory: &isDir) else { continue }
            guard isDir.boolValue else {
                result.files.append(path)
                continue
            }
            let remaining = max(0, limit - result.files.count)
            let root = URL(fileURLWithPath: path, isDirectory: true)
            guard let walker = fileManager.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { continue }
            var found: [String] = []
            var visited = 0
            for case let url as URL in walker {
                visited += 1
                if visited > visitLimit { result.truncated = true; break }
                let regular = (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile ?? false
                if regular && matches(url.path, translate: translate) {
                    found.append(url.path)
                    if found.count > remaining { break }          // one more than fits: enough to know it is cut
                }
            }
            found.sort()
            if found.count > remaining { result.truncated = true }
            result.files.append(contentsOf: found.prefix(remaining))
        }
        return result
    }
}
