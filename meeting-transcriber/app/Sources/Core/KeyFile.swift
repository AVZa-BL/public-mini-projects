import Foundation

/// The API key in a file only the user can read (mode 0600). The app uses it when the Keychain refuses the key.
struct KeyFile {
    let url: URL

    func read() -> String? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let key = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }

    func write(_ key: String) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? fm.removeItem(at: url)          // a new file is created with mode 0600 from the start, never wider
        guard fm.createFile(atPath: url.path, contents: Data(key.utf8), attributes: [.posixPermissions: 0o600]) else {
            throw CocoaError(.fileWriteUnknown)
        }
    }

    func delete() {
        try? FileManager.default.removeItem(at: url)
    }
}
