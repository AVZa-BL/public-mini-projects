import Foundation

/// Last-resort cleanup: terminates a process and all of its descendants
/// (python, ffmpeg). Normally mt.py
/// stops on SIGTERM by itself; this only runs if it hangs.
enum ProcessTree {
    static func descendants(of root: Int32) -> [Int32] {
        let ps = Process()
        ps.executableURL = URL(fileURLWithPath: "/bin/ps")
        ps.arguments = ["-axo", "pid=,ppid="]
        let output = Pipe()
        ps.standardOutput = output
        ps.standardError = FileHandle.nullDevice
        do { try ps.run() } catch { return [] }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        ps.waitUntilExit()
        var children: [Int32: [Int32]] = [:]
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count >= 2, let pid = Int32(parts[0]), let parent = Int32(parts[1]) else { continue }
            children[parent, default: []].append(pid)
        }
        var result: [Int32] = []
        var pending = children[root] ?? []
        while let pid = pending.popLast() {
            result.append(pid)
            pending.append(contentsOf: children[pid] ?? [])
        }
        return result
    }

    static func terminate(rootPID: Int32) {
        let all = descendants(of: rootPID)
        _ = kill(rootPID, SIGTERM)
        for pid in all.reversed() { _ = kill(pid, SIGTERM) }
    }
}
