import Foundation

/// A new data folder for a job the app starts itself (a call it records, Runner.startCall), named as mt.py names its
/// own (`_new_dir` in mt.py): `<yyyyMMdd-HHmmss>-<name>`, or `<yyyyMMdd-HHmmss>` without a name. When another job of that
/// name began in the same second (an mt.py transcription of a file named like the call, started from the queue the
/// moment the call began), the call takes `-2`, `-3` … of that second (`-recording-2` without a name) instead of writing
/// into that job's folder. Each folder is made without intermediate folders, which fails on one that exists, so of two
/// jobs racing for one name only one gets it; mt.py makes its own the same way. Recordings.parse reads all these names
/// back: the date from the stamp, the rest ("call-2", "recording-2") as the name.
enum DataFolder {
    enum Failure: Error, LocalizedError {
        case allTaken(String)
        var errorDescription: String? {
            switch self {
            case .allTaken(let first): return "\(first): 999 data folders of that name in one second"
            }
        }
    }

    /// The time stamp at the front of a data folder's name: Gregorian calendar and ASCII digits whatever the Mac's region
    /// (a Buddhist-calendar year or Arabic-Indic digits would not read back, Recordings.parse), local time as mt.py's.
    static func stamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = Calendar(identifier: .gregorian)
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f.string(from: date)
    }

    /// The name of the `number`-th data folder of that second under that name (1 is the plain one).
    static func name(stamp: String, name: String, number: Int = 1) -> String {
        let name = name.trimmingCharacters(in: .whitespaces)
        if number <= 1 { return name.isEmpty ? stamp : "\(stamp)-\(name)" }
        return "\(stamp)-\(name.isEmpty ? "recording" : name)-\(number)"
    }

    /// Makes a new data folder in `data` (made first if missing) and returns it. Throws when `data` cannot be made, when
    /// a folder cannot be made for another reason than one of that name being there already, or when 999 are taken.
    static func make(in data: URL, name: String, date: Date = Date(), fileManager: FileManager = .default) throws -> URL {
        try fileManager.createDirectory(at: data, withIntermediateDirectories: true)
        let stamp = DataFolder.stamp(date)
        for number in 1...999 {
            let folder = data.appendingPathComponent(DataFolder.name(stamp: stamp, name: name, number: number), isDirectory: true)
            do {
                try fileManager.createDirectory(at: folder, withIntermediateDirectories: false)
                return folder
            } catch let error as CocoaError where error.code == .fileWriteFileExists {
                continue                              // mkdir's EEXIST: something of that name is there, as mt.py decides
            } catch {
                // Taken all the same if something of that name is there (an older Foundation reporting EEXIST otherwise);
                // any other failure is the answer.
                if fileManager.fileExists(atPath: folder.path) { continue }
                throw error
            }
        }
        throw Failure.allTaken(data.appendingPathComponent(DataFolder.name(stamp: stamp, name: name)).path)
    }
}
