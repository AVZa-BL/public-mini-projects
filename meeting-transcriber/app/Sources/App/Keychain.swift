import Foundation
import Security

/// The Claude API key (used only for translation). It goes to the macOS Keychain and is read back to be sure it
/// is there; if the Keychain refuses (an ad-hoc signed app can be refused an item that an earlier build created),
/// it goes to a private file in Application Support instead, and the window says so.
enum KeyStore {
    static let service = "com.github.avza-bl.meeting-transcriber"
    static let account = "ANTHROPIC_API_KEY"
    static let hfAccount = "HF_TOKEN"            // the Hugging Face token for the optional pyannote models

    enum Place { case keychain, file }

    enum Outcome {
        case saved(Place, note: String?)
        case failed(String)
    }

    static func file(_ account: String = account) -> KeyFile {
        let support = Engine(home: FileManager.default.homeDirectoryForCurrentUser).support
        return KeyFile(url: support.appendingPathComponent(account == hfAccount ? "hf-token" : "api-key"))
    }

    private static func base(_ account: String) -> [String: Any] {
        return [kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: service,
                kSecAttrAccount as String: account]
    }

    private static func keychainLoad(_ account: String) -> String? {
        var query = base(account)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Updates the item if there is one, else adds it (deleting first would fail silently on an item another build owns).
    private static func keychainWrite(_ key: String, _ account: String) -> OSStatus {
        let data = Data(key.utf8)
        var status = SecItemUpdate(base(account) as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var query = base(account)
            query[kSecValueData as String] = data
            status = SecItemAdd(query as CFDictionary, nil)
        }
        return status
    }

    private static func describe(_ status: OSStatus) -> String {
        let text = SecCopyErrorMessageString(status, nil) as String? ?? "error"
        return "\(text), code \(status)"
    }

    /// The saved key and where it is. The file wins: it exists only when the Keychain refused the latest save.
    static func load(_ account: String = account) -> (key: String, place: Place)? {
        if let key = file(account).read() { return (key, .file) }
        if let key = keychainLoad(account), !key.isEmpty { return (key, .keychain) }
        return nil
    }

    static func save(_ key: String, _ account: String = account) -> Outcome {
        let status = keychainWrite(key, account)
        let file = self.file(account)
        if status == errSecSuccess && keychainLoad(account) == key {
            file.delete()
            return .saved(.keychain, note: nil)
        }
        let why = status == errSecSuccess ? "it did not return the key it was given" : describe(status)
        do {
            try file.write(key)
            return .saved(.file, note: "The Keychain refused the key (\(why)), so it was saved in a file that only you can read: \(file.url.path)")
        } catch {
            return .failed("The key could not be saved. The Keychain refused it (\(why)) and the file \(file.url.path) could not be written (\(error.localizedDescription)).")
        }
    }

    static func delete(_ account: String = account) {
        SecItemDelete(base(account) as CFDictionary)
        file(account).delete()
    }
}
