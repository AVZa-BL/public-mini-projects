import Foundation
import Security

/// The app's own code signature, read through the Security framework.
enum CodeSignature {
    /// The designated requirement of the running app as text (what `codesign -d -r-` prints), or nil when it cannot be read.
    static func designatedRequirement() -> String? {
        var code: SecCode?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let running = code else { return nil }
        var staticCode: SecStaticCode?
        guard SecCodeCopyStaticCode(running, [], &staticCode) == errSecSuccess, let onDisk = staticCode else { return nil }
        var requirement: SecRequirement?
        guard SecCodeCopyDesignatedRequirement(onDisk, [], &requirement) == errSecSuccess, let found = requirement else { return nil }
        var text: CFString?
        guard SecRequirementCopyString(found, [], &text) == errSecSuccess, let string = text else { return nil }
        return string as String
    }
}
