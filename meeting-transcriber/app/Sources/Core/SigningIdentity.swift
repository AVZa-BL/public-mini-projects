import Foundation

/// What the app's code signature tells macOS about its identity. macOS ties a permission grant (microphone, Screen &
/// System Audio Recording, System Audio Recording Only) to the signed code's "designated requirement": an ad-hoc signature's
/// is the build's own hash, so every rebuild or update is a new identity and the old grant no longer counts; a signature
/// with a certificate names the certificate instead, and the grant survives (app/make_signing_identity.sh makes one).
enum SigningIdentity {
    enum Kind: Equatable {
        case stable        // identifier "…" and certificate leaf = H"…": the same for every build
        case adHoc         // cdhash H"…": a new identity at every build
        case unknown       // no signature could be read
    }

    /// The kind of identity a designated requirement describes (the text `codesign -d -r-` prints after "designated =>").
    static func kind(ofRequirement requirement: String?) -> Kind {
        guard let text = requirement?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else { return .unknown }
        if text.contains("certificate leaf") || text.contains("anchor apple") || text.contains("certificate 1[") { return .stable }
        if text.contains("cdhash") { return .adHoc }
        return .unknown
    }

    /// One line for the window when the build will not keep its permissions.
    static func warning(for kind: Kind) -> String? {
        guard kind == .adHoc else { return nil }
        return "This build is ad-hoc signed: macOS forgets its permissions (microphone, the computer's sound) at every update or rebuild. Build it with the project certificate to keep them (README: Permissions and updates; app/make_signing_identity.sh once)."
    }
}
