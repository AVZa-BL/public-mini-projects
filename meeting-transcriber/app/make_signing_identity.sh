#!/bin/bash
# Makes the one signing identity every build of the app should carry, so macOS keeps the app's permissions
# (microphone, Screen & System Audio Recording, System Audio Recording Only) across rebuilds and updates.
#
# Why: an ad-hoc signature (codesign --sign -) has no stable identity. macOS ties a permission grant to the signed code's
# "designated requirement", which for an ad-hoc build is the build's own hash, so after every rebuild or update the entry
# in System Settings still shows as allowed but no longer matches the app, and the permission does not count. With one
# self-signed code-signing certificate the requirement becomes `identifier "com.github.avza-bl.meeting-transcriber" and
# certificate leaf = H"…"`, the same for every build, and the grant survives. No Apple developer account is needed; the
# certificate is not trusted by anyone but this Mac (Gatekeeper's right-click → Open on a downloaded app stays as it is).
#
#   ./make_signing_identity.sh            # create the certificate, put it in the login keychain, print the CI secrets
#   ./make_signing_identity.sh --export   # print the CI secrets again for an identity that already exists
#
# After this, build_app.sh (and the launcher, which runs it) signs with the identity automatically. The first signed build
# asks for the permissions one last time (it is a new identity to macOS): allow it, and later builds keep the grant.
# To let the releases built by GitHub Actions keep the permissions too, store the two values printed at the end as the
# repository secrets MT_SIGNING_P12 and MT_SIGNING_P12_PASSWORD (optional; without them CI keeps signing ad-hoc).
# macOS only.
set -euo pipefail

NAME="Meeting Transcriber Signing"
# HOME is unset or empty when the script is started by something other than a login shell (an app, a launch agent, `env -i`,
# some `sudo` set-ups): take the account's home folder from the user database then, so the keychain goes to ~/Library and
# never to /Library, and `security` finds the account's keychain search list.
[ -n "${HOME:-}" ] || HOME=$(eval echo "~$(id -un)")
export HOME
# The identity lives in a keychain of its own, with nothing else in it: an export then holds this identity only, never
# another certificate or key of yours. The keychain has an empty password (its file is readable by you alone) and never
# locks, so codesign can use it at every build without a prompt; it is added to your keychain search list.
KEYCHAIN="$HOME/Library/Keychains/meeting-transcriber-signing.keychain-db"
EXPORT=0
case "${1:-}" in
  --export) EXPORT=1 ;;
  -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  '') ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
[ "$(uname -s)" = "Darwin" ] || { echo "ERROR: this script runs on macOS only." >&2; exit 1; }

have_identity() {
  [ -f "$KEYCHAIN" ] && security find-identity -v -p codesigning "$KEYCHAIN" 2>/dev/null | grep -q "\"$NAME\""
}

# Apple's own openssl (LibreSSL, always at /usr/bin) rather than whichever is first on PATH (Homebrew's OpenSSL 3, as a
# rule): its output is what macOS's `security` reads. The explicit PKCS#12 algorithms below make either work.
OPENSSL=/usr/bin/openssl
[ -x "$OPENSSL" ] || OPENSSL=openssl

random_password() {
  "$OPENSSL" rand -hex 16                  # a bounded source: no pipe to close early (set -o pipefail would end the script)
}

print_secrets() {
  # The certificate and its key, packed for the CI job: MT_SIGNING_P12 (base64 of a .p12) and MT_SIGNING_P12_PASSWORD.
  # Exported from the dedicated keychain, which holds this one identity and nothing else.
  local dir pass
  dir=$(mktemp -d)
  pass=$(random_password)
  security unlock-keychain -p "" "$KEYCHAIN" >/dev/null 2>&1 || true
  security export -k "$KEYCHAIN" -t identities -f pkcs12 -P "$pass" -o "$dir/identity.p12" >/dev/null 2>&1 \
    || { echo "ERROR: the identity could not be exported from $KEYCHAIN (macOS may ask you to allow it)." >&2; rm -rf "$dir"; exit 1; }
  echo
  echo "Optional, so that the releases built by GitHub Actions keep the permissions too: store these two repository secrets"
  echo "(Settings > Secrets and variables > Actions, or with the GitHub CLI):"
  echo
  echo "  gh secret set MT_SIGNING_P12_PASSWORD --repo AVZa-BL/mini-projs --body '$pass'"
  echo "  base64 < '$dir/identity.p12' | gh secret set MT_SIGNING_P12 --repo AVZa-BL/mini-projs"
  echo
  echo "The .p12 file stays at $dir/identity.p12 until you delete it (it holds the private key: keep it to yourself)."
}

if have_identity; then
  echo "The identity \"$NAME\" is already in $KEYCHAIN; build_app.sh uses it."
  [ "$EXPORT" = "1" ] && print_secrets
  exit 0
fi
[ "$EXPORT" = "0" ] || { echo "ERROR: no identity \"$NAME\" in $KEYCHAIN yet: run this script without --export first." >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PASS=$(random_password)
cat > "$WORK/req.cnf" <<EOF
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = $NAME
[v3]
basicConstraints = critical, CA:false
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
subjectKeyIdentifier = hash
EOF
echo "Creating the certificate \"$NAME\" (valid 10 years)…"
"$OPENSSL" req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$WORK/req.cnf" \
  -keyout "$WORK/key.pem" -out "$WORK/cert.pem" >/dev/null 2>&1
# The bundle is written in the PKCS#12 form macOS imports (3DES for the key and the certificate, SHA-1 for the MAC):
# OpenSSL 3's defaults (AES-256, SHA-256) make `security import` fail with "MAC verification failed during PKCS12 import
# (wrong password?)" on many macOS versions. The same flags work with Apple's LibreSSL and with Homebrew's OpenSSL 3.
"$OPENSSL" pkcs12 -export -inkey "$WORK/key.pem" -in "$WORK/cert.pem" -name "$NAME" \
  -certpbe PBE-SHA1-3DES -keypbe PBE-SHA1-3DES -macalg sha1 \
  -passout "pass:$PASS" -out "$WORK/identity.p12"
# No "…" right after a variable: macOS's bash 3.2 takes the first byte of a multibyte character following $NAME as part
# of the name ("KEYCHAIN?: unbound variable") in a UTF-8 locale.
echo "Putting it into its own keychain: ${KEYCHAIN}"
if [ ! -f "$KEYCHAIN" ]; then
  security create-keychain -p "" "$KEYCHAIN"
  security set-keychain-settings "$KEYCHAIN"                         # no auto-lock: codesign must not stop a build for a prompt
fi
security unlock-keychain -p "" "$KEYCHAIN"
security import "$WORK/identity.p12" -k "$KEYCHAIN" -P "$PASS" -T /usr/bin/codesign -T /usr/bin/security >/dev/null \
  || { echo "ERROR: macOS could not import the certificate bundle into $KEYCHAIN (the message above says why). Nothing was added." >&2; exit 1; }
# codesign only takes a certificate it can validate: trust this one for code signing on this Mac (user trust settings;
# macOS may ask for your login password once).
security add-trusted-cert -r trustRoot -p codeSign -k "$KEYCHAIN" "$WORK/cert.pem" 2>/dev/null \
  || echo "  (the trust setting was not added; if build_app.sh does not find the identity, open Keychain Access and set the certificate's Code Signing trust to Always Trust)"
# Let codesign use the key without a prompt at every build.
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" "$KEYCHAIN" >/dev/null 2>&1 || true
# Put the keychain on the search list, so codesign and build_app.sh find the identity (the others stay as they are).
if ! security list-keychains -d user | grep -q "meeting-transcriber-signing.keychain-db"; then
  # shellcheck disable=SC2046  # the existing entries are meant to be split into words
  security list-keychains -d user -s "$KEYCHAIN" $(security list-keychains -d user | tr -d '"')
fi
if have_identity; then
  echo "Done: \"$NAME\" is a valid code-signing identity on this Mac."
else
  echo "The certificate is in the keychain but codesign does not see it as valid yet: open Keychain Access, choose the"
  echo "keychain \"meeting-transcriber-signing\", find \"$NAME\", set Trust > Code Signing to Always Trust, then run this script again."
  exit 1
fi
echo "Its designated requirement for the app will be: identifier \"com.github.avza-bl.meeting-transcriber\" and certificate leaf = H\"<this certificate>\""
echo "Next: build the app (./build_app.sh --install, or the launcher's update). The first signed build asks for the"
echo "permissions one last time; later builds keep them."
echo
echo "Optional, so that the releases built by GitHub Actions keep the permissions too: store these two repository secrets"
echo "(Settings > Secrets and variables > Actions, or with the GitHub CLI):"
echo
echo "  gh secret set MT_SIGNING_P12_PASSWORD --repo AVZa-BL/mini-projs --body '$PASS'"
echo "  base64 < '$WORK/identity.p12' | gh secret set MT_SIGNING_P12 --repo AVZa-BL/mini-projs"
echo
echo "(Run those now if you want them: the .p12 is deleted when this script ends. Later: ./make_signing_identity.sh --export.)"
read -r -p "Press Return to finish (the temporary files are then deleted). " _ || true
