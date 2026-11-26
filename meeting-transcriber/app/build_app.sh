#!/bin/bash
# Builds "Meeting Transcriber.app" from meeting-transcriber/app/Sources with Apple's
# stock toolchain (Xcode or the Command Line Tools) and bundles the current mt.py,
# VERSION and requirements.txt (the speech engine is installed on first launch). macOS only.
#
#   ./build_app.sh                      # version from ../VERSION, both architectures
#   ./build_app.sh --arch arm64         # Apple silicon only (faster)
#   ./build_app.sh --install            # also copy the app to ~/Applications
#   ./build_app.sh --zip                # also write Meeting-Transcriber-<version>.zip
#   ./build_app.sh --no-runtime         # do not bundle the Python runtime (a system Python is used at setup)
#   ./build_app.sh --sign "<identity>"  # sign with this keychain identity (default: "Meeting Transcriber Signing" when
#                                       # the keychain has it, else ad-hoc; MT_SIGN_IDENTITY in the environment works too)
#
# Signing: every build must carry the same identity, or macOS forgets the app's permissions at the next build (an ad-hoc
# signature is a new identity every time). make_signing_identity.sh creates the identity once.
# The Python runtime bundled per architecture is python-build-standalone (CPython, PSF licence), pinned below and
# verified against the release's SHA256SUMS; it is downloaded once into app/build/runtime-cache/.
# Output: app/build/Meeting Transcriber.app
set -euo pipefail

APP_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$APP_DIR")
VERSION=$(tr -d ' \n' < "$PROJECT_DIR/VERSION")
BUILD=""
OUT="$APP_DIR/build"
ARCHS="arm64 x86_64"
MIN_MACOS="13.0"
ZIP=0
INSTALL=0
RUNTIME=1
PBS_TAG="20261026"
PBS_PY="3.12.16"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG"

SIGN_IDENTITY="${MT_SIGN_IDENTITY:-}"
DEFAULT_IDENTITY="Meeting Transcriber Signing"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION=$2; shift 2 ;;
    --build) BUILD=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --arch) ARCHS=$2; shift 2 ;;
    --zip) ZIP=1; shift ;;
    --install) INSTALL=1; shift ;;
    --no-runtime) RUNTIME=0; shift ;;
    --sign) SIGN_IDENTITY=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ "$(uname -s)" = "Darwin" ] || { echo "ERROR: the app can only be built on macOS." >&2; exit 1; }
# Started by an app or a launch agent, HOME can be unset or empty: --install and `security` (the signing identity's
# keychain is in the account's search list) need the real home folder, from the user database then.
[ -n "${HOME:-}" ] || HOME=$(eval echo "~$(id -un)")
export HOME
command -v xcrun >/dev/null 2>&1 || {
  echo "ERROR: Xcode Command Line Tools are required: run 'xcode-select --install'." >&2; exit 1;
}
case "$VERSION" in *[!0-9A-Za-z.-]*|'') echo "ERROR: bad version '$VERSION'" >&2; exit 2 ;; esac
COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
[ -n "$BUILD" ] || BUILD=$COMMIT

# The signing identity: the one asked for, else the project's own when this Mac has it, else ad-hoc ("-").
if [ -z "$SIGN_IDENTITY" ] && security find-identity -v -p codesigning 2>/dev/null | grep -q "\"$DEFAULT_IDENTITY\""; then
  SIGN_IDENTITY="$DEFAULT_IDENTITY"
fi
if [ -n "$SIGN_IDENTITY" ]; then
  echo "Signing with the identity \"$SIGN_IDENTITY\" (the app keeps its permissions across builds)."
else
  SIGN_IDENTITY="-"
  echo "Ad-hoc signature: macOS forgets the app's permissions at the next build. Run app/make_signing_identity.sh once to keep them."
fi

SOURCES=("$APP_DIR"/Sources/Core/*.swift "$APP_DIR"/Sources/App/*.swift)
# One Objective-C file: the catch around AVAudioEngine's exceptions, which Swift cannot write (ObjCCatch.h/.m). It is
# compiled with clang per architecture and handed to swiftc with its header as the bridging header.
OBJC_HEADER="$APP_DIR/Sources/App/ObjCCatch.h"
OBJC_SOURCE="$APP_DIR/Sources/App/ObjCCatch.m"
mkdir -p "$OUT/obj"
echo "Building Meeting Transcriber $VERSION ($BUILD) for: $ARCHS"
binaries=()
for arch in $ARCHS; do
  echo "  clang + swiftc -target $arch-apple-macos$MIN_MACOS"
  xcrun clang -target "$arch-apple-macos$MIN_MACOS" -fobjc-arc -O2 -c "$OBJC_SOURCE" -o "$OUT/obj/ObjCCatch-$arch.o"
  xcrun swiftc -O -parse-as-library -module-name MeetingTranscriber \
    -target "$arch-apple-macos$MIN_MACOS" -import-objc-header "$OBJC_HEADER" \
    "${SOURCES[@]}" "$OUT/obj/ObjCCatch-$arch.o" -o "$OUT/obj/MeetingTranscriber-$arch"
  binaries+=("$OUT/obj/MeetingTranscriber-$arch")
done
if [ "${#binaries[@]}" -eq 1 ]; then
  cp "${binaries[0]}" "$OUT/obj/MeetingTranscriber"
else
  xcrun lipo -create "${binaries[@]}" -output "$OUT/obj/MeetingTranscriber"
fi

APP="$OUT/Meeting Transcriber.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/engine"
cp "$OUT/obj/MeetingTranscriber" "$APP/Contents/MacOS/MeetingTranscriber"
chmod 755 "$APP/Contents/MacOS/MeetingTranscriber"
sed -e "s/__VERSION__/$VERSION/" -e "s/__BUILD__/$BUILD/" -e "s/__COMMIT__/$COMMIT/" \
  "$APP_DIR/Info.plist" > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"
# The engine the app runs: always the one from this checkout.
for file in mt.py VERSION requirements.txt requirements-mlx.txt requirements-pyannote.txt; do
  cp "$PROJECT_DIR/$file" "$APP/Contents/Resources/engine/$file"
done
# The bundle's version is the one the app compares with its engine setup, so it must equal VERSION.
printf '%s\n' "$VERSION" > "$APP/Contents/Resources/engine/VERSION"
if [ -d "$APP_DIR/AppIcon.iconset" ]; then
  xcrun iconutil -c icns "$APP_DIR/AppIcon.iconset" -o "$APP/Contents/Resources/AppIcon.icns"
fi

# The app's own Python runtime, one per architecture, so nothing has to be installed on the Mac.
if [ "$RUNTIME" = "1" ]; then
  CACHE="$OUT/runtime-cache"
  mkdir -p "$CACHE"
  SUMS="$CACHE/SHA256SUMS-$PBS_TAG"
  [ -s "$SUMS" ] || curl -fsSL "$PBS_URL/SHA256SUMS" -o "$SUMS"
  for arch in $ARCHS; do
    case "$arch" in arm64) pbs_arch=aarch64 ;; x86_64) pbs_arch=x86_64 ;; *) echo "ERROR: no runtime for $arch" >&2; exit 2 ;; esac
    file="cpython-$PBS_PY+$PBS_TAG-$pbs_arch-apple-darwin-install_only_stripped.tar.gz"
    if [ ! -s "$CACHE/$file" ]; then
      echo "  downloading $file"
      curl -fsSL "$PBS_URL/$file" -o "$CACHE/$file.part" && mv "$CACHE/$file.part" "$CACHE/$file"
    fi
    expected=$(grep " $file\$" "$SUMS" | awk '{print $1}')
    actual=$(shasum -a 256 "$CACHE/$file" | awk '{print $1}')
    if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then echo "ERROR: checksum mismatch for $file" >&2; exit 1; fi
    dest="$APP/Contents/Resources/python/$arch"
    mkdir -p "$dest"
    tar -xzf "$CACHE/$file" -C "$dest" --strip-components=1        # the archive holds one folder, python/
    "$dest/bin/python3" --version >/dev/null 2>&1 || [ "$arch" != "$(uname -m)" ] || { echo "ERROR: the bundled runtime does not run" >&2; exit 1; }
    # Every binary and library inside gets the same signature as the app (a signature is required on Apple silicon).
    find "$dest" -type f \( -name '*.dylib' -o -name '*.so' -o -perm +111 \) \
      -exec codesign --force --sign "$SIGN_IDENTITY" --timestamp=none {} \; 2>/dev/null
  done
fi

plutil -lint "$APP/Contents/Info.plist" >/dev/null
# The app's signature (a signature is required on Apple silicon; no developer account is needed for either kind).
codesign --force --sign "$SIGN_IDENTITY" --timestamp=none "$APP"
codesign --verify --deep --strict "$APP"
REQUIREMENT=$(codesign -d -r- "$APP" 2>&1 | grep '^designated' || true)
echo "  $REQUIREMENT"
if [ "$SIGN_IDENTITY" != "-" ] && ! printf '%s' "$REQUIREMENT" | grep -q 'certificate leaf'; then
  echo "ERROR: the app was signed, but its designated requirement does not name the certificate (macOS would not keep the permissions)." >&2
  exit 1
fi
"$APP/Contents/MacOS/MeetingTranscriber" --self-test

echo "Built: $APP"
if [ "$ZIP" = "1" ]; then
  ZIP_FILE="$OUT/Meeting-Transcriber-$VERSION.zip"
  rm -f "$ZIP_FILE"
  ditto -c -k --keepParent "$APP" "$ZIP_FILE"
  echo "Zipped: $ZIP_FILE"
fi
if [ "$INSTALL" = "1" ]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/Meeting Transcriber.app"
  ditto "$APP" "$HOME/Applications/Meeting Transcriber.app"
  echo "Installed: $HOME/Applications/Meeting Transcriber.app"
fi
