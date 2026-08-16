#!/usr/bin/env bash
# Wraps dist/Mob Cam.app into a distributable .dmg.
set -e

VERSION="$1"
ROOT_DIR="$2"
OUT_DIR="$3"

APP_PATH="$ROOT_DIR/dist/Mob Cam.app"
DMG_STAGING="$ROOT_DIR/packaging/macos/build/dmg-staging"
DMG_FILE="$OUT_DIR/MobCam-$VERSION.dmg"

rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
cp -R "$APP_PATH" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"

mkdir -p "$OUT_DIR"
rm -f "$DMG_FILE"

hdiutil create -volname "Mob Cam" \
  -srcfolder "$DMG_STAGING" \
  -ov -format UDZO \
  "$DMG_FILE"

echo "==> .dmg built: $DMG_FILE"