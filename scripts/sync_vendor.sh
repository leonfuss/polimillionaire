#!/usr/bin/env bash
# Refresh the vendored copy of the lecturer's millionaire_client package
# from a sibling checkout (e.g. if the lecturer ships a fix).
#
# Usage:
#     ./scripts/sync_vendor.sh /path/to/api_client/millionaire_client

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/upstream/millionaire_client"
    exit 1
fi

SRC="$1"
DST="$(dirname "$0")/../src/polimillionaire/_vendor/millionaire_client"

if [ ! -d "$SRC" ]; then
    echo "ERROR: source directory not found: $SRC"
    exit 1
fi

echo "Syncing $SRC -> $DST"
rm -rf "$DST"
cp -R "$SRC" "$DST"
echo "Done. Review the diff with: git diff src/polimillionaire/_vendor/"
