#!/usr/bin/env bash
# install.sh — copy bin/* into ~/.local/bin/, make executable.
# Idempotent: re-running upgrades in place.
set -euo pipefail

DEST="${INSTALL_DIR:-$HOME/.local/bin}"
SRC="$(cd "$(dirname "$0")" && pwd)/bin"

if [[ ! -d "$SRC" ]]; then
  echo "install.sh: can't find bin/ next to this script" >&2
  exit 1
fi

mkdir -p "$DEST"
n=0
for f in "$SRC"/*; do
  [[ -f "$f" ]] || continue
  install -m 0755 "$f" "$DEST/$(basename "$f")"
  printf "  ✓ %s\n" "$(basename "$f")"
  n=$((n + 1))
done

echo
echo "Installed $n wrappers to $DEST"

# PATH check
case ":$PATH:" in
  *":$DEST:"*) ;;
  *)
    echo
    echo "⚠ $DEST is NOT in your PATH. Add this to your shell rc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac

cat <<'EOF'

Next steps:
  1. CDI spec (one-time, needs sudo):
       sudo apt install -y nvidia-container-toolkit
       sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

  2. Video2X side:
       v2x-install-models                                  # ~270 MB community models

  3. Upscale-A-Video side:
       podman build -t localhost/uav:latest containers/uav/   # ~10 GB image
       uav-install-models                                  # ~10 GB model weights

  See docs/CHEATSHEET.md for usage recipes.
EOF
