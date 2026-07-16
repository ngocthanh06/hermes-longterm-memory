#!/usr/bin/env bash
# Bootstrap Longbrain from any directory, so first-time users do not need to
# understand the repository layout before running the real setup workflow.
set -euo pipefail

REPOSITORY="https://github.com/ngocthanh06/longbrain.git"
INSTALL_DIR="${LONGBRAIN_INSTALL_DIR:-$PWD/longbrain}"

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required. Install Git, then run this command again."
  exit 1
fi

if [ -d "$INSTALL_DIR/.git" ]; then
  echo "Updating Longbrain in $INSTALL_DIR"
  git -C "$INSTALL_DIR" pull --ff-only
elif [ -e "$INSTALL_DIR" ]; then
  echo "$INSTALL_DIR already exists but is not a Longbrain Git checkout."
  echo "Choose another location with LONGBRAIN_INSTALL_DIR and try again."
  exit 1
else
  echo "Installing Longbrain in $INSTALL_DIR"
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone --depth 1 "$REPOSITORY" "$INSTALL_DIR"
fi

# Reattach the terminal because setup asks for consent before changing global
# agent configuration, even when this bootstrap script arrived through a pipe.
if [ -r /dev/tty ]; then
  bash "$INSTALL_DIR/setup.sh" </dev/tty
else
  bash "$INSTALL_DIR/setup.sh"
fi
