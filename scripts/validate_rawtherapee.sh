#!/bin/bash
# checks and downloads/extracts rawtherapee
# returns absolute path of rawtherapee-cli
set -e

RT_CLI_PATH="$(pwd)/squashfs-root/usr/bin/rawtherapee-cli"
APP_IMAGE="RawTherapee_5.11_release.AppImage"
INSTALL_DIR="$(pwd)/squashfs-root"

validate_rt_cli() {
    # Check for existing installation
    if [ -f "$RT_CLI_PATH" ]; then
        echo "$RT_CLI_PATH"
        return 0
    fi

    # Clean up any previous partial installs
    [ -d "$INSTALL_DIR" ] && rm -rf "$INSTALL_DIR"

    # Download and install
    wget -q "https://rawtherapee.com/shared/builds/linux/${APP_IMAGE}" || {
        echo "Failed to download AppImage" >&2
        return 1
    }

    chmod +x "$APP_IMAGE"
    ./"$APP_IMAGE" --appimage-extract >/dev/null 2>&1 || {
        echo "AppImage extraction failed" >&2
        return 1
    }

    # Verify installation
    if [ -f "$RT_CLI_PATH" ]; then
        echo "$RT_CLI_PATH"
    else
        echo "Installation completed but rawtherapee-cli not found" >&2
        return 1
    fi
}

# Main execution
absolute_path=$(validate_rt_cli)
echo "export RT_CLI_PATH='$absolute_path'"  # Source this to get path
