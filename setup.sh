#!/usr/bin/env bash
# setup.sh — cross-platform environment setup (Windows/Git Bash, Ubuntu, macOS)
# Usage:  chmod +x setup.sh && ./setup.sh [--camera avt|ids|both|none]
#
#   --camera avt   (default) install vmbpy from the Allied Vision Vimba X SDK wheel
#   --camera ids             install pyueye for IDS uEye cameras
#   --camera both            install both
#   --camera none            skip camera packages (stage only)
set -e

ENV_NAME="iccp-oct"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/$ENV_NAME"

# ── Parse args ────────────────────────────────────────────────────────────────
CAMERA="avt"   # avt | ids | both | none
while [ $# -gt 0 ]; do
  case "$1" in
    --camera) CAMERA="$2"; shift 2 ;;
    --camera=*) CAMERA="${1#*=}"; shift ;;
    -h|--help)
      echo "Usage: ./setup.sh [--camera avt|ids|both|none]"
      exit 0 ;;
    *) echo "Unknown option: $1"; echo "Usage: ./setup.sh [--camera avt|ids|both|none]"; exit 1 ;;
  esac
done
case "$CAMERA" in
  avt|ids|both|none) ;;
  *) echo "ERROR: --camera must be one of: avt, ids, both, none (got '$CAMERA')"; exit 1 ;;
esac

# ── Detect OS ─────────────────────────────────────────────────────────────────
case "$(uname -s)" in
  Linux*)   OS="linux"   ;;
  Darwin*)  OS="mac"     ;;
  MINGW*|MSYS*|CYGWIN*)  OS="windows" ;;
  *)        OS="unknown" ;;
esac

echo "=================================================="
echo "  ICCP 2026 environment setup"
echo "  Platform : $OS"
echo "  Project  : $PROJECT_DIR"
echo "  Camera   : $CAMERA"
echo "=================================================="

# ── 1. Locate conda ───────────────────────────────────────────────────────────
find_conda() {
  # check PATH first
  if command -v conda &>/dev/null; then
    echo "$(command -v conda)"; return
  fi
  local candidates=()
  if [ "$OS" = "windows" ]; then
    candidates=(
      "/c/ProgramData/Anaconda3/Scripts/conda.exe"
      "/c/Users/$USERNAME/Anaconda3/Scripts/conda.exe"
      "/c/Users/$USERNAME/miniconda3/Scripts/conda.exe"
      "/c/Users/$USERNAME/AppData/Local/miniconda3/Scripts/conda.exe"
    )
  elif [ "$OS" = "linux" ]; then
    candidates=(
      "$HOME/anaconda3/bin/conda"
      "$HOME/miniconda3/bin/conda"
      "/opt/anaconda3/bin/conda"
      "/opt/miniconda3/bin/conda"
    )
  elif [ "$OS" = "mac" ]; then
    candidates=(
      "$HOME/anaconda3/bin/conda"
      "$HOME/miniconda3/bin/conda"
      "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
      "/usr/local/Caskroom/miniconda/base/bin/conda"
    )
  fi
  for p in "${candidates[@]}"; do
    [ -f "$p" ] && echo "$p" && return
  done
}

# Pick a base python interpreter for the venv fallback (prefer 3.12).
find_python() {
  for c in python3.12 python3 python; do
    if command -v "$c" &>/dev/null; then
      echo "$(command -v "$c")"; return
    fi
  done
}

# Resolve the python/pip executables inside a venv directory (handles Windows layout).
venv_python() {
  if [ -x "$1/bin/python" ]; then
    echo "$1/bin/python"
  else
    echo "$1/Scripts/python.exe"
  fi
}

CONDA_EXE="$(find_conda)" || true

if [ -n "$CONDA_EXE" ]; then
  BACKEND="conda"
  echo "conda    : $CONDA_EXE"

  # ── 2a. Create / update conda environment ───────────────────────────────────
  echo ""
  if "$CONDA_EXE" env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Environment '${ENV_NAME}' already exists — updating..."
    "$CONDA_EXE" env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
  else
    echo "Creating environment '${ENV_NAME}'..."
    "$CONDA_EXE" env create -f "$PROJECT_DIR/environment.yml"
  fi

  # Resolve python inside the env (pip is invoked as "$PYTHON" -m pip below)
  PYTHON="$("$CONDA_EXE" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)")"
else
  BACKEND="venv"
  echo "conda    : not found — falling back to Python venv"

  BASE_PYTHON="$(find_python)"
  if [ -z "$BASE_PYTHON" ]; then
    echo "ERROR: neither conda nor python3 found. Install Python 3.12+ (or Anaconda/Miniconda) first."
    exit 1
  fi
  echo "python   : $BASE_PYTHON"

  # ── 2b. Create / reuse venv ─────────────────────────────────────────────────
  echo ""
  if [ -d "$VENV_DIR" ]; then
    echo "Virtualenv already exists at $VENV_DIR — reusing..."
  else
    echo "Creating virtualenv at $VENV_DIR..."
    "$BASE_PYTHON" -m venv "$VENV_DIR"
  fi

  PYTHON="$(venv_python "$VENV_DIR")"

  echo "Upgrading pip and installing requirements..."
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
fi
echo "Python   : $PYTHON"

# ── 3. Install camera packages (selected by --camera) ─────────────────────────

install_avt() {
  # Allied Vision: vmbpy from the Vimba X SDK wheel (NOT on PyPI).
  echo ""
  echo "Looking for vmbpy (Allied Vision Vimba X SDK)..."

  local VMBPY_WHEEL=""
  if [ "$OS" = "windows" ]; then
    VMBPY_WHEEL="/c/Program Files/Allied Vision/Vimba X/api/python/vmbpy-1.1.0-py3-none-win_amd64.whl"
  elif [ "$OS" = "linux" ]; then
    # Vimba X on Linux is typically installed to /opt/VimbaX_<version>
    VMBPY_WHEEL="$(find /opt -maxdepth 4 -name "vmbpy-*.whl" 2>/dev/null | head -1)"
  elif [ "$OS" = "mac" ]; then
    # Allied Vision installs Vimba X to /Applications/VimbaX_<version>/
    VMBPY_WHEEL="$(find /Applications -maxdepth 6 -name "vmbpy-*.whl" 2>/dev/null | head -1)"
  fi

  if [ -f "$VMBPY_WHEEL" ]; then
    "$PYTHON" -m pip install "$VMBPY_WHEEL"
    echo "vmbpy installed from: $VMBPY_WHEEL"
  else
    echo "WARNING: Vimba X SDK wheel not found."
    echo "  Download Vimba X for your platform from:"
    echo "  https://www.alliedvision.com/en/products/software.html"
    if [ "$OS" = "linux" ]; then
      echo "  After install, the wheel is usually at:"
      echo "  /opt/VimbaX_<version>/api/python/vmbpy-*.whl"
    elif [ "$OS" = "mac" ]; then
      echo "  macOS: install the Vimba X .pkg (Intel or Apple Silicon build)."
      echo "  After install, the wheel is usually at:"
      echo "  /Applications/VimbaX_<version>/api/python/vmbpy-*.whl"
      echo "  Then run:  pip install /Applications/VimbaX_<version>/api/python/vmbpy-*.whl"
    fi
    if [ "$OS" != "mac" ]; then
      echo "  Then run:  pip install <path-to-wheel>"
    fi
  fi
}

install_ids_suite_linux() {
  # Install the IDS Software Suite from its self-extracting installer (ueye_*.run),
  # which provides libueye_api.so. The .run requires root and --auto runs it
  # non-interactively. We search common locations for the installer.
  local RUN
  RUN="$(find "$PROJECT_DIR" "$HOME/Downloads" -maxdepth 2 -iname "ueye_*_amd64.run" 2>/dev/null | head -1)"
  if [ -z "$RUN" ]; then
    echo "  IDS Software Suite installer (ueye_*.run) not found."
    echo "  Download it from https://en.ids-imaging.com/downloads.html and place it in"
    echo "  $PROJECT_DIR or ~/Downloads, then re-run:  ./setup.sh --camera ids"
    return
  fi
  echo "  Found IDS installer: $RUN"
  echo ""
  echo "  ┌─ ADMINISTRATOR ACCESS REQUIRED (sudo) ─────────────────────────────"
  echo "  │ The next steps need root and will prompt for your password. They will:"
  command -v apt-get &>/dev/null && \
  echo "  │   1. Install the OpenMP runtime that the IDS driver depends on:"
  command -v apt-get &>/dev/null && \
  echo "  │        sudo apt-get install -y libomp5"
  echo "  │   2. Install the IDS Software Suite (uEye driver + libueye_api.so):"
  echo "  │        sudo $RUN --auto"
  echo "  │   3. Refresh the dynamic linker cache:"
  echo "  │        sudo ldconfig"
  echo "  └────────────────────────────────────────────────────────────────────"
  echo ""

  # libueye_api.so depends on the OpenMP runtime (libomp.so.5); install it on
  # Debian/Ubuntu so pyueye can load the driver.
  if command -v apt-get &>/dev/null; then
    echo "  [sudo] apt-get install -y libomp5"
    sudo apt-get install -y libomp5 || echo "  WARNING: could not install libomp5 (install manually: sudo apt-get install libomp5)"
  fi

  echo "  [sudo] $RUN --auto"
  chmod +x "$RUN" 2>/dev/null || true
  local workdir; workdir="$(mktemp -d)"
  if ( cd "$workdir" && sudo "$RUN" --auto ); then
    echo "  [sudo] ldconfig"
    sudo ldconfig 2>/dev/null || true
    echo "  IDS Software Suite installed."
  else
    echo "  WARNING: IDS Software Suite install failed."
    echo "    Likely causes: sudo unavailable here, or an older uEye version is installed."
    echo "    Install manually with:  sudo $RUN --auto"
    echo "    (If an old version exists, uninstall it first: sudo ueyesetup -r all)"
  fi
  rm -rf "$workdir"
}

install_ids() {
  # IDS uEye: pyueye Python wrapper (on PyPI). The underlying IDS Software Suite
  # / uEye driver must be installed separately for the wrapper to find its lib.
  echo ""
  echo "Installing pyueye (IDS uEye camera)..."
  "$PYTHON" -m pip install pyueye

  if "$PYTHON" -c "from pyueye import ueye" 2>/dev/null; then
    echo "pyueye installed and importable."
    return
  fi

  echo "pyueye installed, but the IDS runtime library was not found."
  if [ "$OS" = "linux" ]; then
    install_ids_suite_linux
    if "$PYTHON" -c "from pyueye import ueye" 2>/dev/null; then
      echo "pyueye now importable."
    fi
  elif [ "$OS" = "mac" ]; then
    echo "  macOS: install the IDS Software Suite .dmg from"
    echo "  https://en.ids-imaging.com/downloads.html"
  elif [ "$OS" = "windows" ]; then
    echo "  Windows: install the IDS Software Suite (provides ueye_api.dll) from"
    echo "  https://en.ids-imaging.com/downloads.html"
  fi
}

case "$CAMERA" in
  avt)  install_avt ;;
  ids)  install_ids ;;
  both) install_avt; install_ids ;;
  none) echo ""; echo "Skipping camera packages (--camera none)." ;;
esac

# ── 4. Linux: udev rule for Thorlabs APT USB (KDC101) ────────────────────────
if [ "$OS" = "linux" ]; then
  echo ""
  # The KDC101 exposes a serial tty (/dev/ttyUSB0) in the "tty" subsystem; its
  # USB ids live on a parent device, so we match SUBSYSTEM=="tty" with ATTRS{...}
  # (with the S — walks up to the parent). MODE 0666 makes it usable without the
  # user being in the dialout group. (A SUBSYSTEM=="usb"/ATTR rule would only
  # touch /dev/bus/usb/* and never /dev/ttyUSB0.)
  UDEV_RULE='SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="faf0", MODE="0666", GROUP="dialout"'
  UDEV_FILE="/etc/udev/rules.d/99-thorlabs-apt.rules"
  if [ ! -f "$UDEV_FILE" ]; then
    echo "┌─ ADMINISTRATOR ACCESS REQUIRED (sudo) ─────────────────────────────"
    echo "│ Granting your user access to the Thorlabs USB stage controller needs"
    echo "│ root and will prompt for your password. It will:"
    echo "│   1. Write a udev rule so the device is accessible without root:"
    echo "│        $UDEV_FILE"
    echo "│        ($UDEV_RULE)"
    echo "│   2. Reload udev rules:  sudo udevadm control --reload-rules"
    echo "│   3. Re-apply to the device (add event, so MODE takes effect now):"
    echo "│        sudo udevadm trigger --action=add --subsystem-match=tty"
    echo "└────────────────────────────────────────────────────────────────────"
    echo "[sudo] writing $UDEV_FILE + reloading udev"
    # NOTE: a plain 'udevadm trigger' fires a 'change' event, which does NOT
    # re-apply MODE/GROUP. We must use --action=add for the new permissions to
    # take effect on the already-connected device (a replug would also work).
    if echo "$UDEV_RULE" | sudo tee "$UDEV_FILE" > /dev/null \
       && sudo udevadm control --reload-rules \
       && sudo udevadm trigger --action=add --subsystem-match=tty; then
      echo "udev rule installed and applied (or just replug the KDC101 USB cable)."
    else
      echo "WARNING: could not install udev rule (sudo unavailable or denied)."
      echo "  To install it later, run:"
      echo "    echo '$UDEV_RULE' | sudo tee $UDEV_FILE"
      echo "    sudo udevadm control --reload-rules"
      echo "    sudo udevadm trigger --action=add --subsystem-match=tty"
    fi
  else
    echo "udev rule already present: $UDEV_FILE"
  fi
fi

# ── 5. Register Jupyter kernel ────────────────────────────────────────────────
echo ""
echo "Registering Jupyter kernel..."
"$PYTHON" -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
echo "Kernel registered."

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup complete!"
echo ""
if [ "$BACKEND" = "conda" ]; then
  echo "  Activate :  conda activate $ENV_NAME"
else
  if [ "$OS" = "windows" ]; then
    echo "  Activate :  source $ENV_NAME/Scripts/activate"
  else
    echo "  Activate :  source $ENV_NAME/bin/activate"
  fi
fi
echo "  VSCode   :  Ctrl+Shift+P → Python: Select Interpreter"
echo "              → Python ($ENV_NAME)"
echo ""

if [ "$OS" = "linux" ]; then
  echo "  Platform notes (Ubuntu):"
  echo "    Stage  : KDC101 confirmed working via pylablib USB APT"
  echo "    Camera : install Vimba X for Linux, then re-run setup"
  echo "    USB    : udev rule installed — reconnect devices"
elif [ "$OS" = "mac" ]; then
  echo "  Platform notes (macOS):"
  echo "    Stage  : KDC101 connects via Apple's built-in FTDI driver"
  echo "             (exposes /dev/cu.usbserial-*) — no kext changes needed"
  echo "    Camera : install Vimba X .pkg for macOS (Intel or Apple Silicon)"
  echo "             wheel path: /Applications/VimbaX_<version>/api/python/vmbpy-*.whl"
elif [ "$OS" = "windows" ]; then
  echo "  Platform notes (Windows):"
  echo "    Stage  : requires Thorlabs Kinesis / APT driver installed"
  echo "    Camera : requires Vimba X SDK installed"
fi
echo "=================================================="
