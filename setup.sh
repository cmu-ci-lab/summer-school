#!/usr/bin/env bash
# setup.sh — cross-platform environment setup (Windows/Git Bash, Ubuntu, macOS)
# Usage:  chmod +x setup.sh && ./setup.sh
set -e

ENV_NAME="iccp"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

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

CONDA_EXE="$(find_conda)"
if [ -z "$CONDA_EXE" ]; then
  echo "ERROR: conda not found. Install Anaconda or Miniconda first."
  exit 1
fi
echo "conda    : $CONDA_EXE"

# ── 2. Create / update conda environment ──────────────────────────────────────
echo ""
if "$CONDA_EXE" env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "Environment '${ENV_NAME}' already exists — updating..."
  "$CONDA_EXE" env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
else
  echo "Creating environment '${ENV_NAME}'..."
  "$CONDA_EXE" env create -f "$PROJECT_DIR/environment.yml"
fi

# Resolve python/pip inside the env
PYTHON="$("$CONDA_EXE" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)")"
PIP="$("$CONDA_EXE" run -n "$ENV_NAME" python -c "import sys,os; print(os.path.join(os.path.dirname(sys.executable), 'pip' if sys.platform!='win32' else 'Scripts/pip.exe'))")"
echo "Python   : $PYTHON"

# ── 3. Install vmbpy from Vimba X SDK wheel ───────────────────────────────────
echo ""
echo "Looking for vmbpy (Allied Vision Vimba X SDK)..."

VMBPY_WHEEL=""
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
  "$PIP" install "$VMBPY_WHEEL"
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

# ── 4. Linux: udev rule for Thorlabs APT USB (KDC101) ────────────────────────
if [ "$OS" = "linux" ]; then
  echo ""
  UDEV_RULE='SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="faf0", MODE="0666", GROUP="plugdev"'
  UDEV_FILE="/etc/udev/rules.d/99-thorlabs-apt.rules"
  if [ ! -f "$UDEV_FILE" ]; then
    echo "Adding udev rule for Thorlabs APT USB (requires sudo)..."
    echo "$UDEV_RULE" | sudo tee "$UDEV_FILE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "udev rule installed. Reconnect the KDC101 USB cable."
  else
    echo "udev rule already present: $UDEV_FILE"
  fi
fi

# ── 4b. macOS: stage backend (pyftdi + libusb) ───────────────────────────
# The KDC101 uses Thorlabs' custom FTDI PID (0xfaf0).  On macOS:
#   * Apple's FTDI driver ignores that PID, so NO /dev/cu.usbserial-* appears.
#   * pylablib's ft232 backend (pyft232 + libftdi 0.x ABI) segfaults vs libftdi 1.5.
# So we drive the FTDI chip with pyftdi (pure-Python, libusb based, supports
# custom PIDs).  pyftdi needs a matching-architecture libusb; on Apple Silicon
# the only reliable arm64 build comes from conda-forge.  See kinesis_stage.py.
if [ "$OS" = "mac" ]; then
  echo ""
  echo "Installing macOS stage backend (libusb via conda-forge + pyftdi)..."
  "$CONDA_EXE" install -n "$ENV_NAME" -c conda-forge libusb -y
  "$PIP" install pyftdi
  echo "macOS stage backend ready (pyftdi over libusb)."
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
echo "  Activate :  conda activate iccp"
echo "  VSCode   :  Ctrl+Shift+P → Python: Select Interpreter"
echo "              → Python (iccp)"
echo ""

if [ "$OS" = "linux" ]; then
  echo "  Platform notes (Ubuntu):"
  echo "    Stage  : KDC101 confirmed working via pylablib USB APT"
  echo "    Camera : install Vimba X for Linux, then re-run setup"
  echo "    USB    : udev rule installed — reconnect devices"
elif [ "$OS" = "mac" ]; then
  echo "  Platform notes (macOS):"
  echo "    Stage  : KDC101 driven via pyftdi over libusb (kinesis_stage.py)."
  echo "             Apple's FTDI driver ignores Thorlabs' custom PID 0xfaf0,"
  echo "             so there is NO /dev/cu.usbserial-* port — pyftdi/libusb"
  echo "             claim the device directly. move_stage.py auto-selects this."
  echo "    Camera : install Vimba X .pkg for macOS (Intel or Apple Silicon)"
  echo "             wheel path: /Applications/VimbaX_<version>/api/python/vmbpy-*.whl"
elif [ "$OS" = "windows" ]; then
  echo "  Platform notes (Windows):"
  echo "    Stage  : requires Thorlabs Kinesis / APT driver installed"
  echo "    Camera : requires Vimba X SDK installed"
fi
echo "=================================================="
