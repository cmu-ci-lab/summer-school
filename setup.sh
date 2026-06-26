#!/usr/bin/env bash
# setup.sh — cross-platform environment setup (Windows/Git Bash, Ubuntu, macOS)
#
# Uses conda when available; otherwise falls back to a plain Python venv (.venv)
# so the project still installs on machines WITHOUT Anaconda/Miniconda.
# Force the venv path even when conda is present (e.g. to test it):
#     FORCE_VENV=1 ./setup.sh
#
# Usage:  chmod +x setup.sh && ./setup.sh
set -e

ENV_NAME="iccp-oct"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VIMBA_URL="https://www.alliedvision.com/en/support/software-downloads/vimba-x-sdk/vimba-x"

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

# ── 1. Locate conda (skipped if FORCE_VENV=1) ─────────────────────────────────
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
      "$HOME/opt/anaconda3/bin/conda"
      "$HOME/opt/miniconda3/bin/conda"
      "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
      "/usr/local/Caskroom/miniconda/base/bin/conda"
    )
  fi
  for p in "${candidates[@]}"; do
    [ -f "$p" ] && echo "$p" && return
  done
}

if [ "${FORCE_VENV:-0}" = "1" ]; then
  CONDA_EXE=""
  echo "FORCE_VENV=1 → skipping conda, using a Python venv."
else
  CONDA_EXE="$(find_conda)"
fi

# ── 2. Create the environment (conda if available, else venv) ─────────────────
echo ""
if [ -n "$CONDA_EXE" ]; then
  USE_CONDA=1
  echo "conda    : $CONDA_EXE"
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
else
  USE_CONDA=0
  echo "conda    : not found — using a Python venv (.venv) instead."
  PYBIN="$(command -v python3 || command -v python || true)"
  if [ -z "$PYBIN" ]; then
    echo "ERROR: neither conda nor python3 found."
    echo "       Install Python 3.12+ (python.org / brew install python) or Anaconda, then re-run."
    exit 1
  fi
  echo "python   : $PYBIN  ($("$PYBIN" --version 2>&1))"
  PYVER="$("$PYBIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "$PYVER" in
    3.12|3.13) ;;
    *) echo "WARNING: Python $PYVER found, but dependencies are pinned for 3.12."
       echo "         Install Python 3.12 (brew install python@3.12) if pip install fails." ;;
  esac
  echo "Creating venv at: $VENV_DIR"
  "$PYBIN" -m venv "$VENV_DIR"
  if [ "$OS" = "windows" ]; then
    PYTHON="$VENV_DIR/Scripts/python.exe"
    PIP="$VENV_DIR/Scripts/pip.exe"
  else
    PYTHON="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
  fi
  "$PYTHON" -m pip install --upgrade pip
  echo ""
  echo "Installing Python dependencies (mirrors environment.yml)..."
  # opencv-python replaces conda-forge 'opencv'; the rest match environment.yml's pip list.
  "$PIP" install \
    numpy==2.4.6 scipy==1.18.0 matplotlib==3.11.0 Pillow==12.2.0 \
    pyserial==3.5 pylablib==1.4.5 pyftdi \
    opencv-python ipykernel
fi
echo "Python   : $PYTHON"

# ── 3. Install vmbpy from the Vimba X SDK wheel (Allied Vision cameras only) ───
echo ""
echo "Looking for vmbpy (Allied Vision Vimba X SDK)..."

VMBPY_WHEEL=""
if [ "$OS" = "windows" ]; then
  VMBPY_WHEEL="$(find "/c/Program Files/Allied Vision" -maxdepth 6 -name "vmbpy-*.whl" 2>/dev/null | head -1)"
elif [ "$OS" = "linux" ]; then
  # Vimba X on Linux installs to /opt/VimbaX_<version>
  VMBPY_WHEEL="$(find /opt -maxdepth 5 -name "vmbpy-*.whl" 2>/dev/null | head -1)"
elif [ "$OS" = "mac" ]; then
  # The macOS .dmg installs the wheel under /Users/Shared/Allied Vision;
  # older builds used /Applications/VimbaX_<version>. Search both.
  VMBPY_WHEEL="$(find "/Users/Shared/Allied Vision" "/Applications" -maxdepth 6 -name "vmbpy-*.whl" 2>/dev/null | head -1)"
fi

if [ -f "$VMBPY_WHEEL" ]; then
  "$PIP" install "$VMBPY_WHEEL"
  echo "vmbpy installed from: $VMBPY_WHEEL"
else
  echo "WARNING: Vimba X SDK (vmbpy wheel) not found."
  echo "  Only needed for Allied Vision cameras — skip this if you use an IDS camera."
  echo "  Download Vimba X from:"
  echo "    $VIMBA_URL"
  case "$OS" in
    windows)
      echo "  Installer : VimbaX_Setup-2026-1-Win64.exe"
      echo "  Wheel     : C:\\Program Files\\Allied Vision\\Vimba X\\api\\python\\vmbpy-*.whl" ;;
    linux)
      echo "  Installer : VimbaX_Setup-2026-1-Linux64.tar.gz      (x86_64)"
      echo "              VimbaX_Setup-2026-1-Linux_ARM64.tar.gz  (ARM64)"
      echo "  Wheel     : /opt/VimbaX_<version>/api/python/vmbpy-*.whl" ;;
    mac)
      echo "  Installer : VimbaX_Setup-2023-4-macOS.dmg"
      echo "  Wheel     : /Users/Shared/Allied Vision/Vimba X/Vmbpy/vmbpy-*.whl" ;;
  esac
  echo "  Then install it with:"
  echo "    \"$PIP\" install <path-to-vmbpy-*.whl>"
fi

# ── 4. Linux: udev rule for Thorlabs APT USB (KDC101) ─────────────────────────
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

# ── 4b. macOS: stage backend (libusb + pyftdi) ────────────────────────────────
# The KDC101 uses Thorlabs' custom FTDI PID (0xfaf0).  On macOS:
#   * Apple's FTDI driver ignores that PID, so NO /dev/cu.usbserial-* appears.
#   * pylablib's ft232 backend (pyft232 + libftdi 0.x ABI) segfaults vs libftdi 1.5.
# So we drive the FTDI chip with pyftdi (pure-Python, libusb based, supports
# custom PIDs).  pyftdi needs a matching-architecture libusb: from conda-forge
# under conda, or from Homebrew under a venv.  See stage.py / kinesis_stage.py.
if [ "$OS" = "mac" ]; then
  echo ""
  echo "Installing macOS stage backend (libusb + pyftdi)..."
  if [ "$USE_CONDA" = "1" ]; then
    "$CONDA_EXE" install -n "$ENV_NAME" -c conda-forge libusb -y
  else
    if command -v brew &>/dev/null; then
      brew list libusb &>/dev/null || brew install libusb
    else
      echo "WARNING: Homebrew not found — cannot install libusb automatically."
      echo "  Install Homebrew (https://brew.sh), then run:  brew install libusb"
      echo "  (libusb is required to drive the KDC101 stage via pyftdi.)"
    fi
  fi
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
if [ "$USE_CONDA" = "1" ]; then
  echo "  Activate :  conda activate iccp-oct"
  echo "  VSCode   :  Ctrl+Shift+P → Python: Select Interpreter → Python (iccp-oct)"
else
  if [ "$OS" = "windows" ]; then
    echo "  Activate :  .venv\\Scripts\\activate"
  else
    echo "  Activate :  source .venv/bin/activate"
  fi
  echo "  VSCode   :  Ctrl+Shift+P → Python: Select Interpreter → ./.venv"
fi
echo ""

if [ "$OS" = "linux" ]; then
  echo "  Platform notes (Ubuntu):"
  echo "    Stage  : KDC101 confirmed working via pylablib USB APT"
  echo "    Camera : install Vimba X for Linux, then re-run setup"
  echo "    USB    : udev rule installed — reconnect devices"
elif [ "$OS" = "mac" ]; then
  echo "  Platform notes (macOS):"
  echo "    Stage  : KDC101 driven via pyftdi over libusb (stage.py)."
  echo "             Apple's FTDI driver ignores Thorlabs' custom PID 0xfaf0,"
  echo "             so there is NO /dev/cu.usbserial-* port — pyftdi/libusb"
  echo "             claim the device directly."
  echo "    Camera : install Vimba X .dmg (VimbaX_Setup-2023-4-macOS.dmg),"
  echo "             wheel path: /Users/Shared/Allied Vision/Vimba X/Vmbpy/vmbpy-*.whl"
elif [ "$OS" = "windows" ]; then
  echo "  Platform notes (Windows):"
  echo "    Stage  : requires Thorlabs Kinesis / APT driver installed"
  echo "    Camera : requires Vimba X SDK installed (VimbaX_Setup-2026-1-Win64.exe)"
fi
echo "=================================================="
