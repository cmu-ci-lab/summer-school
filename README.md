# Interferometry @ ICCP Summer School 2026

Control code for a depth-from-focus / interferometry rig: a **Thorlabs KDC101**
brushed-motor stage plus an optional camera (Allied Vision Vimba X or IDS).

This guide covers setting it up and running it on **any Linux computer**.

---

## 1. Requirements

- Linux (tested on Ubuntu, kernel 5.15)
- **Python 3.12+** (tested on 3.13) — or Anaconda/Miniconda
- The Thorlabs KDC101 stage controller connected over USB (it enumerates as an
  FTDI USB-serial device, USB id `0403:faf0`, e.g. `/dev/ttyUSB0`)
- *(optional)* Allied Vision Vimba X SDK for the camera

No conda is required — `setup.sh` automatically falls back to a Python `venv`
when conda is not installed.

---

## 2. One-shot setup

```bash
chmod +x setup.sh
./setup.sh                 # default: Allied Vision (AVT) camera
./setup.sh --camera ids    # IDS uEye camera instead
./setup.sh --camera both   # both camera stacks
./setup.sh --camera none   # stage only, no camera packages
```

`setup.sh` will:

1. **Detect conda.** If found, it creates/updates the conda env `iccp-oct` from
   `environment.yml`. If not found, it creates a `venv` at `./iccp-oct/` and
   installs `requirements.txt` into it.
2. **Install the selected camera package(s)** (see §5):
   - `avt` (default) → `vmbpy` from a local Vimba X SDK wheel
   - `ids` → `pyueye` from PyPI
   If the camera's SDK/driver isn't installed you'll just get a warning, and the
   stage will still work.
3. **Install a udev rule** for the Thorlabs USB device (needs `sudo`; see §3 if
   it can't prompt for a password).
4. **Register a Jupyter kernel** named `Python (iccp-oct)` for use in VS Code /
   Jupyter.

> **Administrator (sudo) access.** Some steps need root and will **prompt for
> your password**. `setup.sh` always prints exactly what it is about to run with
> `sudo` before doing it, so you can see what's happening. The privileged actions
> are, at most:
> - **Stage USB access** — write `/etc/udev/rules.d/99-thorlabs-apt.rules`, then
>   `udevadm control --reload-rules` + `udevadm trigger` (§3).
> - **IDS camera only** — `apt-get install -y libomp5`, run the IDS driver
>   installer `sudo ueye_*.run --auto`, then `ldconfig` (§5).
>
> If `sudo` isn't available, each step is skipped with a warning and the exact
> manual command to run later — nothing fails silently.

### Activate the environment

```bash
# venv (no conda):
source iccp-oct/bin/activate

# conda:
conda activate iccp-oct
```

---

## 3. USB permissions (one-time, important)

By default `/dev/ttyUSB0` is owned by `root:dialout` with mode `0660`, so a normal
user gets **"Permission denied"** when opening it. `setup.sh` tries to install a
udev rule to fix this, but it needs `sudo` — if the script ran non-interactively
and skipped it, install the rule manually:

```bash
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="faf0", MODE="0666", GROUP="dialout"' \
  | sudo tee /etc/udev/rules.d/99-thorlabs-apt.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=tty   # NOT a plain 'trigger' — see below
```

> **Use `--action=add`.** A plain `udevadm trigger` fires a *change* event, which
> reloads the rules but does **not** re-apply `MODE`/`GROUP` to an
> already-connected device — so the permissions won't change until you replug or
> reboot. `--action=add` (or physically replugging the USB cable) applies them
> immediately.

> **Important — the rule must match the `tty` subsystem.** The KDC101 appears as
> a serial port `/dev/ttyUSB0`, which lives in the **`tty`** subsystem; its USB
> vendor/product ids are on a *parent* device. So the rule uses
> `SUBSYSTEM=="tty"` with **`ATTRS{...}`** (the `S` walks up to the parent). A
> `SUBSYSTEM=="usb"` + `ATTR{...}` rule only changes `/dev/bus/usb/*` and leaves
> `/dev/ttyUSB0` as `root:dialout 0660` — so it appears to work until you reboot,
> then fails with "Permission denied".

`udevadm trigger` re-applies the rule to the already-connected device. To open
the **currently** plugged device immediately (e.g. for a one-off test):

```bash
sudo chmod 666 /dev/ttyUSB0      # temporary — does NOT survive a reboot/replug
```

**Alternative** (permanent, but requires logging out and back in to take effect):

```bash
sudo usermod -aG dialout $USER
```

`dialout` is the Linux group traditionally granted access to serial ports.

---

## 4. Running the stage

With the environment activated and permissions set:

```bash
python move_stage.py
```

`move_stage.py` connects to the stage, homes it, and steps through a few
positions. **Homing physically moves the stage** — make sure it is clear to
travel first.

### How the stage connects on Linux

`stage.py` connects to the KDC101 by **opening the serial port directly**
(`/dev/ttyUSB0`, located by FTDI VID `0x0403`) rather than by APT serial number.
This avoids needing the `ft232`/`libftdi` backend, which is often missing on
Linux.

### Motor scale (units)

The conversion from internal steps to millimetres is **auto-detected**: pylablib
queries the controller for the stage ID that was programmed into it via Thorlabs
Kinesis, and looks up the correct steps-per-mm.

If auto-detection fails you'll see a warning that positions are in *raw steps*.
In that case, pass the stage model explicitly:

```python
from stage import ThorlabsStage
stage = ThorlabsStage(units="mm", stage="Z825")   # or Z806, Z812, MTS25-Z8, MTS50-Z8, PRM1-Z8, CR1-Z7, ...
stage.connect()
```

Or program the stage type once in Thorlabs Kinesis so it persists on the controller.

---

## 5. Camera (optional)

Two camera stacks are supported; pick one with the `--camera` flag when running
`setup.sh`. Each Python package needs the matching vendor driver installed at the
system level.

### Allied Vision (AVT) — `--camera avt` (default)

Uses the **Vimba X SDK** (`vmbpy`), which is **not** on PyPI.

1. Download Vimba X for Linux from
   <https://www.alliedvision.com/en/products/software.html>
2. Install it (typically to `/opt/VimbaX_<version>/`).
3. Run `./setup.sh` (or `--camera avt`) — it finds and installs the wheel at
   `/opt/VimbaX_<version>/api/python/vmbpy-*.whl`.

Scripts: `camera.py`, `capture_camera.py`.

### IDS uEye — `--camera ids`

Uses **`pyueye`** (on PyPI). The wrapper needs the underlying **IDS Software
Suite / uEye driver** for its native library `libueye_api.so`, which in turn
depends on the **OpenMP runtime** `libomp.so.5`.

On Linux, `./setup.sh --camera ids` handles all of this for you:

1. `pip install pyueye` into the environment.
2. If `pyueye` can't load the driver, it looks for the IDS installer
   `ueye_*_amd64.run` in the **project directory** or **`~/Downloads`**.
3. It then (with explicit sudo prompts):
   - `sudo apt-get install -y libomp5`  ← the OpenMP runtime the driver needs
   - `sudo ueye_*.run --auto`           ← installs the uEye driver + `libueye_api.so`
   - `sudo ldconfig`                    ← refreshes the linker cache

So the full IDS setup is just:

```bash
# 1. Download the IDS Software Suite (.tgz) for Linux — requires an IDS account:
#    https://en.ids-imaging.com/downloads.html
# 2. Extract it; it yields ueye_<version>_amd64.run (leave it in ~/Downloads).
# 3. Run setup, which finds and installs it:
./setup.sh --camera ids
```

> The IDS installer is a self-extracting CMake/CPack archive; `--auto` runs it
> non-interactively. It **refuses to install over an older uEye version** — if so,
> remove the old one first: `sudo ueyesetup -r all`.

To install the driver manually instead:

```bash
sudo apt-get install -y libomp5
sudo ~/Downloads/ueye_4.96.1.2054_amd64.run --auto
sudo ldconfig
```

Verify it works:

```bash
iccp-oct/bin/python -c "from pyueye import ueye; print(ueye.HIDS(0))"
```

Scripts: `ids_camera.py`, `capture_ids.py`.

Without a camera SDK the stage still works; only the camera scripts need it. Use
`--camera none` to skip camera packages entirely.

---

## 6. Project files

| File | Purpose |
|------|---------|
| `setup.sh` | Environment setup (conda or venv) |
| `environment.yml` / `requirements.txt` | Dependency lists |
| `stage.py` | `ThorlabsStage` wrapper for the KDC101 |
| `move_stage.py` | Minimal stage move/home demo |
| `find_thorlabs_port.py` | List connected serial/Thorlabs devices |
| `camera.py`, `capture_camera.py` | Allied Vision (Vimba X) capture |
| `ids_camera.py`, `capture_ids.py` | IDS camera capture |
| `depth_from_focus.py`, `run_depth.py`, `visualise_depth.py` | Depth-from-focus pipeline |

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Permission denied: '/dev/ttyUSB0'` | Install the udev rule / `chmod 666` the device (§3) |
| `No Thorlabs FTDI serial port found (VID 0x0403)` | Check the USB cable; run `python find_thorlabs_port.py` |
| `UserWarning: ft232 backend is not available` | Harmless — we connect via the serial port directly |
| Positions look wrong / "raw steps" warning | Pass `stage="..."` explicitly (§4) |
| `vmbpy` / AVT camera import errors | Install the Vimba X SDK and re-run setup (§5) |
| `pyueye`: `found ['libueye_api.so.1'], but it's not usable` | The IDS driver is missing its OpenMP dependency: `sudo apt-get install -y libomp5` (§5) |
| `pyueye`: `could not find any library for ueye_api` | IDS Software Suite not installed — run `./setup.sh --camera ids` with the `ueye_*.run` in `~/Downloads` (§5) |
| IDS installer: `old version of this driver installed` | `sudo ueyesetup -r all`, then re-run (§5) |
