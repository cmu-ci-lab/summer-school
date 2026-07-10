# Interferometry @ ICCP Summer School 2026

Build a Michelson-style interferometer on a breadboard, align it to see white-light
interference fringes, and use it as a time-domain OCT scanner to recover a depth map of a coin.

## Where to start

**Follow [`instructions.html`](instructions.html) for the guide.** It walks through the full
lab — construction, alignment, and OCT scanning — with photos and videos of every step.
Open it in a browser after cloning (it loads its media from `instructions_media/`, so keep
the two together):

```bash
git clone https://github.com/cmu-ci-lab/summer-school.git
cd summer-school
open instructions.html        # macOS; on Windows just double-click it
```

## Quick start (software only)

```bash
./setup.sh                    # builds the iccp-oct environment (conda or .venv)
conda activate iccp-oct       # or: source .venv/bin/activate
python test_hardware.py       # sanity-check the stage + camera
python visualizer.py          # live camera view + coherence scan panel
```

---

Disclaimer: Claude (Anthropic) was used to assist with part of the code base as well as
formatting of the instructions.
