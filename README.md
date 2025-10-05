# Temporal Video Overlay Removal Toolkit

This repository contains a command-line workflow for extracting video frames, describing unwanted overlays, and cleaning them using classical, optical-flow, FFmpeg delogo, or state-of-the-art deep inpainting models. The instructions below walk through the full setup on macOS (Apple Silicon) and describe how to run each stage end to end.

---
## 1. Prerequisites

- macOS with Apple Silicon (tested on M3)
- Homebrew (`/opt/homebrew/bin/brew`)
- Python 3.9+ (system Python is fine for the classical pipeline)
- Xcode command-line tools (`xcode-select --install`)

### 1.1 Clone the project
```bash
cd ~/cool
git clone https://github.com/your-org/temporal-vid-edit.git
cd temporal-vid-edit
```

---
## 2. Base environment for classical/flow pipelines

1. Create a Python virtual environment (example with `python3`):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify OpenCV can load and write frames:
   ```bash
   python -m pip show opencv-python
   ```

---
## 3. Deep inpainting environment (E2FGVI-HQ)

The deep pipeline uses a separate Conda environment tuned to the model’s dependencies.

### 3.1 Install Miniforge (Conda-forge variant)
```bash
# Download and install silently to ~/miniforge3
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o ~/Miniforge3-MacOSX-arm64.sh
bash ~/Miniforge3-MacOSX-arm64.sh -b -p ~/miniforge3
```

### 3.2 Accept Anaconda ToS (if prompted)
Miniforge defaults to conda-forge channels, which do not require ToS acceptance. If the system prompts for Anaconda ToS, run:
```bash
eval "$(~/miniforge3/bin/conda shell.zsh hook)"
conda tos accept --channel https://repo.anaconda.com/pkgs/main --tos-root $HOME/.conda_tos
conda tos accept --channel https://repo.anaconda.com/pkgs/r --tos-root $HOME/.conda_tos
```

### 3.3 Create the `e2fgvi` environment
```bash
eval "$(~/miniforge3/bin/conda shell.zsh hook)"
conda create -n e2fgvi python=3.9 -y
conda activate e2fgvi
```

### 3.4 Install compatible PyTorch and MMCV
```bash
# PyTorch / torchvision / torchaudio CPU wheels compatible with mmcv 1.7.x
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install mmcv-full for torch 1.12 CPU
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.12.0/index.html
```

### 3.5 Install remaining dependencies
```bash
pip install numpy==1.23.5 opencv-python==4.6.0.66 pillow tqdm pyyaml matplotlib
```

---
## 4. Project setup tasks

### 4.1 Download the deep model
The toolkit provides helper scripts under `scripts/`.

```bash
# From project root
eval "$(~/miniforge3/bin/conda shell.zsh hook)"
conda activate e2fgvi
python scripts/download_e2fgvi.py  # clones repo into third_party/E2FGVI and pulls checkpoints
```

If you already have the checkpoint file (e.g., `third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth`), skip the download step.

### 4.2 Verify helper scripts load
```bash
python -m py_compile scripts/run_deep_inpaint.py scripts/e2fgvi_infer.py
```

---
## 5. Workflow overview

1. **Extract frames** from the source video.
2. **Annotate overlays** (JSON file listing frame indices and bounding boxes).
3. **Clean overlays** using one of the supported methods.
4. **Inspect outputs** (frames and/or rendered video).

All commands assume you are in the project root.

---
## 6. Frame extraction
```bash
source .venv/bin/activate
python main.py extract input_video.mp4 frames
```
- Outputs numbered PNGs (`frames/frame_000000.png`, …) and `frames/metadata.json` capturing timestamps and FPS.

---
## 7. Create overlay metadata

Use `util/overlay-picker.html` to load a representative frame and click four corners of the overlay. The tool snaps to grid, produces JSON entries, and supports multiple regions per frame.

Example `overlays.json` entry:
```json
[
  {"frame": 0, "x1": 14, "y1": 52, "x2": 98, "y2": 84},
  {"frame": 1, "x": 14, "y": 52, "width": 84, "height": 32}
]
```

Only list frames that actually contain the intrusive text or watermark.

---
## 8. Cleaning overlays

### 8.1 Temporal median (default)
```bash
source .venv/bin/activate
python main.py inpaint frames overlays.json \
  --output-video cleaned_temporal.mp4 \
  --window 2
```
- Useful for short-lived overlays or when neighboring frames provide clean pixels.

### 8.2 Optical-flow aggregation (`--method flow`)
```bash
source .venv/bin/activate
python main.py inpaint frames overlays.json \
  --method flow \
  --output-video cleaned_flow.mp4 \
  --window 3 \
  --flow-max-sources 8 \
  --flow-max-distance 200
```
- Warps overlay-free frames toward the target and blends with Poisson cloning for smoother fills.

### 8.3 FFmpeg delogo (`--method delogo`)
```bash
source .venv/bin/activate
python main.py inpaint frames overlays.json \
  --method delogo \
  --output-video cleaned_delogo.mp4 \
  --codec libx264
```
- Delegates to FFmpeg’s `delogo` filter using the original video reference (requires `ffmpeg` installed).

### 8.4 Deep inpainting (`--method deep`)
1. Activate the `e2fgvi` environment:
   ```bash
   eval "$(~/miniforge3/bin/conda shell.zsh hook)"
   conda activate e2fgvi
   ```

2. Run the helper script (Point `--deep-weights` to your downloaded checkpoint if different):
   ```bash
   python scripts/run_deep_inpaint.py \
     --frames frames \
     --overlays overlays.json \
     --output-video cleaned_deep.mp4 \
     --context 6 \
     --python $(which python) \
     --extra --deep-weights third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth \
     --extra --deep-weights-arg=--ckpt
   ```
- The runner exports masked segments, invokes `scripts/e2fgvi_infer.py`, and stitches the hallucinated frames back into the master sequence.

3. Optional flags:
   - `--extra --deep-extra-arg=--step 15` to adjust E2FGVI’s stride.
   - `--deep-context <N>` to expand temporal padding around overlay spans.

---
## 9. Outputs and verification

- Cleaned video: `cleaned_*.mp4`
- Optional cleaned frames: use `--output-frames cleaned_frames` to inspect per-frame results.
- Temporary deep inpainting clips: set `--deep-keep-temp` when debugging to keep the exported segment directories under `/tmp/deep_inpaint_*`.

Inspect the cleaned video with any media player (`open cleaned_deep.mp4`).

---
## 10. Troubleshooting

- **mmcv-full install errors**: ensure you pair PyTorch 1.12.1 with the CPU wheel index `torch1.12.0`. Newer PyTorch versions (2.x) are incompatible with the released mmcv-full wheels.
- **NumPy 2.x warnings**: downgrade NumPy to `1.23.5` in the `e2fgvi` environment and use `opencv-python==4.6.0.66` to maintain compatibility.
- **Terms of Service errors (Anaconda)**: Miniforge avoids these by default. If you’re using Anaconda, set `CONDA_TOS_ROOT=$HOME/.conda_tos` before accepting.
- **Missing packages**: The deep wrapper (`scripts/e2fgvi_infer.py`) prints explicit hints if PyTorch, Pillow, or OpenCV cannot be imported.
- **Slow inference**: E2FGVI is GPU-accelerated but can run on CPU at the cost of runtime. Consider reducing context or overlay span for faster iterations.

---
## 11. Directory structure reference

```
project-root/
├── main.py                      # CLI entrypoint
├── requirements.txt             # Base dependencies
├── scripts/
│   ├── download_e2fgvi.py       # Clone E2FGVI repo & download checkpoint
│   ├── run_deep_inpaint.py      # Orchestrates deep inpainting pipeline
│   └── e2fgvi_infer.py          # Headless wrapper around E2FGVI test script
├── temporal_vid_edit/
│   ├── frames.py                # Frame extraction utilities
│   ├── overlays.py              # Overlay parsing helpers
│   ├── inpaint.py               # Temporal median inpainting
│   ├── flow_inpaint.py          # Optical-flow aggregation inpainting
│   ├── ffmpeg_delogo.py         # FFmpeg delogo pipeline
│   └── deep_inpaint.py          # Segment export + model orchestration
├── util/overlay-picker.html     # Browser tool for selecting overlay bounds
├── frames/                      # (generated) extracted frames + metadata
├── overlays.json                # (user) overlay annotations
├── cleaned_*.mp4                # (generated) cleaned videos
└── third_party/E2FGVI/          # (downloaded) deep model repository
```

---
## 12. Summary

1. Set up two environments: `.venv` for classical methods, `e2fgvi` (Conda) for deep inpainting.
2. Extract frames, annotate overlays, and run the desired cleaning method.
3. Use the helper scripts to manage E2FGVI download, inference, and post-processing without manual path juggling.
4. Inspect the cleaned outputs and iterate on overlay annotations or method parameters as needed.

Happy editing!
