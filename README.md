# Temporal Video Overlay Removal Toolkit

Clean text or watermark overlays from videos using frame extraction, overlay metadata, and several restoration backends (temporal median, flow-aligned aggregation, FFmpeg delogo, or the E2FGVI-HQ deep model). This README covers the full workflow, including a Google Compute Engine L4 setup for fast inference.

---
## 1. Repository Layout

```
project-root/
├── main.py                    # CLI entrypoint
├── requirements.txt           # Base dependencies (OpenCV, NumPy)
├── temporal_vid_edit/         # Core Python package
│   ├── frames.py              # Frame extraction utilities
│   ├── overlays.py            # Overlay JSON helpers
│   ├── inpaint.py             # Temporal median model
│   ├── flow_inpaint.py        # Optical-flow aggregation
│   ├── ffmpeg_delogo.py       # FFmpeg delogo pipeline
│   └── deep_inpaint.py        # Segment export + external model hook
├── scripts/
│   ├── download_e2fgvi.py     # Clone/download E2FGVI repo & weights
│   ├── run_deep_inpaint.py    # Orchestrates the deep backend
│   └── e2fgvi_infer.py        # Headless wrapper around E2FGVI test script
├── util/overlay-picker.html   # Browser helper for selecting overlay regions
└── frames/, overlays.json, cleaned_*.mp4, etc. (generated artefacts)
```

---
## 2. Preparing Overlay Metadata

These steps run comfortably on your local machine (macOS, Linux, or Windows):

1. **Create a Python virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Extract frames** from the source video:
   ```bash
   python main.py extract input_video.mp4 frames
   ```
   This writes sequential PNG frames to `frames/` and generates `frames/metadata.json` (timestamps, FPS, etc.).

3. **Annotate overlays** via the browser helper:
   ```
   open util/overlay-picker.html   # or double-click in Finder/Explorer
   ```
   - Load a representative frame image.
   - Click the four corners of the overlay.
   - Use the optional “Frame range (e.g. 0-65)” field to duplicate the region across many frames.
   - Click “Add region to list” to append entries to the JSON preview.
   - Copy the output into `overlays.json` (one entry per frame with `frame`, `x1`, `y1`, `x2`, `y2`).

4. **Choose a restoration backend**:
   - `temporal` (default): temporal median fills with optional OpenCV Telea fallback.
   - `flow`: optical-flow alignment + Poisson blending (good for mid-length overlays).
   - `delogo`: FFmpeg’s `delogo` filter on the original video (requires `ffmpeg`).
   - `deep`: E2FGVI-HQ neural inpainting (GPU recommended; see section 3).

---
## 3. GPU Inference on Google Compute Engine (L4)

Running the deep model on CPU is slow (~10 minutes for a short clip). A single NVIDIA L4 (24 GB) on Google Compute Engine cuts this to ~1–2 minutes. These steps assume Ubuntu 22.04.

### 3.1 Launch an L4 instance
1. In Google Cloud Console → Compute Engine → VM instances → **Create Instance**.
2. Use machine family **g2-standard-4** (4 vCPU, 32 GB RAM) with **1× L4 GPU**.
3. Boot disk: Ubuntu 22.04 LTS (x86_64).
4. Allow HTTP/HTTPS if you’ll download files directly; otherwise leave default.
5. Create and SSH into the VM.

### 3.2 Install system packages and NVIDIA drivers
```bash
sudo apt update
sudo apt install -y build-essential git wget ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install NVIDIA drivers (L4 works well with the 535+ series)
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```
Reconnect after the reboot:
```bash
nvidia-smi  # should list the L4 GPU
```

### 3.3 Install Miniforge (Conda-forge distribution)
```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o ~/Miniforge3.sh
bash ~/Miniforge3.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate
```
(If you prefer Miniconda that works too; adjust paths accordingly.)

### 3.4 Create the `e2fgvi` environment with GPU PyTorch + mmcv
```bash
conda create -n e2fgvi python=3.9 -y
conda activate e2fgvi

# PyTorch 1.12.1 with CUDA 11.3 (compatible with mmcv-full 1.7.x)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --index-url https://download.pytorch.org/whl/cu113

# mmcv-full wheel for torch 1.12 / CUDA 11.3
# pip install mmcv-full==1.7.2 -f \
#     https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

pip install -U openmim
mim install mmcv-full==1.7.2

# Remaining dependencies
pip install numpy==1.23.5 opencv-python pillow tqdm pyyaml matplotlib
```
> **Note:** If Google updates the images to newer CUDA versions, ensure the driver supports CUDA 11.3 (most do). You can also move to torch 1.13 + cu117 with the corresponding mmcv wheel if preferred.

### 3.5 Clone the project and copy assets
```bash
cd ~
git clone https://github.com/your-org/temporal-vid-edit.git
cd temporal-vid-edit
```
Transfer `frames/`, `overlays.json`, and any necessary assets from your local machine. Common options:
- `gcloud compute scp --recurse ~/cool/temporal-vid-edit/frames <instance>:/home/<user>/temporal-vid-edit/frames`
- `gsutil cp` to/from a Cloud Storage bucket
- `rsync` over SSH if you have larger datasets

### 3.6 Fetch the E2FGVI-HQ model
```bash
conda activate e2fgvi
python scripts/download_e2fgvi.py
```
If you already have `E2FGVI-HQ-CVPR22.pth`, place it under `third_party/E2FGVI/release_model/`.

### 3.7 Run deep inpainting on the GPU
```bash
source ~/miniforge3/bin/activate && conda activate e2fgvi && \
python scripts/run_deep_inpaint.py \
  --frames frames \
  --overlays overlays.json \
  --output-video cleaned_deep.mp4 \
  --context 6 \
  --python $(which python) \
  --extra --deep-weights third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth \
  --extra --deep-weights-arg=--ckpt
```
- The helper exports masked segments, runs `scripts/e2fgvi_infer.py`, and stitches outputs.
- Results appear as `cleaned_deep.mp4` in the project root (plus temporary data in `/tmp/deep_inpaint_*`).
- Copy the video back to your machine when done and **stop/delete** the VM to avoid charges.

---
## 4. Alternative Backends (no GPU required)

All commands below run in the local `.venv` environment created earlier.

### 4.1 Temporal median model
```bash
source .venv/bin/activate
python main.py inpaint frames overlays.json \
  --output-video cleaned_temporal.mp4 \
  --window 2
```

### 4.2 Flow-aligned model
```bash
source .venv/bin/activate
python main.py inpaint frames overlays.json \
  --method flow \
  --output-video cleaned_flow.mp4 \
  --window 3 \
  --flow-max-sources 8 \
  --flow-max-distance 150
```

### 4.3 FFmpeg delogo
```bash
source .venv/bin/activate
python main.py inpaint frames overlays.json \
  --method delogo \
  --output-video cleaned_delogo.mp4 \
  --codec libx264
```
Make sure `ffmpeg` is installed on your machine (`brew install ffmpeg` or `sudo apt install ffmpeg`).

---
## 5. Troubleshooting & Tips

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` / `Pillow is required` | Install missing packages inside the active environment. The deep wrapper prints explicit hints. |
| mmcv wheel build errors | Ensure torch version matches the wheel index (e.g., torch 1.12.1 with cu113 url). |
| NumPy 2.x incompatibility warnings | Downgrade to `numpy==1.23.5` and use `opencv-python==4.6.0.66` in the `e2fgvi` env. |
| Slow inference | Increase `--flow-max-sources`, use the L4 GPU pipeline, or reduce overlay spans. |
| Overlay JSON ordering | The picker now sorts entries by frame automatically and supports range inputs. |
| OpenAI Codex CLI | Install in either environment (`pip install openai-codex-cli`) and authenticate normally. |

---
## 6. Quick Reference Commands

```bash
# Local prep (one-time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py extract input.mp4 frames

# Annotate overlays
open util/overlay-picker.html

# Fast GPU run on GCE (after environment setup)
conda activate e2fgvi
python scripts/run_deep_inpaint.py --frames frames --overlays overlays.json \
  --output-video cleaned_deep.mp4 --context 6 --python $(which python) \
  --extra --deep-weights third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth \
  --extra --deep-weights-arg=--ckpt

# Alternative local cleanups
python main.py inpaint frames overlays.json --output-video cleaned_temporal.mp4
python main.py inpaint frames overlays.json --method flow --output-video cleaned_flow.mp4
python main.py inpaint frames overlays.json --method delogo --output-video cleaned_delogo.mp4
```

### AWS CDK automation
If you prefer running on AWS instead of GCP, a TypeScript CDK project under `iac/` provisions a GPU host (default `g5.xlarge`, A10G 24 GB, ~\$1/hr on-demand):

```bash
cd iac
npm install
INSTANCE_KEY_NAME=my-key ./deploy.sh       # deploys the stack
./destroy.sh                               # removes all resources
```

Pass additional context like `-c instanceType=g4dn.xlarge` for a cheaper T4 option (~\$0.53/hr) or `-c amiId=ami-...` to pin a specific image.

---
## 7. Conclusion

1. Prepare frames and overlay annotations locally.
2. For heavy lifting, spin up a Google Compute L4 instance, configure the `e2fgvi` environment, and run the deep pipeline.
3. Use the other backends for quick local iterations or further refinement.
4. Copy outputs back, archive overlays, and shut down your cloud resources when done.

Happy editing!
