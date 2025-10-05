# Google Compute Engine Setup Guide

This guide assumes a fresh Ubuntu 22.04+ VM on Google Cloud with an NVIDIA L4 or similar GPU. Run each block in sequence after SSH'ing into the instance.

---
## 1. Update base system and install utilities
```bash
sudo apt update
sudo apt install -y build-essential git curl wget ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev unzip python3-pip
```

## 2. Install NVIDIA drivers (only needed on stock Ubuntu images)
```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
# reconnect via SSH once reboot completes
nvidia-smi   # verify GPU is detected
```

## 3. Install Miniforge (Conda-forge distribution)
```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o ~/Miniforge3.sh
bash ~/Miniforge3.sh -b -p $HOME/miniforge3
source ~/miniforge3/bin/activate
```

Optionally add to shell profile:
```bash
echo "source \$HOME/miniforge3/bin/activate" >> ~/.bashrc
```

## 4. Create the deep inpainting Conda environment
```bash
conda create -n e2fgvi python=3.9 -y
conda activate e2fgvi

# PyTorch 1.12.1 with CUDA 11.3 (works with mmcv 1.7.x on L4/G4)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --index-url https://download.pytorch.org/whl/cu113

# mmcv-full and supporting libs
pip install mmcv-full==1.7.2 -f \
    https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

pip install numpy==1.23.5 opencv-python pillow tqdm pyyaml matplotlib gdown
```

## 5. Clone the project (or pull latest)
```bash
cd ~/cool
git clone https://github.com/your-org/temporal-vid-edit.git   # skip if repo already present
cd temporal-vid-edit
```

## 6. Prepare data
- Copy `input_video.mp4`, `frames/`, and `overlays.json` to the VM (via `gcloud compute scp`, `gsutil`, or other transfer).
- If you need to extract frames on the VM:
  ```bash
  source .venv/bin/activate  # if using the local virtualenv, otherwise use Conda or system Python
  python main.py extract input_video.mp4 frames
  deactivate
  ```

## 7. Download the E2FGVI-HQ model checkpoint
```bash
conda activate e2fgvi
python scripts/download_e2fgvi.py
# If download fails, manually place E2FGVI-HQ-CVPR22.pth under third_party/E2FGVI/release_model/
```

Alternatively use gdown directly:
```bash
mkdir -p third_party/E2FGVI/release_model
cd third_party/E2FGVI/release_model
~/.local/bin/gdown --fuzzy "https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing"
cd ../../../
```

## 8. Run deep inpainting
```bash
conda activate e2fgvi
python scripts/run_deep_inpaint.py \
  --frames frames \
  --overlays overlays.json \
  --output-video cleaned_deep.mp4 \
  --context 6 \
  --python $(which python) \
  --extra --deep-weights third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth \
  --extra --deep-weights-arg=--ckpt
```

### Memory tips
- Reduce `--context` (e.g., `--context 3`) if the job exits with code -9 (OOM).
- Split `overlays.json` into smaller ranges for very long clips.
- For smaller output, you can pass `--deep-extra-arg=--set_size --deep-extra-arg=--width --deep-extra-arg=1280 --deep-extra-arg=--height --deep-extra-arg=720`.

## 9. Alternative backends (optional)
Use the base Python environment if you want to run other methods locally:
```bash
python main.py inpaint frames overlays.json --output-video cleaned_temporal.mp4
python main.py inpaint frames overlays.json --method flow --output-video cleaned_flow.mp4
python main.py inpaint frames overlays.json --method delogo --output-video cleaned_delogo.mp4
```

## 10. Copy results and clean up
- Download `cleaned_deep.mp4` back to your workstation (`gcloud compute scp` or `gsutil cp`).
- Delete the VM when finished to stop charges:
  ```bash
  gcloud compute instances delete <instance-name> --zone=<zone> --delete-disks=all --quiet
  ```

Happy editing!
