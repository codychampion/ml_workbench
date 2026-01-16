# Models Directory

This directory contains model files for training.

## Setup: Copy Your Wan 2.2 Model

Copy your Wan 2.2 model file from ComfyUI to this directory:

### Windows (PowerShell or Command Prompt)
```powershell
# From ml_workbench directory
copy "C:\Users\cody\ComfyUI\models\unet\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" models\unet\

# Or if you're in a different location
copy "path\to\ComfyUI\models\unet\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" "C:\Users\cody\Documents\GitHub\ml_workbench\models\unet\"
```

### Git Bash (Windows)
```bash
# From ml_workbench directory
cp /c/Users/cody/ComfyUI/models/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors models/unet/
```

### Linux/Mac
```bash
# From ml_workbench directory
cp ~/ComfyUI/models/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors models/unet/
```

## Expected File Structure

After copying, you should have:
```
models/
└── unet/
    └── wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors  (~13-15GB)
```

## Verify the File

Check the file exists:
```bash
ls -lh models/unet/*.safetensors
```

You should see something like:
```
-rw-r--r-- 1 user user 14G Jan 16 09:00 wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
```

## Alternative: Use Symlink (Advanced)

Instead of copying, you can create a symbolic link to save disk space:

### Windows (PowerShell as Administrator)
```powershell
New-Item -ItemType SymbolicLink -Path "models\unet\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" -Target "C:\Users\cody\ComfyUI\models\unet\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
```

### Linux/Mac
```bash
ln -s ~/ComfyUI/models/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors models/unet/
```

## Git Ignore

The `.gitignore` file excludes `*.safetensors` from git to avoid committing large model files.

## Model Information

- **Model**: Wan 2.2 Text-to-Video (High Noise Expert)
- **Architecture**: HunyuanVideo DiT (Diffusion Transformer)
- **Parameters**: 14B (active), 27B (total with MoE)
- **Format**: FP8 Scaled Safetensors
- **Size**: ~13-15GB
- **Use**: LoRA training base model

## Troubleshooting

### File Not Found Error
```
RuntimeError: Could not load model from ./models/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
```

**Solution**: Copy the model file to this directory (see above)

### Permission Denied
```
PermissionError: [Errno 13] Permission denied
```

**Solution**: Check file permissions and ensure Docker has access to the file

### Wrong Model Format
If you get errors about incompatible model format, the script will tell you to use HuggingFace instead:
```bash
--model tencent/HunyuanVideo-1.5
```

The trained LoRA will work in ComfyUI regardless of which base model you use for training!
