# Workflow Templates

This directory stores workflow templates for various services.

## ComfyUI Workflows (`comfyui/`)

Store your ComfyUI workflow JSON files here for version control and sharing.

### IMPORTANT: Save vs Export

ComfyUI has two ways to save workflows:

1. **"Save" button** = Saves to browser local storage (NOT a file)
   - Quick saves while working
   - Not visible in filesystem
   - Lost if you clear browser data

2. **"Export" button** (or drag workflow to desktop) = Saves as JSON file
   - Use this to save workflows to `workflows/comfyui/`
   - Can be version controlled with git
   - Shareable across machines

### How to export workflows:

1. **Create your workflow in ComfyUI**
2. **Drag the workflow from the UI to your desktop** OR right-click → Export
3. **Save the JSON file to `workflows/comfyui/`** on your local machine
4. **Commit to git** to share with others

### How to load workflows:

**Option A: Load from browser**
- Click "Load" button in ComfyUI
- Browse your previously saved workflows (from browser storage)

**Option B: Load from file**
- Drag a JSON file from your desktop into ComfyUI
- Or use Load → navigate to `/app/workflows/comfyui/`

### Workflow naming convention:
- `txt2img_basic.json` - Basic text-to-image generation
- `lora_training.json` - LoRA training workflow
- `style_transfer.json` - Style transfer workflow
- etc.
