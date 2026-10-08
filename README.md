# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models





## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/RealRebelAI/ComfyUI-GGUF_KREA-2.git
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/RealRebelAI/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF_KREA-2
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF_KREA-2\requirements.txt
```


