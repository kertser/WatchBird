# Installing CUDA-Enabled OpenCV for WatchBird

## Current Status
The standard `opencv-python` package from PyPI does **NOT** include CUDA support.
Your current setup uses **DirectML for face embedding** (works with NVIDIA/AMD/Intel GPUs via ONNX Runtime).

## Prerequisites for CUDA OpenCV

Before installing CUDA OpenCV wheels, you **MUST** have:

1. **NVIDIA GPU with CUDA support**
2. **CUDA Toolkit 12.x** installed: https://developer.nvidia.com/cuda-downloads
3. **cuDNN 8.x or 9.x** installed (REQUIRED!): https://developer.nvidia.com/cudnn-downloads

### Installing cuDNN (Required!)

1. Download cuDNN from: https://developer.nvidia.com/cudnn-downloads
   - Select: Windows, x86_64, CUDA 12.x, **Local Installer (Zip)**
2. Extract the zip file
3. Copy the contents to your CUDA installation:
   ```powershell
   # Run as Administrator
   $cudnn = "C:\path\to\extracted\cudnn-windows-x86_64-9.x.x.x_cuda12-archive"
   $cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
   
   Copy-Item "$cudnn\bin\*.dll" "$cuda\bin\" -Force
   Copy-Item "$cudnn\include\*.h" "$cuda\include\" -Force
   Copy-Item "$cudnn\lib\x64\*.lib" "$cuda\lib\x64\" -Force
   ```
4. Verify: `dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudnn*.dll"`

## Option 1: Use Pre-built CUDA Wheels

### A. From cudawarped (GitHub releases)
1. Visit: https://github.com/cudawarped/opencv-python-cuda-wheels/releases
2. Download the `.whl` file for Windows
3. Uninstall existing OpenCV and install:
   ```powershell
   uv pip uninstall opencv-python opencv-python-headless opencv-contrib-python
   uv pip install path/to/opencv_contrib_python-X.X.X-cp37-abi3-win_amd64.whl
   ```

### B. From Anaconda (alternative)
```powershell
# Create a conda environment instead of venv
conda create -n watchbird python=3.12
conda activate watchbird
conda install -c conda-forge opencv=*=*cuda*
```

## Option 2: Build from Source with CLion (30-60 min)

### Prerequisites:
1. **CLion** with bundled CMake
2. **Visual Studio 2022** (Community/Pro/Enterprise) with **"Desktop development with C++"** workload
3. **CUDA Toolkit 12.x**: https://developer.nvidia.com/cuda-downloads
4. **cuDNN 9.x**: Already installed at `C:\Program Files\NVIDIA\CUDNN\v9.18`

### Step 1: Clone OpenCV repositories
```powershell
cd F:\Projects
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```

### Step 2: Configure CLion to use Visual Studio Toolchain

**IMPORTANT:** CUDA requires Visual Studio compiler (MSVC), not MinGW/GCC!

1. Open CLion
2. Go to **File → Settings → Build, Execution, Deployment → Toolchains**
3. Click **+** to add a new toolchain
4. Select **Visual Studio**
5. Set:
   - **Name**: `Visual Studio 2022`
   - **Architecture**: `x64` (also called amd64 - this is 64-bit, required for CUDA)
   - CLion should auto-detect the Visual Studio paths
6. **Move it to the top** of the list (make it default) or ensure it's selected for this project
7. Click **Apply**

### Step 3: Open OpenCV in CLion
1. **File → Open** → Select `F:\Projects\opencv` folder
2. CLion will detect CMakeLists.txt automatically

### Step 4: Configure CMake in CLion
1. Go to **File → Settings → Build, Execution, Deployment → CMake**
2. Set **Toolchain** to **Visual Studio 2022**
3. Set **Generator** to **Visual Studio 17 2022** (or leave as default with VS toolchain)
4. In the **CMake options** field, paste:

```
-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=F:/Projects/opencv_cuda -DOPENCV_EXTRA_MODULES_PATH=F:/Projects/opencv_contrib/modules -DWITH_CUDA=ON -DCUDA_ARCH_BIN=7.5 -DWITH_CUDNN=ON -DCUDNN_LIBRARY="C:/Program Files/NVIDIA/CUDNN/v9.18/lib/12.9/x64/cudnn.lib" -DCUDNN_INCLUDE_DIR="C:/Program Files/NVIDIA/CUDNN/v9.18/include/12.9" -DOPENCV_DNN_CUDA=ON -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON -DBUILD_opencv_python3=ON -DPYTHON3_EXECUTABLE=F:/Projects/WatchBird/.venv/Scripts/python.exe -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF
```

5. Set **Build type** to `Release`
6. Click **Apply** then **OK**

> **Note:** Your GPU is **Quadro RTX 5000** with compute capability **7.5**

### Step 5: Build
1. Wait for CMake to finish configuring (check the CMake tool window)
2. **Build → Build Project** (or Ctrl+F9)
3. This takes 30-60 minutes depending on your CPU

### Step 6: Install
1. In CLion's Terminal, run:
```powershell
cmake --build cmake-build-release --target install
```

### Step 7: Copy to WatchBird venv
```powershell
# First uninstall existing opencv
cd F:\Projects\WatchBird
uv pip uninstall opencv-python opencv-python-headless opencv-contrib-python

# Copy the built cv2 module
$src = "F:\Projects\opencv_cuda\python\cv2"
$dst = "F:\Projects\WatchBird\.venv\Lib\site-packages\cv2"
Remove-Item $dst -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item $src $dst -Recurse

# Also need numpy
uv pip install numpy
```

### Step 8: Verify
```powershell
cd F:\Projects\WatchBird
python -c "import cv2; print('OpenCV:', cv2.__version__); print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```
Should print `CUDA: 1` (or more)

## Option 3: Keep Current Setup (Recommended for Now)

Your current setup already uses GPU for the **most expensive operation** (face embedding via ArcFace):
- **Face Embedding**: DirectML GPU ✅ (this is 80% of compute time)
- **Face Detection**: CPU (YuNet is very lightweight)

The performance gain from CUDA OpenCV would be marginal since YuNet is already fast on CPU.

### Current Performance:
- 13-14 FPS with DirectML face embedding (ArcFace R100)
- This is acceptable for real-time face recognition

### If you need more FPS:
1. Reduce resolution: `[640, 480]` in config.yaml
2. Increase `embedding_sample_interval` to 5 (embed every 5th frame)
3. Use MobileFaceNet instead of ArcFace (smaller, faster model)

## Verify CUDA Installation

After installing CUDA OpenCV, test with:
```python
import cv2
print("OpenCV version:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
# Should print 1 or more if CUDA is working
```

