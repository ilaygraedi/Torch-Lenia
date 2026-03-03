# 🧬 Torch-Lenia: GPU-Accelerated Artificial Life

A high-performance simulation of Continuous Cellular Automata (inspired by the OpenLenia framework). 
This project models the evolution of virtual, multi-channel (RGB) organisms by calculating complex spatial interactions in real-time.

> **Engineering Note:** Originally built with NumPy on the CPU, the engine was completely rewritten using **PyTorch**. This leveraged GPU acceleration (CUDA) and tensor mathematics, allowing the simulation of millions of pixels at 60 FPS.

![20260303-1621-39 6991350-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/28dd0ddf-43f4-4c21-8345-c16811c56758)

## ✨ Key Features

* **Multi-Channel Interaction:** Supports 3 color channels (RGB) with cross-channel interactions using advanced matrix multiplication (`torch.tensordot`).
* **GPU Acceleration:** Shifts heavy computations from the CPU to the GPU (VRAM) using PyTorch, enabling massive grid sizes (e.g., 800x800) without frame drops.
* **Fast Fourier Transform (FFT):** Utilizes `torch.fft.fft2` to calculate environmental convolutions and organism growth efficiently, replacing traditional nested loops.
* **Custom Biology (DNA):** Configurable parameters (growth rates, interaction weights, radii) to discover new macroscopic cellular behaviors and "gliders".

## 🚀 How to Run

1. **Install Dependencies:**
   Make sure you have a Python environment set up, then run the following in your terminal:
   
   `pip install torch torchvision pygame numpy`

2. **Run the Simulation:**
   
   `python main.py`

3. **Controls:**
   * **Left Click:** Inject biological matter into the ecosystem.
   * *Tip:* Click once in the center and watch the cells interact, form membranes, and evolve!

## 🧠 The Architecture & Challenges

Migrating this engine to PyTorch required solving several architectural challenges:
* **Tensor Broadcasting & Permutation:** Handling dimensional differences between PyTorch tensors `(Channels, Height, Width)` and Pygame's expected screen format `(Width, Height, Channels)` using `.permute()`.
* **Memory Bottlenecks:** Synchronizing GPU data with CPU-bound rendering safely via `.cpu().numpy()`.
* **Data Types:** Forcing strict type matching (`float32` vs `float64`) during FFT operations and weight dot-products.
