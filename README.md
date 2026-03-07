# Torch-Lenia: Continuous Cellular Automata Engine

A high-performance, GPU-accelerated implementation of **Lenia** (Continuous Cellular Automata) built entirely from scratch using PyTorch and Python. 

This project explores artificial life, complex systems, and self-organizing patterns by simulating a continuous universe where cell states, space, and time are fluid rather than discrete.

## 🌌 Visual Showcase: The Primordial Soup
The current `main.py` is configured to run a **3-channel (RGB) Primordial Soup**. 
By injecting absolute random noise into a multi-channel environment with specific cross-channel attraction and repulsion weights, the system spontaneously self-organizes into complex, life-like, moving structures.

*(Tip: Add a screenshot or a GIF of your running simulation here!)*

## ✨ Key Features
* **GPU-Accelerated Physics:** Utilizes PyTorch's Fast Fourier Transform (`torch.fft.fft2`) for lightning-fast convolution operations over the entire grid.
* **Multi-Channel (RGB) Support:** Fully supports N-dimensional universes. The current configuration uses 3 interacting channels with a complex weight matrix.
* **Data-Driven Architecture:** Complete decoupling of the physics engine from the organism data. The simulation rules are injected dynamically via a `settings` dictionary.
* **Mathematical Noise Reduction:** Implements an elegant floating-point error thresholding mechanism (`1e-4`) to prevent FFT mathematical noise from accumulating.
* **Real-Time Rendering:** Uses `Pygame` to render the complex tensor states directly to the screen at a smooth 60 FPS.

## 🛠️ Tech Stack
* **PyTorch:** Core engine, tensor operations, FFT convolutions, and GPU (`cuda`) compatibility.
* **NumPy:** Mathematical arrays and structural data definitions.
* **Pygame:** Real-time visual rendering and window management.

## 🚀 How to Run

1. Clone this repository:
    git clone https://github.com/ilaygraedi/Torch-Lenia.git
    cd Torch-Lenia

2. Install the required dependencies:
    pip install torch numpy pygame

3. Run the simulation:
    python main.py

## 🧠 Project Architecture
* `engine.py`: The core physics engine. Contains the `Lenia` class which handles the creation of the kernels, the growth functions, and the update logic.
* `main.py`: The entry point and configuration hub. It defines the DNA (the `settings` dictionary) of the universe and runs the main loop.

## 🔮 Future Implementations
* Support for multi-kernel layers per channel (to allow for hyper-complex creatures).
* Integration of Genetic Algorithms to automate the discovery of new stable creatures and behaviors.

---
*Created as an exploration into software architecture, continuous mathematics, and artificial life.*
