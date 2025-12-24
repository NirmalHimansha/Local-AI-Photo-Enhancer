# 🎨 AI Photo Enhancer

A powerful command-line tool for enhancing photos using AI, with special focus on natural portrait enhancement. Similar to Remini app but runs **locally on your computer** with full privacy.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/ai-photo-enhancer/graphs/commit-activity)


---

## ✨ Features

- 🤖 **AI-Powered Enhancement** - Uses Real-ESRGAN for professional photo upscaling
- 👤 **Smart Face Detection** - Automatically detects and enhances faces naturally
- 📁 **Batch Processing** - Process entire folders of photos at once
- 🎯 **Natural Results** - No over-sharpening or artificial-looking artifacts
- 🎨 **Multiple Modes** - Auto, face, photo, and anime enhancement modes
- ⚡ **GPU Acceleration** - Faster processing with NVIDIA GPUs (optional)
- 🔒 **Privacy First** - Everything runs locally, no data sent to cloud
- 🆓 **Completely Free** - No subscriptions, no watermarks

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Enhance a single photo
python photo_enhancer.py input.jpg

# Enhance all photos in a folder
python photo_enhancer.py ./input_folder -b ./output_folder
```

That's it! 🎉

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Examples](#-examples)
- [Options](#-options)
- [Requirements](#-requirements)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 💾 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 4GB RAM (8GB recommended)
- 5GB free disk space

### Step 1: Clone Repository

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python torch torchvision numpy pillow
```

### Step 3: Verify Installation

```bash
python photo_enhancer.py --help
```

If you see the help message, you're ready to go! ✓

---

## 📖 Usage

### Single Image Enhancement

```bash
# Basic enhancement (auto-detects faces)
python photo_enhancer.py photo.jpg

# Specify output name
python photo_enhancer.py input.jpg -o output.jpg

# Face enhancement mode
python photo_enhancer.py portrait.jpg -m face

# 2x upscaling (faster)
python photo_enhancer.py photo.jpg -s 2
```

### Batch Processing

```bash
# Enhance all images in a folder
python photo_enhancer.py ./input_folder -b ./output_folder

# Batch with specific mode
python photo_enhancer.py ./portraits -b ./enhanced -m face

# Fast batch processing
python photo_enhancer.py ./photos -b ./output -s 2
```

---

## 🎯 Examples

### Example 1: Portrait Enhancement

```bash
python photo_enhancer.py portrait.jpg -m face -s 4
```

**Result:** Natural-looking face enhancement with 4x resolution increase.

### Example 2: Batch Family Photos

```bash
python photo_enhancer.py ./family_photos -b ./enhanced_photos
```

**Result:** All photos enhanced automatically, faces detected and enhanced.

### Example 3: Landscape Photos

```bash
python photo_enhancer.py landscape.jpg -m photo -s 2
```

**Result:** Clean 2x upscaling without face processing.

### Example 4: Old Photo Restoration

```bash
python photo_enhancer.py old_photo.jpg -m face -s 4 -o restored.jpg
```

**Result:** Restored old portrait with enhanced faces.

### Example 5: Quick Preview

```bash
python photo_enhancer.py test.jpg -s 1 -m face
```

**Result:** Face enhancement only, no upscaling (fast preview).

---

## ⚙️ Options

### Command Line Arguments

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `input` | path | - | Input file or folder path (required) |
| `-b`, `--batch` | path | - | Output folder for batch processing |
| `-o`, `--output` | path | auto | Output file path (single mode) |
| `-m`, `--mode` | auto/face/photo/anime | auto | Enhancement mode |
| `-s`, `--scale` | 1/2/4 | 4 | Upscaling factor |
| `--sharpen` | flag | off | Add subtle sharpening |
| `--help` | flag | - | Show help message |

### Enhancement Modes

| Mode | Best For | Description |
|------|----------|-------------|
| `auto` | Mixed content | Auto-detects and enhances faces |
| `face` | Portraits | Forces face enhancement |
| `photo` | Landscapes | General photo enhancement |
| `anime` | Illustrations | Optimized for anime/drawings |

### Scaling Options

| Scale | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `1` | ⚡ Fastest | Good | Just enhance, no upscaling |
| `2` | ⚡ Fast | Great | Quick 2x upscaling |
| `4` | 🐢 Slower | Best | Maximum quality 4x upscaling |

---

## 📦 Requirements

### Python Packages

```text
opencv-python>=4.5.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pillow>=9.0.0
```

Install all with:
```bash
pip install -r requirements.txt
```

### System Requirements

**Minimum:**
- Python 3.7+
- 4GB RAM
- 2GB free disk space

**Recommended:**
- Python 3.8+
- 8GB RAM
- NVIDIA GPU with CUDA support
- 5GB free disk space

### First Run

On first run, the tool will download AI models (~250MB). This is a one-time download.

---

## 🎨 Supported Formats

**Input:** JPG, JPEG, PNG, BMP, TIFF, WebP  
**Output:** Same as input format

---

## ⚡ Performance

### Processing Speed

| Hardware | Scale | Time per Image |
|----------|-------|----------------|
| CPU (Intel i5) | 2x | ~30 seconds |
| CPU (Intel i5) | 4x | ~60 seconds |
| GPU (NVIDIA RTX 3060) | 2x | ~5 seconds |
| GPU (NVIDIA RTX 3060) | 4x | ~15 seconds |

### Memory Usage

| Scale | RAM Usage | VRAM Usage (GPU) |
|-------|-----------|------------------|
| 1x | 1-2GB | 1-2GB |
| 2x | 2-3GB | 2-3GB |
| 4x | 3-4GB | 3-4GB |

---

## 🔧 Troubleshooting

### Common Issues

#### "Command not found"

```bash
# Try with 'python' instead of 'python3'
python photo_enhancer.py photo.jpg
```

#### Package installation fails

```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Out of memory

```bash
# Use 2x scaling instead of 4x
python photo_enhancer.py photo.jpg -s 2
```

#### Slow processing

```bash
# Use lower scaling for faster results
python photo_enhancer.py photo.jpg -s 2
```

### Getting Help

If you encounter issues:

1. Search for similar problems
2. Create a new issue with:
   - Error message
   - Command you ran
   - System information (OS, Python version)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📖 Improve documentation
- 🔧 Submit pull requests

### Code Style

- Follow PEP 8 guidelines
- Add comments for complex logic
- Update documentation for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project uses:
- **Real-ESRGAN** - BSD 3-Clause License
- **PyTorch** - BSD-style License
- **OpenCV** - Apache 2.0 License

---

## 🙏 Acknowledgments

### AI Models

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao et al. - State-of-the-art image super-resolution

### Libraries

- [PyTorch](https://pytorch.org/) - Neural network framework
- [OpenCV](https://opencv.org/) - Image processing
- [Pillow](https://python-pillow.org/) - Image I/O
- [NumPy](https://numpy.org/) - Numerical computing

### Inspiration

Inspired by apps like Remini, but designed to run locally with full privacy and no costs.

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/ai-photo-enhancer?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/ai-photo-enhancer?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/ai-photo-enhancer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/ai-photo-enhancer)

---

## 🗺️ Roadmap

- [ ] Add GUI interface
- [ ] Support for video enhancement
- [ ] More AI models (GFPGAN for faces)
- [ ] Batch processing progress bar
- [ ] Custom model training
- [ ] Docker container support
- [ ] Web interface

---

## ⭐ Show Your Support

If this project helped you, please consider:

- ⭐ Starring the repository
- 🐛 Reporting bugs
- 💡 Suggesting features
- 📢 Sharing with others

---

## 📝 Changelog

### Version 1.0.0 (2024)

- ✨ Initial release
- 🤖 AI-powered enhancement with Real-ESRGAN
- 👤 Face detection and enhancement
- 📁 Batch processing support
- ⚡ GPU acceleration
- 🎨 Multiple enhancement modes

---

<div align="center">

**Made with ❤️ by [Your Name](https://github.com/nirmalhimansha)**

If you found this project helpful, please consider giving it a ⭐!

</div>
