# 🔧 Setup Instructions

Install the required packages for audio processing and embedding:

```bash
# Fix soundfile compatibility
!pip uninstall -y soundfile
!pip install soundfile==0.10.3.post1

# Install dependencies
!pip install numpy
!pip install audioread
!pip install PySoundFile
!pip install tensorflow_io

# Reinstall TensorFlow to avoid conflicts
!pip uninstall -y tensorflow tensorflow_io
!pip install tensorflow

# Pretrained model support
!pip install tensorflow_hub

# Audio analysis
!pip install librosa
