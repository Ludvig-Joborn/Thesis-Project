# Thesis-Project
This is repository for Mattias & Ludvig's master thesis work. It's pretty cool, if you ask me.

```
Python version 3.8.x was used in this project.
```

## Project Setup:

```bash
# Clone the project
git clone git@github.com:Ludvig-Joborn/Thesis-Project.git

# Change to project directory
cd Thesis-Project

# Create a virutal environment
python -m venv venv

# Activate the environment (bash)
source venv/bin/activate # may be 'venv/Scripts/activate'
# Activate the environment (powershell)
./venv/Scripts/Activate.ps1

# Install required packages
pip3 install -r requirements.txt
```
## Test Code
Run the test code below to see if the installation was successful:
```Python
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

x = torch.rand(5, 3)
print("Randomized tensor test:")
print(x, "\n")

print("torch version:", torch.__version__)
print("torchaudio version:", torchaudio.__version__)
print("CUDA available:", torch.cuda.is_available())
```

If this error shows for torchaudio: 
`UserWarning: No audio backend is available.` \
Execute the following:

SoundFile for Windows:
```bash
pip install PySoundFile
```
Sox for Linux:
```bash
pip install sox
```
