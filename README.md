
# ML-Fashion-Example aka What I was hoping for @CIMData :)

This script downloads a dataset of 10k small images of clothing / fashion products and tries to classify each one of them, i.e. guess if it is a bag, pullover, shoe, etc...  

For every run, it computes the accuracy (how many images were classified correctly) and tries to adjust the model to improve for the next run.

## Installation

### CUDA / GPU
- CUDA-Support by GPU : https://dewiki.de/Lexikon/CUDA  
- CUDA Toolkit Downloads: https://developer.nvidia.com/cuda-toolkit-archive

I'm using NVIDIA GTX 1080, which has a max support of CudaToolkit v11.3.
Install the Cuda-Toolkit that's compatible with your GPU (if at all).
With no CUDA-support it will fallback to CPU, so if you don't want to hastle with CUDA installation, don't worry and skip right to the CPU section.

Use the Anaconda Command Prompt.

```
conda create --name ml Python=3.8.10
conda activate ml
conda config --set solver classic # was needed for me as i was trying out libmamba, which did not work as expected. You may ignore this line if you want
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

### CPU

If you can't use CUDA / your GPU, just use the CPU instead and install the latest packages

```
conda install pytorch torchvision torchaudio -c pytorch
```

## Running the script

From the Anaconda Command Prompt, change directory to the one with the source files and start VSCode from there.
cd <path>
code .From VSCode Terminal, run:

```
cd <path>
code .
````

You can use the VSCode terminal from that point on or continue using the Anaconda prompt

```
python ./ml.py
```

- This will download the datasets automatically into the `data`-folder.
- Then it will train the model over and over again.
- You can change the amount of maximmum training rounds by adjusting the `maxEpoch` variable
- It will stop automatically once it couldn't improve accuracy compared to the previous run
- The trained model will be saved to `model.pth`, so you don't have to do it everytime you run the program. Delete this file if you want to re-train.
- It will try to classify each of the 10k images in the dataset

Max. accuracy I could reach was about 80%

