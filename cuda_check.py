import torch

if torch.cuda.is_available():
    print("Congratulations, CUDA support is active")
elif torch.backends.mps.is_available():
    print("Congratulations! Looks like you're using a Mac device with MPS support (Metal Performance Shaders). Have fun!")
else:
    print("Nah, sorry mate! Looks like your CPU has got to do all the heavy lifting.")