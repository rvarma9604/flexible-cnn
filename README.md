# flexible-cnn
In this project I have tried to create a flexible version of convolutional neural network using PyTorch, where all that one needs to do is just plug-in the required parameters for the model and have the model ready without bothering about the underlying implementation details.

# Files
- main.py - this is where you modify the network parameters.
- models.py - PyTorch implementation for the CNN and Data class.
- helper.py - contains various utility functions that deal with training the model and plotting the performance curves (makes use of GPU if available).

# Experiment
### Configuration
- Modify the network parameter in main.py according to your requirements.

### Run 
```bash
python main.py
```
