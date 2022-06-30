# Video Classification Using Deep Learning
## Deep Learning course MSc in AI
### Trained using a small subset, in this case 4 classes, of UCF 101 Action Recognition Dataset.

#### How to run

1. Create a virtual enviroment
2. Install --> requirements.txt
3. run ./src/main.py

ALERT: Only for cuda-enable devices! 

You can selected the desired classes of UCF 101 dataset by changing the CONFIGURATION dictionary in main.py. It is essential that you should monitor the models if you change classes. 

### TO-DO:
1. Optimize the pre-trained CNN using backpropagation (not freezed).
2. Use Attention Mechanism.
3. Use Multimodal Data (e.g. Video and Audio) and develop a model for that purpose.
4. Train using all the 101 classes of UCF or change task???
