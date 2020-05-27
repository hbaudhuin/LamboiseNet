# LamboiseNet

*Master Thesis about Change Detection in Satellite Imagery using Deep Learning*

Héloïse BAUDHUIN - Antoine LAMBOT

### Requirements :

##### Dataset:
You need all the **Earth_*** (1-32) instances from the dataset.
They need to be in the DATA folder.

Link to the dataset : https://drive.google.com/drive/folders/1rd1vseWiFSqQc5-93XSRQW9Bzzcgqc6H?usp=sharing 

##### Weights and metrics:
If you want to run our already trained **Light UNet++** with the -reload argument, you need the following files :
- Weights/last.pth
- Loss/last.pth
- Loss/last_metrics.pth

Links to the files :

https://drive.google.com/drive/folders/1qbZm-b4gdhzzMCP09XwWx2wJKxsSXBJL?usp=sharing
https://drive.google.com/drive/folders/1-DdCZxCv7OInvpUnbbT-4p2Uhc_v6ztI?usp=sharing

##### Librairies:
- PyTorch (1.3.1+)
- numpy
- scikit-learn
- matplotlib
- imageio
- Pillow
- imgaug
- tqdm

### Usage :
```
python3 train.py 
                [-h]
                [--epochs EPOCHS] 
                [--learning_rate LEARNING_RATE]
                [--n_data_augm N_DATA_AUGM] 
                [-reload] 
                [-save]
                
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs the model will run (1 by default)
  --learning_rate LEARNING_RATE
                        starting learning rate (0.001 by default)
  --n_data_augm N_DATA_AUGM
                        number of data augmentation instances to generate per
                        original instance (2 by default)
  -reload               reload the model weights and metrics from the last run
                        (disabled by default)
  -save                 save the weights and metrics of the model when it has
                        finished running (disabled by default)
```

