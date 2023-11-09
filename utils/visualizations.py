# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Function to visualize images
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  

# Function to plot metrics given from training
def plot_metrics(train_loss,train_acc,val_loss, val_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle('Performance on training')
    ax1.plot(train_loss, color='blue')
    ax1.plot(val_loss, color='red')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax2.plot(train_acc, color='blue')
    ax2.plot(val_acc, color='red')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax1.legend(['Train', 'Val'])
    ax2.set_ylim(0, 1)
    plt.show()

