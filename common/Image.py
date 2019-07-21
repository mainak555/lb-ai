import numpy as np
import matplotlib.pyplot as plt
#%%
class Image:
    @staticmethod
    def RGB(image, channel='rgb'):
        if channel == 'r' or channel == 'R':
            return image[:, :, 0]
        elif channel == 'g' or channel == 'G':
            return image[:, : 1]
        elif channel == 'b' or channel == 'B':
            return image[:, :, 2]
        else:
            return image[:, :, :]
#################
    @staticmethod
    def Show(images, labels=None, figsize=(15, 15), gridsize=(5, 4),
        channel='rgb', cmap='gray', _getLabelName = lambda x: x):        
        fig = plt.figure(figsize=figsize)
        if len(images.shape) == 4 and images.shape[0] >= gridsize[0] * gridsize[1]:
            for i in range(gridsize[0] * gridsize[1]):
                index = np.random.randint(1, images.shape[0])    
                plt.subplot(gridsize[0], gridsize[1], i+1)
                plt.tight_layout()
                if cmap is not None:
                    plt.imshow(Image.RGB(images[index], channel), cmap=cmap, interpolation=None)
                else:
                    plt.imshow(Image.RGB(images[index], channel), interpolation=None)   #[:,:,0]=height x width x channel (RGB)
                if labels is not None:
                    plt.title('Label[{}]: {} ({})'.format(index, labels[index], _getLabelName(labels[index])))
                plt.xticks([])
                plt.yticks([])
        else:
            if cmap is not None:
                plt.imshow(Image.RGB(images, channel), cmap=cmap, interpolation=None)
            else:
                plt.imshow(Image.RGB(images, channel), interpolation=None)  #[:,:,0]=height x width x channel (RGB)
            if labels is not None:
                plt.title('Label: {} ({})'.format(labels, _getLabelName(labels)))
        plt.show()
#################
    # @staticmethod
    # def ShowByClass(images, labels, figsize=(15, 15), gridsize=(5, 4),
    #         channel='rgb', cmap='gray', _getLabelName = lambda x: x):
        
    #     fig = plt.figure(figsize=figsize)
    #     if len(images.shape) < 4 or images.shape[0] < gridsize[0] * gridsize[1]:
    #         return
        
    #     class_id = 0
    #     class_count = 0
    #     images_by_classes = None
    #     for i in range(gridsize[0] * gridsize[1]):
    #         index = np.random.randint(0, images.shape[0], size=1)
