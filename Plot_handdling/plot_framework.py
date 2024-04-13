import matplotlib.pyplot as plt
import matplotlib

class PlotFrameWork:
    def __init__(self):
        self.instances=[]
        pass
    
    """_summary_
        Clean display of training loss curve
    """
    def plot_loss(self, losses,title='Mean Squared Error', label='Training Loss' , xlabel='Epochs', ylabel='Loss'  ):

        plt.plot(losses, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    
    
    """_summary_
        Clean display of training and validation losses curves
    """    
    def plot_train_val_loss(self, train_losses, test_losses ,title='Mean Squared Error' ):

        # plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label='Training Loss', lw=10)
        plt.plot(test_losses, label='Test Loss')
        plt.title(title)
        
        # Take first this part static
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        
    """_summary_
        Display of training losses curves for the same model but with differents parameters...
    """  
    def plot_multi_losses(self):
        
        number_plot = len(self.instances)
        # Assuming we have two lines
        num_cols = number_plot // 2 if number_plot % 2 == 0 else number_plot // 2 + 1
        fig, axes = plt.subplots(2, num_cols , figsize=(15, 8))
        fig.suptitle('Training Loss')
        for i, instance in enumerate(self.instances):

            row = i // num_cols
            col = i % num_cols

            # print("row, col", row, col)
            axes[row, col].plot(instance.train_losses)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].set_title('fig_{} : lr = {} , n_epoch = {} '.format( i+1, instance.lr, instance.n_epochs), fontstyle='oblique')
        
        plt.tight_layout()
        plt.show()
        
        
