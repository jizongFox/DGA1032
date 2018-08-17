import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = '3channels_train_lr_0.005000.csv'
paired_filenames = [filename, filename.replace('train','val')] if filename.find('train')>=0 else [ filename.replace('val','train'),filename]

train_curves, val_curves = pd.read_csv(paired_filenames[0],index_col=False).iloc[:,1:], pd.read_csv(paired_filenames[1],index_col=False).iloc[:,1:]


def visual(term):

    fig = plt.figure(figsize=(15,7))
    ax1= fig.add_subplot(131)
    ax1.plot(train_curves['foregound'],'b', label = 'train')
    ax1.plot(val_curves['foregound'],'g', label = 'val')
    ax1.hlines(y=train_curves['foregound'].max(),xmin=0,xmax=len(train_curves['foregound']),colors='b',linestyles='dashed')
    ax1.text(train_curves['foregound'].values.argmax(),train_curves['foregound'].max(),'%.3f'%train_curves['foregound'].max(),)
    ax1.hlines(y=val_curves['foregound'].max(),xmin=0,xmax=len(val_curves['foregound']),colors='g',linestyles='dashed')
    ax1.text(val_curves['foregound'].values.argmax(),val_curves['foregound'].max(),'%.3f'%val_curves['foregound'].max(),)
    ax1.legend()
    ax1.set_ylim([0.7,1])

    ax2= fig.add_subplot(132)
    ax2.plot(train_curves['background'],'b', label = 'train')
    ax2.plot(val_curves['background'],'g', label = 'val')
    ax2.hlines(y=train_curves['background'].max(),xmin=0,xmax=len(train_curves['background']),colors='b',linestyles='dashed')
    ax2.text(train_curves['background'].values.argmax(),train_curves['background'].max(),'%.3f'%train_curves['background'].max(),)
    ax2.hlines(y=val_curves['background'].max(),xmin=0,xmax=len(val_curves['background']),colors='g',linestyles='dashed')
    ax2.text(val_curves['background'].values.argmax(),val_curves['background'].max(),'%.3f'%val_curves['background'].max(),)
    ax2.legend()
    ax2.set_ylim([0.7,1.05])


    ax3 = fig.add_subplot(133)
    ax3.plot((train_curves['background']+train_curves['foregound'])/2,'b', label = 'train')
    ax3.plot((val_curves['background']+val_curves['foregound'])/2,'g', label = 'val')
    ax3.hlines(y=((train_curves['background']+train_curves['foregound'])/2).max(),xmin=0,xmax=len(train_curves['background']),colors='b',linestyles='dashed' )
    ax3.text((train_curves['background']/2+train_curves['foregound']/2).values.argmax(), ((train_curves['background']+train_curves['foregound'])/2).max(), s='%.3f'%((train_curves['background']+train_curves['foregound'])/2).max() )

    ax3.hlines(y=((val_curves['background']+val_curves['foregound'])/2).max(),xmin=0,xmax=len(val_curves['background']),colors='b',linestyles='dashed' )
    ax3.text((val_curves['background']/2+val_curves['foregound']/2).values.argmax(), ((val_curves['background']+val_curves['foregound'])/2).max(), s='%.3f'%((val_curves['background']+val_curves['foregound'])/2).max() )

    ax3.legend()
    ax3.set_ylim([0.7,1])

    plt.show()


visual('mean')
