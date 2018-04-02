import pandas as pd
import matplotlib.pyplot as plt
import os
import errno


def photo_summary_df(filepath, captured_col, hist=False, figpath=False):
    '''DOCSTRING
       This takes a filepath, throws it into a pd df and gets counts of the col
       given to the function as an argument. it also takes two optional
       arguments, hist and figpath. hist will turn the histogram producing
       function on or off, figpath will save aforementioned hist (after safely
       attempting to make the folder given in figpath).
       ----------
       INPUTS
       filepath : the file to the directory
       captured_col : the column that makes a histogram / dictionary of counts
       hist : if you want a histogram print out or not
       figpath : if you want to save the file, and where to save it
       ----------
       RETURNS
       counts: dictionary of the counts of the unique data points in a column
    '''
    df = pd.read_csv(filepath)
    df.head()
    captured = pd.unique(df[captured_col])
    print(captured)
    counts = df[captured_col].value_counts().to_dict()
    if hist:
        plt = make_hist_pd(filepath, captured_col, figpath)
        plt.show()
    return counts


def make_hist_pd(filepath, captured_col, figpath=False):
    '''DOCSTRING
       makes a histogram from a pd df and col. optionally save the histogram
       to a safely made folder
       ----------
       INPUTS
       filepath : the file to the directory
       captured_col : the column that makes a histogram / dictionary of counts
       hist : if you want a histogram print out or not
       figpath : if you want to save the file, and where to save it
       ----------
       RETURNS
       plt of the histgram
    '''
    plt.figure(figsize=(20, 10))
    df[captured_col].hist(xrot=90, bins=len(captured))
    plt.title('Number of images by animal')
    plt.tight_layout()
    if figpath:
        figpath = '/'+figpath if figpath[0] != '/' else figpath
        try:
            os.makedirs('/'.join(figpath.split('/')[:-1])+'/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        plt.savefig(figpath)
    return plt


if __name__ == "__main__":
    plt.close('all')
    kb_df = photo_summary_df('data/kb_photos.csv', 'CommonName',
                             hist=True, figpath='plots/kb_df.png')
    lynx = photo_summary_df('data/linx_data.csv', 'Species',
                            hist=True, figpath='plots/linx_hist.png')
