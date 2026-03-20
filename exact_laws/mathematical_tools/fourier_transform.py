import scipy.fft as scp
import numpy as np

def fft(tab,way='scipy'): 
    if 'scipy' in way:
        return scp.rfftn(tab)
    elif 'numpy' in way:
        return np.fft.rfftn(tab)
    
def ifft(tab,way='scipy'): 
    if 'scipy' in way:
        return scp.irfftn(tab)
    elif 'numpy' in way:
        return np.fft.irfftn(tab)

