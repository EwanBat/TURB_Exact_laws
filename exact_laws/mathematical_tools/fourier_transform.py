import scipy.fft as scp
import numpy as np

def fft(tab,way='scipy', traj=False): 
    if traj:
        return scp.rfft
    if 'scipy' in way:
        return scp.rfftn
    elif 'numpy' in way:
        return np.fft.rfftn
    
def ifft(tab,way='scipy', traj=False):
    if traj:
        return scp.irfft
    if 'scipy' in way:
        return scp.irfftn
    elif 'numpy' in way:
        return np.fft.irfftn

