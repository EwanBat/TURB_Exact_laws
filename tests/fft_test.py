import numpy as np
import matplotlib.pyplot as plt

# Randome 3D array
data = np.sqrt(np.random.rand(100, 100, 100)) * np.exp(1j * 2 * np.pi * np.random.rand(100, 100, 100))

def diagonal_3D(array):
    """Extract the diagonal of a 3D array."""
    return np.array([array[i, i, i] for i in range(min(array.shape))])

# Take the diagonal of the 3D array
diagonal = diagonal_3D(data)

# Perform FFT on the diagonal
fft_result = np.fft.fftn(diagonal)

# Take the TF of the array
fft_3d = np.fft.fftn(data)

# Take the diagonal of the TF
diagonal_fft_3d = diagonal_3D(fft_3d)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.yscale('log')
plt.plot(np.real(diagonal), label='Diagonal of 3D array')
plt.plot(np.real(diagonal_fft_3d), label='FFT of Diagonal')
plt.title('FFT of Diagonal vs Diagonal of 3D Array')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.yscale('log')
plt.plot(np.imag(diagonal), label='Diagonal of 3D array')
plt.plot(np.imag(diagonal_fft_3d), label='FFT of Diagonal')
plt.title('FFT of Diagonal vs Diagonal of 3D Array')
plt.xlabel('Index')
plt.ylabel('Value')

plt.legend()
plt.grid()
plt.show()