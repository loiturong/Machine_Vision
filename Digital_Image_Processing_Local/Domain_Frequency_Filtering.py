import numpy as np
import cv2 as cv

__all__ = []

# region Chapter 4 function
def discrete_fourier_transform(image: np.ndarray) -> list[np.ndarray]:
    _M, _N = image.shape

    _FR = np.zeros((_M, _N))
    _FI = np.zeros((_M, _N))

    _x = np.array([[list(range(0, _M))] * _N])[0].T
    _y = np.array([[list(range(0, _N))] * _M])[0]

    for u in range(_M):
        for v in range(_N):
            _FR[u, v] = np.sum(image * np.cos(2 * np.pi * (1.0 * u * _x / _M + 1.0 * v * _y / _N)))
            _FI[u, v] = np.sum(image * np.sin(2 * np.pi * (1.0 * u * _x / _M + 1.0 * v * _y / _N)))

    # cut smol value
    _FR = np.where(np.abs(_FR) < 1e-10, 0, _FR)
    _FI = np.where(np.abs(_FI) < 1e-10, 0, _FI)

    return [_FR, _FI]

# endregion

# region Testing
if __name__ == '__main__':
    f = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]], dtype=np.float32)
    M, N = f.shape

    # step for filtering in frequency domain

    # 1. obtain padding size
    P = 2 * M
    Q = 2 * N

    # 2. replicate padding
    padded_image = np.zeros((P, Q))
    padded_image[:M, :N] = f

    # 3. center the Fourier transform
    x = np.array([[list(range(0, M))] * N])[0].T
    y = np.array([[list(range(0, N))] * M])[0]
    padded_image[:M, :N] = f * (-1) ** (x + y)

    # 4. compute DFT
    Fourier_domain = discrete_fourier_transform(padded_image)[0]

    # 5. Construct a real, symmetric filter transfer function

# endregion