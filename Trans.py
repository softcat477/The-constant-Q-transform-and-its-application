import numpy as np
from scipy.sparse import bsr_matrix
import scipy
import timeit
import tqdm
from tabulate import tabulate
x = np.random.random(1024)
mk = 0

def DFT_slow(x):
    #global mk
    # https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    nn = np.arange(N)
    k = nn.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * nn / N)
    print (np.allclose(M[0,:], M[20,:]))
    return np.dot(M, x)

def FFT(x, off=""):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x, off)
    else:
        X_even = FFT(x[::2], off+"\t")
        X_odd = FFT(x[1::2], off+"\t")
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
def mcqt(x, sr, table, hop_size=512, window_type=None):
    """
    table: [k, freq, window_sample, window_ms, Q]
    Outputs:
        y : ndarray of shape (x_length/hop_size, #k)
        freqs : list of length #k, [hz of k=0, hz of k=1, ...]
    """
    
    y = np.zeros((len(table), np.int(np.floor(x.shape[0]/hop_size))), dtype=complex)
    freq_axis = []
    
    # Sliding(?) CQT
    for t_ptr in range(y.shape[1]): # Frame index
        x_offset = t_ptr * hop_size
        for k_ptr in range(len(table)):
            N_k = table[k_ptr][2]
            Q = table[k_ptr][-1]
            
            # Get window of length N_k
            if type(window_type) == type(None):
                window = np.ones(N_k) # Rect if not specified
            elif type(window_type) == type("str"):
                window = scipy.signal.hamming(N_k) # Else, use hamming window
           
            # Pad _x if needed
            if x_offset-N_k//2 < 0: # Left boundary
                _x = x[0:x_offset+N_k//2]
                _x = np.concatenate((np.zeros(N_k//2-(_x.shape[0]-N_k//2)), _x), axis=0)
            else:
                _x = x[x_offset-N_k//2 : x_offset+N_k//2]

            if _x.shape[0] < N_k: # Right boundary
                _x = np.concatenate((_x, np.zeros(N_k-_x.shape[0])), axis=0)
            
            # dft with varied length
            _x = _x*window
            _x = _x.reshape(-1, 1)
            n = np.arange(N_k)
            M = np.exp(-1j*2*np.pi*Q*n/N_k).reshape(1, -1)
            tmp = np.dot(M, _x)
                
            # Assign
            y[k_ptr, t_ptr] = tmp[0][0]/N_k
            
    # Frequency axis
    freqs = [table[i][1] for i in range(len(table))]
    return y, freqs
            
def table2(sr, NFFT=0, verbose=False, base_Q =34, fix_B=-1, fmin=175.0):
    """
    Plot table 2
    """
    kmin =  0
    #S = 32000 # Paper
    S = sr # Mine
    table = []
    f, Q = 0.0, base_Q
    isChange = False
    
    i = 0
    while True:
        if f >= 1568.0:
            Q =base_Q*2
        else:
            Q =base_Q*1
           
        """
        if Q==base_Q*2 and isChange==False:
            fmin = f
            kmin = i-1
        """
            
        if base_Q == 17:
            f = 2**(1*(i-kmin)/12)*fmin
        elif base_Q == 34:
            f = 2**(1*(i-kmin)/24)*fmin
            
        if fix_B==12 or fix_B==24:
            f = 2**(1*(i-kmin)/fix_B)*fmin
            
        if f > sr/2:
            break
            
        N = (S/f)*Q
        if NFFT != 0 and round(N) > NFFT:
            print ("k={} Window size(sample) {} > Nfft size(sample) {}".format(i, round(N), NFFT))
        table.append([i, round(f, 2), round(N), round(N/S*1000), Q])
        i += 1
    if verbose:
        print ("SR : {}; NFFT : {}".format(S, NFFT))
        print(tabulate(table, headers=["Channel(k)", "Frequency(Hz)", "Window(Samples)", "Window(ms)", "Q", "i-kmin"]))
    return table

def hamm(N, N_k):
    """
    An efficient algorithm for the calculation of a constant Q transform
    eq7, w[n-(N/2-N_k/2), k], with zeros output [N/2-N_k/2, N/2+N_k/2]
    """
    n = np.arange(0, N)
    w = 0.54 - 0.46 * np.cos(2.0 * np.pi * (n-(N/2-N_k/2)) / N_k)
    w[:int(N/2-N_k/2)] = 0.0
    w[int(N/2+N_k/2):] = 0.0
    return w

def tempKernel(N, table):
    """
    N for fft length
    Output:
        table : List[[k, hz, win_samp, win_sec, Q, temporal_kernel]]
    """
    for i in range(len(table)):
        N_k = table[i][2]
        Q = table[i][-1]
        window = hamm(N, N_k)
        y = np.zeros(N, dtype=complex)
        for n in range(N):
            _out = window[n] * np.exp(1j*2*np.pi*(Q/N_k)*(n-N/2))
            #_out = 1.0 * np.exp(1j*2*np.pi*(Q/N_k)*(n-N/2))
            y[n] = _out
        table[i].append(y)
    return table

def specKernel(N, table):
    """
    N for fft length
    Output:
        table : List[[k, hz, win_samp, win_sec, Q, temporal_kernel, spectral_kernal]]
        spectral_kernel = K[k, k_cq] in eq7, An efficient algorithm for the calculation of a constant Q transform
    """
    for i in range(len(table)):
        assert (table[i][-1].shape[0] == N)
        #def mdft(x, nfft, window_type=None):
        _out = scipy.fft.fft(table[i][-1], N)
        table[i].append(_out)
    return table

def eq5(x, sr, N, table, hop_size=512, MINVAL=0.10):
    y = np.zeros((len(table), np.int(np.floor(x.shape[0]/hop_size))), dtype=complex)
    freq_axis = []
    
    # Sliding(?) CQT
    for t_ptr in range(y.shape[1]): # Frame index
        x_offset = t_ptr * hop_size
        
        # zero-centered _x with length N
        if x_offset-N//2 < 0: # Left boundary
            _x = x[0:x_offset+N//2]
            _x = np.concatenate((np.zeros(N//2-(_x.shape[0]-N//2)), _x), axis=0)
        else:
            _x = x[x_offset-N//2 : x_offset+N//2]
            
        if _x.shape[0] < N: # Right boundary
            _x = np.concatenate((_x, np.zeros(N-_x.shape[0])), axis=0)

        X_ks = scipy.fft.fft(_x, N)
            
        for k_ptr in range(len(table)): # In a frame, k=0~Max_k, CQT of the kth bin
            Ycq_kcq = 0+0j
            spec_kernel = table[k_ptr][-1] # the kth spectral kernel
            valid_indices = np.where((np.abs(spec_kernel)/np.max(np.abs(spec_kernel))) >= MINVAL)[0]
            
            # Method 1, iteration
            #for k in valid_indices: # The kth nonzero term in the spectral kernl, the SUM part in eq5
                # X[k]
                #X_k = 0+0j
                #for n_X_k in range(N):
                #    _out = np.exp(-2.0*np.pi*1j*(k*n_X_k/N))*x[n_X_k]
                #    X_k += _out
                # >>> X[k]*comjugate(spec_kernel)
                #X_k = X_k*np.conjugate(spec_kernel[k])
            #    X_k = X_ks[k]*np.conjugate(spec_kernel[k])
            #    Ycq_kcq += X_k
            # Method 2, vectorize it
            #valid_mask = (np.abs(spec_kernel)/np.max(np.abs(spec_kernel))) >= MINVAL
            #Ycq_kcq = np.sum(X_ks*np.conjugate(spec_kernel)*valid_mask)

            # Method 2.5 Sum up the valid indices only
            Ycq_kcq = np.sum(X_ks[valid_indices]*np.conjugate(spec_kernel[valid_indices]))

            # Method 3, Sparse matrix
            #mask_out = (np.abs(spec_kernel)/np.max(np.abs(spec_kernel))) < MINVAL
            #X_ks[mask_out] = 0.0
            #s_Xks = bsr_matrix(X_ks.reshape(1, -1))
            #s_kernel = bsr_matrix(np.conjugate(spec_kernel).reshape(-1, 1))
            #Ycq_kcq = (s_Xks * s_kernel).toarray()[0][0]
            #Ycq_kcq = np.sum(X_ks*np.conjugate(spec_kernel)*valid_mask)
                
            # 1/N * SUM{X[k]*comjugate(spec_kernel)}
            Ycq_kcq /= N
            # Assign
            y[k_ptr, t_ptr] = Ycq_kcq
    
    # Frequency axis
    freqs = [table[i][1] for i in range(len(table))]
    return y, freqs

if __name__ == "__main__":
    X = DFT_slow(x)
    print (X.shape)
    exit()
    Xfft = FFT(x)
    print (X.shape)
    print (Xfft.shape)
    print (np.allclose(DFT_slow(x), np.fft.fft(x)))
    print (np.allclose(FFT(x), np.fft.fft(x)))

    t_mfft = timeit.repeat(setup='from __main__ import FFT, x', stmt='FFT(x)', repeat=1, number=10)
    t_fft = timeit.repeat(setup='from __main__ import x; import numpy as np', stmt='np.fft.fft(x)', repeat=1, number=10)
    t_slow = timeit.repeat(setup='from __main__ import DFT_slow, x', stmt='DFT_slow(x)', repeat=1, number=10)

    print ("Slow : {}".format(min(t_slow)))
    print ("fast : {}".format(min(t_fft)))
    print ("mFFT : {}".format(min(t_mfft)))
