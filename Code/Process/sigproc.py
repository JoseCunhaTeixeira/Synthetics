# -*- coding: utf-8 -*-
""" Signal processing library
----------------------------

Library for all signal processing tools : filtering, tapering, etc.

Try to mimic the matlab land processing toolbox when possible
 
Expected matrix order is Column major (C-style) for faster processing
i.e. (n_trace, n_sample)


 @author: brondeleux, phardouin
"""

from scipy import fftpack
from scipy import optimize
from mkl_fft import rfft, irfft, fft, ifft
# from numpy.fft import rfft, irfft, fft, ifft
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-10


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FFT
# -----------------------------------------------------------------------------

def rfft_min_ech(vector, min_ech=None):
    """ rfft with minimal samples condition (zero pad if necessary) """
    if vector.ndim == 1:
        vector = vector.reshape((1, vector.size))
    (_, n) = nZ(vector)
    if min_ech is None:
        min_ech = n
    return rfft(vector, n=max(min_ech, n))
# -----------------------------------------------------------------------------

def rfft_cut(vector, duration, si):
    """ rfft with minimal samples condition (zero pad if necessary) """
    ech = int((duration / si)+1)
    return rfft_min_ech(vector, min_ech = ech)
# -----------------------------------------------------------------------------

def rfft2complex(vector):
    """ rfft2complex(Y)
    Transform a rfft real vector of length n in a complex vector of length n//2+1
    """
    # Initialization
    if vector.ndim == 1:
        vector = vector.reshape((1, vector.size))
    (_, n) = nZ(vector)

    if n % 2:  # odd
        return np.hstack((vector[..., 0:1], vector[..., 1::2] + 1j*vector[..., 2::2]))
    else:  # even
        return np.hstack((vector[..., 0:1], vector[..., 1:-1:2] + 1j*vector[..., 2::2], vector[..., -1:]))


# -----------------------------------------------------------------------------
def complex2rfft(vector, nsamp):
    """ complex2rfft(Y,n):
    Transform a complex fft vector of length n in a real rfft vector of length 2*(n-1) (+1 si n impair)
    """
    # Initialization
    if vector.ndim == 1:
        vector = vector.reshape((1, vector.size))
    (ntr, n) = nZ(vector)
    assert (nsamp in (2*(n-1), 2*n-1)), "number of sample %i not compatible with rfft shape (%i,%i)" \
                                                 % (nsamp, ntr, n)

    outvect = np.zeros((ntr, nsamp), dtype=np.double)
    outvect[..., 0] = vector[..., 0].real
    outvect[..., 1::2] = vector[..., 1:].real
    if nsamp % 2:  # odd
        outvect[..., 2::2] = vector[..., 1:].imag
    else:
        outvect[..., 2::2] = vector[..., 1:-1].imag

    return outvect


# -----------------------------------------------------------------------------
def centerfft(vector, nel=-1):
    """
    fftshift like : put 0 time in the middle with nel elements above and below
    """
    if vector.ndim == 1:
        vector = vector.reshape((1, vector.size))
    (ntr, n) = nZ(vector)
    vector = vector.reshape((ntr, n))
    if nel>0:
        nel = np.min((nel, (n-1)//2))
    else:
        nel = (n-1)//2
    return np.hstack((vector[..., -nel:], vector[..., 0:nel+1]))

# -----------------------------------------------------------------------------
def centerfft_dur_si(vector, duration , si):
    ech = int((duration / si) /2)
    return centerfft(vector, nel=ech)
# -----------------------------------------------------------------------------
def fftshift(vector):
    """    fftshift like
    """
    if vector.ndim == 1:
        vector = vector.reshape((1, vector.size))
    (ntr, n) = nZ(vector)
    vector = vector.reshape((ntr, n))
    f = fftpack.fftfreq(n,1)
    ifo = np.argsort(f)
    return vector[..., ifo]

# -----------------------------------------------------------------------------
def filterLPF(vector, si, f_low, f_high):
    """ Low pass filter on rFFT data

    args:
        vector (numpy array) : rfft-data
        si (float) : sampling interval in seconds
        f_low (float) : begin slope frequency
        f_high (float) : end slope frequency

    returns:
        (numpy array) : filtered rfft-data
    """
    # Initialization
    [f_low, f_high] = np.sort([f_low, f_high])+1e-10*np.array([-1, 1])
    (_, n) = nZ(vector)
    # Parameters
    f = fftpack.rfftfreq(n, si)
    filc = (f <= f_low) + ((f > f_low)*(f < f_high))*(0.5*(1+np.sin(np.pi*(f-((f_low+f_high)/2))/(f_low-f_high))))
    # Processing
    return vector*filc[None, :]


# -----------------------------------------------------------------------------
def filterHPF(vector, si, f_low, f_high):
    """ High pass filter on rFFT data

    args:
        vector (numpy array) : rfft-data
        si (float) : sampling interval in seconds
        f_low (float) : begin slope frequency
        f_high (float) : end slope frequency

    returns:
        (numpy array) : filtered rfft-data
    """
    # Initialization
    [f_low, f_high] = np.sort([f_low, f_high])+1e-10*np.array([-1, 1])
    (_, n) = nZ(vector)
    # Parameters
    f = fftpack.rfftfreq(n, si)
    filc = (f >= f_high) + ((f > f_low)*(f < f_high))*(0.5*(1+np.sin(np.pi*(f-((f_low+f_high)/2))/(f_high-f_low))))
    # Processing
    return vector*filc[None, :]


# -----------------------------------------------------------------------------
def filterBPF(vector, si, f_1, f_2, f_3, f_4, op_notch=False):
    """ Band pass filter on rFFT data

    args:
        vector (numpy array) : rfft-data
        si (float) : sampling interval in seconds
        f_1, f_2, f_3, f_4 (float) : corners of box-frequency
        op_notch (bool) : specifies if notch or band pass (optional)

    returns:
        (numpy array) : filtered rfft-data
    """
    # Initialization
    [f_1, f_2, f_3, f_4] = np.sort([f_1, f_2, f_3, f_4])+1e-10*np.array([-1, 1, -1, 1])
    (_, n) = nZ(vector)
    # Parameters
    f = fftpack.rfftfreq(n, si)
    fil1 = ((f > f_1)*(f < f_2))*(0.5*(1+np.sin(np.pi*(f-((f_2+f_1)/2))/(f_2-f_1))))
    fil2 = ((f >= f_2)*(f <= f_3))
    fil3 = ((f > f_3)*(f < f_4))*(0.5*(1+np.sin(np.pi*(f-((f_3+f_4)/2))/(f_3-f_4))))
    filc = fil1+fil2+fil3

    # Processing
    outvect = vector*filc[None, :]
    if op_notch:
        outvect = vector-outvect

    return outvect
def filterBPF_multiple(vector, si, fc, fp = 1/6, op_notch=False):
    """ multiple Band pass filter on rFFT data 

    args:
        vector (numpy array) : rfft-data
        si (float) : sampling interval in seconds
        fc (float) : center frequences in Hz
        fp (float):  decend window size in proportion to centual frequency, half window size= 2*p_band
        op_notch (bool) : specifies if notch or band pass (optional)

    returns:
        (numpy array) : filtered rfft-data
    """
    # Initialization
    freq_box = fc.reshape((-1,1)) *  np.array([1 -3*fp , 1-2*fp, 1+2*fp, 1 +3*fp]) # linear ratio
    # freq_box = np.power(10, np.log10(fc.reshape((-1,1))) *  np.array([1 -3*fp , 1-2*fp, 1+2*fp, 1 +3*fp])) # log ratio

    nb_filter = freq_box.shape[0]
    (_, n) = nZ(vector)
    # Parameters
    f = fftpack.rfftfreq(n, si).reshape((-1,1))
    
    fc_matrix = np.repeat(fc.reshape((1,-1)),f.size,axis=0)
    s_matrix = ( np.divide( ( np.repeat(f,nb_filter,axis=1) - (1-2.5*fp)* fc_matrix ), (fp*fc_matrix )))
    f_sig = 0.5*( 1+ np.sin(np.pi* s_matrix))

    # Filter matrix in frequency domain 
    outvect = np.zeros((vector.shape[0],vector.shape[1], nb_filter))
    F = np.zeros((f.size, nb_filter))
    for i in range(nb_filter):
        filc = np.zeros((f.size,))
        index_ones = ((f>freq_box[i,1]) & (f<freq_box[i,2])).flatten()
        filc[index_ones] = 1
        index_sig = (( (f>freq_box[i,0]) & (f<freq_box[i,1]) ) | ( (f>freq_box[i,2]) &(f<freq_box[i,3]) )).flatten()
        filc[index_sig] = f_sig[index_sig,i]

        F[:,i] = filc
        outvect[:,:,i] = vector*filc[None,:]     

    # display_BF_frequency(F,f.reshape((-1,)),fc,fp)
    if op_notch:
        outvect = np.repeat(vector,nb_filter,axis=2)-outvect
        
    return outvect

def display_BF_frequency(F,f,fc,fp):
    xticks = np.arange(0,fc.size,10)
    xticklabels = [f"{ff:5.2f}" for ff in fc[xticks]]

    yticks = np.arange(0,f.size,200)
    yticklabels = [f"{ff:5.2f}" for ff in f[yticks]]

    plt.figure(figsize=(16,9))
    plt.imshow(F, aspect='auto',cmap='gray')
    plt.title(f'ratio_quartwin=1/{int(np.round(1/fp))}')
    plt.xticks(xticks,xticklabels)
    plt.yticks(yticks,yticklabels)
    plt.xlabel('central frequency (Hz)')
    plt.ylabel('filter frequency (Hz)')
    plt.colorbar()
    # plt.savefig(f"/share/projects/MONITORING/MINOS_TEST/TEST_CAIFANG/improve_pick/PNG/pick/Filter_BF_fp{int(np.round(1/fp))}.png",dpi=150,bbox_inches='tight')
    plt.show()
    # plt.close()

    plt.figure(figsize=(16,9))
    plt.plot(f, F, '--o')
    plt.plot(fc, np.zeros(fc.shape), 'k*',label='central frequency')
    plt.title(f'Band-pass filter H(f), ratio_quartwin=1/{int(np.round(1/fp))}')
    plt.xlim([0, 1.3*np.max(fc)])
    plt.xlabel('frequency (Hz)')
    plt.ylabel('filter response')
    plt.legend()
    # plt.savefig(f"/share/projects/MONITORING/MINOS_TEST/TEST_CAIFANG/improve_pick/PNG/pick/Filter_BF_response_fp{int(np.round(1/fp))}.png",dpi=150,bbox_inches='tight')
    plt.show()
    # plt.close()
# -----------------------------------------------------------------------------
def whitenF(vector, si, bw, perc):
    """
        Whiten an rfft-ed Matrix along columns, with whitening percentage perc,
        and bandpass filter

    args:
        vector (numpy array) : rfft-data
        si (float) : sampling interval in seconds
        bw (4-float tuple) : (f_1, f_2, f_3, f_4)
        perc (float) : whitening percentage of total rms

    returns:
        (numpy array) : whitened rfft-data
    """
    assert len(bw) == 4, "Bandwith argument has wrong length (%i, expected 4)" % len(bw)
    (_, n) = nZ(vector)
    # normalize
    percoef = perc/1e2*rms(vector) + EPS
    # TODO: compare with direct rfft format to see if it is faster
    outvect = rfft2complex(vector)
    outvect = outvect/(np.abs(outvect)+percoef[:, None])*np.sqrt(n)
    return filterBPF(complex2rfft(outvect, n), si, *bw)


# -----------------------------------------------------------------------------
def correlF(aa, bb):
    """ Correlation of 1D/2D arrays issued from rfft, in frequency domain
        column dimension must be compatible

    args:
        aa, bb (numpy array) : rfft-data to correlate

    usage : c = correlF(a,b)

    ex : >>> z = irfft(convolF(rfft(x),rfft(y)))

    """
    # reshape for vector compatibility
    for v in [aa, bb]:
        if v.ndim == 1:
            v = v.reshape((1, v.size))

        (_, n) = nZ(v)
        # v = v.reshape((ntr, n))

    # and multiply
    cc = np.zeros_like(aa)
    cc[..., 0] = aa[..., 0]*bb[..., 0]  # continuous (real)
    cc[..., 1:-1:2] = aa[..., 1:-1:2]*bb[..., 1:-1:2] + aa[..., 2::2]*bb[..., 2::2]
    cc[..., 2::2] = (-aa[..., 1:-1:2]*bb[..., 2::2] + aa[..., 2::2]*bb[..., 1:-1:2])
    if not n % 2:  # even
        cc[..., -1] = aa[..., -1]*bb[..., -1]
    return cc


# -----------------------------------------------------------------------------
def convolF(aa, bb):
    """ convolution of 1D/2D arrays issued from rfft, in frequency domain
    column dimension must be compatible

    args:
        aa, bb (numpy array) : rfft-data to convolute

    usage : c = convolF(a,b)

    ex : >>> z = irfft(convolF(rfft(x),rfft(y)))

    """
    # reshape for vector compatibility
    for v in [aa, bb]:
        if v.ndim == 1:
            v = v.reshape((1, v.size))
        (_, n) = nZ(v)
        # v = v.reshape((ntr, n))

    # and multiply
    cc = np.zeros_like(aa)
    cc[..., 0] = aa[..., 0]*bb[..., 0]  # continuous (real)
    cc[..., 1:-1:2] = aa[..., 1:-1:2]*bb[..., 1:-1:2] - aa[..., 2::2]*bb[..., 2::2]
    cc[..., 2::2] = (aa[..., 1:-1:2]*bb[..., 2::2] + aa[..., 2::2]*bb[..., 1:-1:2])
    if not n % 2:  # even
        cc[..., -1] = aa[..., -1]*bb[..., -1]
    return cc


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PROCESS
# -----------------------------------------------------------------------------
def rms(vector):
    """ returns the rms of a numpy array """
    return np.sqrt(np.mean(vector**2, -1))


# -----------------------------------------------------------------------------
def nZ(vector):
    """ returns the would-be 2d shape of a nd-numpy array
    with last dimension preserved"""
    vectshape = vector.shape
    vectdim = vector.ndim
    if vectdim == 0:
        return [0, 0]
    elif vectdim == 1:
        return [1, vectshape[0]]
    else:
        return [np.prod(vectshape[0:vectdim-1]), vectshape[-1]]


# -----------------------------------------------------------------------------
def filterLP(vector, si, f_low, f_high):
    """ Low pass filter on time data

    args:
        vector (numpy array) : data in t-x domaine
        si (float) : sampling interval in seconds
        f_low (float) : begin slope frequency
        f_high (float) : end slope frequency

    returns:
        (numpy array) : filtered data
    """
    # Initialization
    vectshape = vector.shape
    vectdim = vector.ndim
    (ntr, n) = nZ(vector)
    if vectdim > 2:
        vector = vector.reshape(ntr, n)
    # Processing
    vectout = irfft(filterLPF(rfft(vector), si, f_low, f_high))
    if vectdim > 2:
        vectout = vectout.reshape(vectshape)
    return vectout


# -----------------------------------------------------------------------------
def filterHP(vector, si, f_low, f_high):
    """ High pass filter on time data

    args:
        vector (numpy array) : data  in t-x domain
        si (float) : sampling interval in seconds
        f_low (float) : begin slope frequency
        f_high (float) : end slope frequency

    returns:
        (numpy array) : filtered data
    """
    # Initialization
    vectshape = vector.shape
    vectdim = vector.ndim
    (ntr, n) = nZ(vector)
    if vectdim > 2:
        vector = vector.reshape(ntr, n)
    # Processing
    vectout = irfft(filterHPF(rfft(vector), si, f_low, f_high))
    if vectdim > 2:
        vectout = vectout.reshape(vectshape)
    return vectout


# -----------------------------------------------------------------------------
def filterBP(vector, si, f_1, f_2, f_3, f_4, op_notch=False):
    """ Band pass filter on time data

    args:
        vector (numpy array) : data
        si (float) : sampling interval in seconds
        f_1, f_2, f_3, f_4 (float) : corners of box-frequency
        op_notch (bool) : specifies if notch or band pass (optional)

    returns:
        (numpy array) : filtered data
    """
    # Initialization
    (ntr, n) = nZ(vector)
    vectshape = vector.shape
    vectdim = vector.ndim
    if vectdim > 2:
        vector = vector.reshape(ntr, n)
    # Processing
    vectout = irfft(filterBPF(rfft(vector), si, f_1, f_2, f_3, f_4, op_notch))
    if vectdim > 2:
        vectout = vectout.reshape(vectshape)
    return vectout
def filterBP_multiple(vector, si, fc, fp=1/6, op_notch=False):
    """ Band pass filter on time data

    args:
        vector (numpy array) : data
        si (float) : sampling interval in seconds
        f_1, f_2, f_3, f_4 (float) : corners of box-frequency
        op_notch (bool) : specifies if notch or band pass (optional)

    returns:
        (numpy array) : filtered data
    """
    # Initialization
    (ntr, n) = nZ(vector)
    nf = fc.size
    vectshape = vector.shape
    vectdim = vector.ndim
    if vectdim > 2:
        vector = vector.reshape(ntr, n)
    # Processing
    vectout = np.zeros((vector.shape[0],vector.shape[1],nf))
    s_filted= filterBPF_multiple(rfft(vector), si, fc, fp = fp, op_notch=op_notch)
    for i in range(nf):
        vectout[:,:,i] = irfft(s_filted[:,:,i])
    if vectdim > 2:
        vectout = vectout.reshape(vectshape)
    return vectout

# -----------------------------------------------------------------------------
def whiten(vector, si=1, bw=None, perc=2):
    """ Whitening filter on time data

    args:
        vector (numpy array) : data
        si (float) : sampling interval in seconds
        bw = (f_1, f_2, f_3, f_4) (float tuple) : corners of box-frequency
        perc (float) : whitening percentage

    returns:
        (numpy array) : filtered data
    """
    # Initialization
    (ntr, n) = nZ(vector)

    if bw is None:
        bw = (0,1/(si*(n-1)), 0.5/si-1/(si*(n-1)), 0.5/si)
    vectshape = vector.shape
    vectdim = vector.ndim
    if vectdim > 2:
        vector = vector.reshape(ntr, n)
    # Processing
    vectout = irfft(whitenF(rfft(vector), si, bw, perc))
    if vectdim > 2:
        vectout = vectout.reshape(vectshape)
    #vectout = vectout / rms(vectout)[:,None]
    return vectout

def correl(vector1, vector2):
    """ Correlation (computed in freq domain but input in time)
        zero lag is centered as in fftpack.fftfreq(dim_1)
    """
    # Initialization
    (ntr, n) = nZ(vector1)
    vectshape = vector1.shape
    assert vector1.shape == vector2.shape
    vectdim = vector1.ndim
    if vectdim > 2:
        vector1 = vector1.reshape(ntr, n)
        vector2 = vector2.reshape(ntr, n)
    # Processing
    vectout = irfft(correlF(rfft(vector1), rfft(vector2)))
    order = np.argsort(fftpack.fftfreq(n), axis=-1)
    vectout = vectout[..., order]
    if vectdim > 2:
        vectout = vectout.reshape(vectshape)
    return vectout


# -----------------------------------------------------------------------------
def ltasta(vector, nl, ns, two_side=False):
    """ (ratio,lta,sta) = ltasta(A,nl,ns,twoSide=False)

    Compute the sta / lta ratio in fourier domain
    ratio is computed on the squared signal (not rms)

    output:
      (ratio,lta,sta)
    OR if twoside=True
      ((ratioUp,ltaUp,staUp),(ratioDown,ltaDown,staDown))

    """
    # reshape for vector compatibility
    if vector.ndim == 1:
        vector = vector.reshape((1, vector.size))
    (ntr, n) = nZ(vector)
    vector = vector.reshape((ntr, n))

    # do FFT and convolve both long term and short term window
    nechFFT = fftpack.helper.next_fast_len(n+nl+ns)
    vector = taperize(vector,n//6)
    A2F = rfft(vector**2, nechFFT)

    winsta = np.zeros((1, ns+nl))
    winsta[0, nl:] = 1/ns
    winsta = rfft(winsta, nechFFT)
    winlta = rfft(np.ones((1, nl))/nl, nechFFT)
    sta = irfft(convolF(A2F, winsta), nechFFT)
    lta = irfft(convolF(A2F, winlta), nechFFT)
    sta = sta[..., nl+ns:nl+ns+n]
    lta = lta[..., 0:n]
    sta[np.where(sta<0)] = 0
    lta[np.where(lta<0)] = 0
    res = (sta/(lta+1e-20), lta, sta)

    if two_side:
        # Do the other side
        winlta = np.zeros((1, ns+nl))
        winlta[0, ns:] = 1/nl
        winlta = rfft(winlta, nechFFT)
        winsta = rfft(np.ones((1, ns))/ns, nechFFT)
        sta = irfft(convolF(A2F, winsta), nechFFT)
        lta = irfft(convolF(A2F, winlta), nechFFT)
        sta = sta[..., 0:n]
        lta = lta[..., ns+nl:n+ns+nl]
        sta[np.where(sta<0)] = 0
        lta[np.where(lta<0)] = 0
        res = (res, (sta/(lta+1e-10), lta, sta))

    return res


# -----------------------------------------------------------------------------
def every_max(vector):
    """ indM,valM = every_max(A)

    Returns indices and values for each row in A with cubic interpolation when possible
    """
    # reshape for vector compatibility and nd-array
    if vector.ndim == 1:
        vector = vector.reshape((1, vector.size))
    (ntr, n) = nZ(vector)
    vector = vector.reshape((ntr, n))

    iMax = vector.argmax(axis=-1)
    vMax = vector[(range(vector.shape[0]), iMax)]
    # determine if we can make a cubic interpolation
    compMore = np.logical_and(iMax > 0, iMax < (n-1))

    if compMore.any():

        # Proceed to cubic interpolation
        icomp = np.r_[0:ntr][compMore]
        # iMore = vstack((iMax-1,iMax,iMax+1))
        vMore = np.hstack((vector[(icomp, iMax[icomp]-1)],
                           vMax[icomp], vector[(icomp, iMax[icomp]+1)]))
        iMax = iMax*1.0
        if vMore.ndim == 1:
            vMore = vMore.reshape((1, vMore.size))

        c = vMore[..., 1]
        b = (vMore[..., 2]-vMore[..., 0])/2
        a = (vMore[..., 2]+vMore[..., 0]-2*vMore[..., 1])/2
        iMax[icomp] = iMax[icomp]-b/2/a
        vMax[icomp] = c - b**2/4/a
    return iMax, vMax


def LA(vector, y, nrec_max=10):
    """
    x = LA(A,y,nrec_max)

    Solve Least Absolute Deviation problem y=A.x by iterative least square

    Input :
        A,y : as in y=Ax
        nrec_max : maximum number of iteration (default 10)

    Output:
        x : as in argmin(|y-Ax|)
    """
    vector = vector/(rms(y)+EPS)
    y = y/(rms(y)+EPS)  # TODO: compatibility with matrix y

    x = optimize.lsq_linear(vector, y)['x']
    count_rec = 0
    converged = False
    while not converged or count_rec <= nrec_max:
        xold = x
        W = 1./np.maximum(np.sqrt(np.abs(y-np.dot(vector, x))), 1e-2)
        x = optimize.lsq_linear(vector*W[..., None], y*W)['x']
        if rms(xold-x) < 1e-6:
            #print('LA : converged\n')
            converged = True
        count_rec += 1

    if count_rec == nrec_max:
        print('LA :maximum iteration exceeded\n')

    return x


def taperize(vector, ntap, inplace=False):
    """
    Apply a hanning tapper of length ntap at the beginning and the end of Z
    """
    if not inplace:
        vector = vector.copy()
    ntap = min(ntap, vector.shape[1]//2)
    wind = np.hanning(2*ntap+1)
    vector[..., 0:ntap] = vector[..., 0:ntap] * wind[None, 0:ntap]
    vector[..., -ntap:] = vector[..., -ntap:] * wind[None, ntap+1:]
    return vector


def taperize_along(vector, ind, ntap, mode='b', inplace=False):
    """
    Apply tapper of length ntap starting at (mode'B') or ending at (mode 'E') ind along Z rows
    """
    assert vector.ndim == 2, "taperize_along : only 2-dimensional Z are accepted"
    ind = np.squeeze(ind)
    assert ind.ndim == 1 and len(ind) == vector.shape[0], \
        "taperize_along : invalid indices, should be 1D and of length Z.shape[0]"
    assert type(mode) is str and (mode.lower() == 'b' or mode.lower() == 'e'),\
        "taperize_along : invalid mode value ('B' or 'E' only)"
    mode = mode.lower()
    ind = ind.astype(int)
    assert ind.max() < vector.shape[1], "invalid index: longer than shape[1]"
    if not inplace:
        vector = vector.copy()

    wind = np.hanning(2*ntap+1)
    if mode == 'b':
        wind = wind[0:ntap]
        istart = np.maximum(ind-ntap, 0)
        istop = ind
        wstart = istart+ntap-ind
        wstop = wstart*0 + ntap
    else:  # mode='e'
        wind = wind[ntap+1:]
        istop = np.minimum(ind+ntap, vector.shape[1])
        istart = ind
        wstop = istop-ind
        wstart = wstop*0
    for krow in range(vector.shape[0]):
        vector[krow, istart[krow]:istop[krow]] = vector[krow, istart[krow]:istop[krow]]*wind[wstart[krow]:wstop[krow]]
        if mode == 'b':
            vector[krow, :istart[krow]] = 0
        else:
            vector[krow, istop[krow]:] = 0

    return vector

def agc(vector, nl):
    """ apply AGC on data

    args:
        vector (numpy array) : data
        nl : number of samples for AGC window

    returns:
        (vector_agc, _rms) : normalized data and corresponding coefficients
    """
    _, _, _rms = ltasta(vector, 0, nl)
    _rms[:, -nl:] = _rms[:, -nl-1].reshape(-1, 1) + np.zeros((1, nl))
    return (vector/(_rms+EPS), _rms)

def normalize_rms(data):
    return data/rms(data)[:, None]


def makeFV(vector, si, offset, vmin, vmax, dv, fmax):
    """   Construct a FV dispersion diagram

    args :
        vector (numpy array) : data
        si (float) : sampling interval in seconds
        offsets (numpy array) : offsets in meter
        vmin, vmax (float) : velocities to scan in m/s
        dv (float) : velocity step in m/s
        fmax (float) : maximum frequency computed

    returns :
        f : frequency axis
        v : velocity axis
        FV: dispersion plot

    """
    npad = 1024
    (_, n) = nZ(vector)
    n = np.max((n,npad))
    vector = rfft_min_ech(vector, min_ech=npad)
    offset = np.squeeze(offset)

    f = fftpack.rfftfreq(n, si)
    imax = np.where(f > fmax)[0][0]
    f = rfft2complex(f)
    f = np.real(f[..., 0:imax//2+1])

    #vector = whitenF(vector, si, [0, 1e-2, fmax, fmax*1.1], 1)
    vector = rfft2complex(vector)
    vector = vector[..., 0:imax//2+1]

    v = np.array(np.r_[vmin:vmax:dv], dtype = float)
    FV = np.zeros((len(v), max(f.shape)))
    for kv, vv in enumerate(v):
        dphi = 2 * np.pi * offset[..., None] * f / vv
        FV[kv, :] = np.abs(np.sum(vector*np.exp(1j*dphi), axis=0))

    return FV, v, f.T.squeeze()

def transFK(vector, si, dx):
    (ntr, n) = nZ(vector)

    vector = rfft(vector)

    # Parameters
    f = fftpack.rfftfreq(n, si)
    k = fftpack.fftfreq(ntr, dx)
    #imax = np.where(f>fmax)[0][0]

    #vector = whitenF(vector, si, [0, 1e-2, fmax, fmax*1.1],1)
    vector = rfft2complex(vector) #now in FX complex

    f = rfft2complex(f)
    f = np.real(f[..., 0:vector.shape[1]])
    vector = fft(vector, axis=0) #now in FK
    return vector,f,k

def filFK(vector, si, dx, vmin, vmax, agcLen=0.5, applyAGC=True):
    """   FK filter (still experimental)

    """

    (_, n) = nZ(vector)
    nl = int(agcLen/si)

    if applyAGC:
        vector, _rms = agc(vector, nl)
    else:
        _rms = np.zeros_like(vector) + 1

    vector,_,k = transFK(vector, si, dx)

    fmin = np.abs(k*vmin)
    ifmin = np.maximum(np.round(fmin * ((n-1)*si)), 1)
    fmax = np.abs(k*vmax)
    ifmax = np.maximum(np.round(fmax * ((n-1)*si)), 1)
    ifmax = np.minimum(ifmax, vector.shape[1]-1)
    ifmin = np.minimum(ifmin, vector.shape[1]-1)
    taperize_along(vector, ifmin, n//200, 'b', True)
    taperize_along(vector, ifmax, n//200, 'e', True)

    return irfft(complex2rfft(ifft(vector, axis=0), n))*_rms


def ampSpectra(vector, si):
    '''Vector in the larger sense of an array
    '''
    # Initialization
    vectdim = vector.ndim
    (ntr, n) = nZ(vector)
    if vectdim > 2:
        vector = vector.reshape(ntr, n)

    #version using fft (vs using rfft)
    ampSpectra = np.abs(fft(vector))
    fpack = fftpack.fftfreq(n, si)
    
    #version using rfft (vs using fft)
    ampSpectra = np.abs(rfft2complex(rfft(vector)))
    idxp = np.argwhere(fpack>=0)#keep positive part of frequencies for display

    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(2,1)
    #ax[1].plot(fpack[idxp],ampSpectrar[10,idxp])
    #ax[1].plot(fpack[idxp],ampSpectra[10,idxp])
    #ax[0].plot(fpack[idxp],np.average(ampSpectra[:,idxp],axis=0))
    #plt.show()

    return ampSpectra[:,idxp].squeeze(), fpack[idxp].squeeze()

def remove_median_trace(data):
    med_trace = np.median(data,axis=0)
    return data-med_trace
if __name__ == '__main__':

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt


    ntr = 100
    nech = 1000
    si = 2e-3
    dx = 3
    data = np.zeros((ntr, nech))
    data[ntr//2, nech//2]=1
    fk = filFK(data, si, dx, 100, 500, applyAGC=False)

def multiproc(data, param, key):
        """ all preprocessing step before correlation or detection etc..., 
        ORDER MATTERS -> first func applied first ([fa,fb] -> fb(fa(data)))
        """
        #List of preprocessing function, with list of secondary arguments
        if data is not None:
            if data.shape[0]>0:
                for fun, args in param[key]:
                    for ind,arg in enumerate(args):
                        if isinstance(arg,str) :
                            args[ind] = param[arg]
                    data = fun(data, *args)
            else :
                data = []
        else :
            data = []               
        return data

def pad_data(data,before,after):
    return np.pad(data,((0,0),(before,after)),'constant', constant_values=(0, 0))  