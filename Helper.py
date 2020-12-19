from matplotlib.ticker import Formatter, ScalarFormatter
from matplotlib.ticker import LogLocator
import numpy as np
import librosa
from tabulate import tabulate
from matplotlib import pyplot as plt

class NoteFormatter(Formatter):
    # From https://librosa.org/doc/main/_modules/librosa/display.html#NoteFormatter
    '''Ticker formatter for Notes

    Parameters
    ----------
    octave : bool
        If `True`, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If `True`, ticks are always labeled.

        If `False`, ticks are only labeled if the span is less than 2 octaves

    See also
    --------
    LogHzFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> plt.figure()
    >>> ax1 = plt.subplot(2,1,1)
    >>> ax1.bar(np.arange(len(values)), values)
    >>> ax1.set_ylabel('Hz')
    >>> ax2 = plt.subplot(2,1,2)
    >>> ax2.bar(np.arange(len(values)), values)
    >>> ax2.yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax2.set_ylabel('Note')
    '''
    def __init__(self, octave=True, major=True):

        self.octave = octave
        self.major = major


    def __call__(self, x, pos=None):

        if x <= 0:
            return ''

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ''

        cents = vmax <= 2 * max(1, vmin)

        return librosa.core.hz_to_note(int(x), octave=self.octave, cents=cents)

class LogHzFormatter(Formatter):
    # From https://librosa.org/doc/0.7.2/_modules/librosa/display.html#LogHzFormatter
    '''Ticker formatter for logarithmic frequency

    Parameters
    ----------
    major : bool
        If `True`, ticks are always labeled.

        If `False`, ticks are only labeled if the span is less than 2 octaves

    See also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> plt.figure()
    >>> ax1 = plt.subplot(2,1,1)
    >>> ax1.bar(np.arange(len(values)), values)
    >>> ax1.yaxis.set_major_formatter(librosa.display.LogHzFormatter())
    >>> ax1.set_ylabel('Hz')
    >>> ax2 = plt.subplot(2,1,2)
    >>> ax2.bar(np.arange(len(values)), values)
    >>> ax2.yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax2.set_ylabel('Note')
    '''
    def __init__(self, major=True):
        self.major = major

    def __call__(self, x, pos=None):
        if x <= 0:
            return ''

        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ''

        return '{:g}'.format(x)

def plotTemp(table_k, plot_num, sr, N, isLinear=False, display_ratio=2, MINVAL=0.1):
    pick = []
    if isLinear:
        pick = [i for i in range(plot_num)]
    else:
        pick = np.linspace(0, len(table_k)-1, plot_num)
        pick = [int(i) for i in pick]

    v_cont = []

    plt.figure(figsize=(6, 6))
    for i, idx in enumerate(pick):
        _table = table_k[idx]
        tmp = _table[-2]
        k = _table[0]

        ax = plt.subplot(plot_num, 1, '{}'.format(i+1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(True)
        
        P = tmp.real
        plt.plot(P)
        plt.ylabel("k = {}".format(k))
        
    plt.tight_layout()

def plot2Kernel(table_k1, table_k2, plot_num, sr, N, isLinear=False, display_ratio=2, MINVAL=0.10):
    pick = []
    if isLinear:
        pick = [i for i in range(plot_num)]
    else:
        pick = np.linspace(0, len(table_k)-1, plot_num)
        pick = [int(i) for i in pick]

    v_cont = []

    plt.figure(figsize=(20, 20))
    for i, idx in enumerate(pick):
        ax = plt.subplot(plot_num, 1, '{}'.format(i+1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for table_k in [table_k1, table_k2]:
            _table = table_k[idx]
            tmp = _table[-1]
            k = _table[0]
            f = _table[1]
            
            P = np.abs(tmp[:N//display_ratio])/np.max(tmp[:N//display_ratio])
            ignore = sum(P<MINVAL)
            P[P<MINVAL] = 0.0
            freqs = [i*(sr/N) for i in range(N//display_ratio)]
            plt.plot(freqs, P)

        plt.ylabel("k = {}".format(k))
        
        v_cont.append([k, f, round(ignore/P.shape[0]*100, 2)])
    plt.tight_layout()

def plotCqt(cqt, freqs, sr, hop_size=512, title=""):
    """
    Input: 
        cqt: ndarray with shape (#k, sr*sec/hop_size)
        freqs : list with length #k [Hz of k=0, Hz of k=1, ...]
        sr : Sample Rate
        hop_size : Hop size
    """
    db_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)

    x = np.array([i*hop_size/sr for i in range(db_cqt.shape[1])])
    y = np.array(freqs)
    ax.pcolormesh(x, y, db_cqt, cmap='jet')

    ax.set_title("CQT {}".format(title))
    ax.set_xlabel("Time")
    ax.set_yscale("log", basey=2)

    ax.yaxis.set_major_formatter(LogHzFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=2.0))
    ax.yaxis.set_minor_formatter(LogHzFormatter(major=False))
    ax.yaxis.set_minor_locator(LogLocator(base=2.0,
                                      subs=2.0**(np.arange(1, 12)/12.0)))
    ax.yaxis.set_label_text('Hz')

        

def plotKernel(table_k, plot_num, sr, N, isLinear=False, display_ratio=2, MINVAL=0.1):
    pick = []
    if isLinear:
        pick = [i for i in range(plot_num)]
    else:
        pick = np.linspace(0, len(table_k)-1, plot_num)
        pick = [int(i) for i in pick]

    v_cont = []

    plt.figure(figsize=(6, 6))
    for i, idx in enumerate(pick):
        _table = table_k[idx]
        tmp = _table[-1]
        k = _table[0]
        f = _table[1]
        
        ax = plt.subplot(plot_num, 1, '{}'.format(i+1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        P = np.abs(tmp[:N//display_ratio])/np.max(tmp[:N//display_ratio])
        ignore = sum(P<MINVAL)
        P[P<MINVAL] = 0.0
        freqs = [i*(sr/N) for i in range(N//display_ratio)]
        plt.plot(freqs, P)
        plt.ylabel("k = {}".format(k))
        
        v_cont.append([k, f, round(ignore/P.shape[0]*100, 2)])
    plt.tight_layout()
    print(tabulate(v_cont, headers=["Channel(k)", "Frequency(Hz)", "Discard %"]))

def plot2Kernel(table_k1, table_k2, plot_num, sr, N, isLinear=False, display_ratio=2, MINVAL=0.10):
    pick = []
    if isLinear:
        pick = [i for i in range(plot_num)]
    else:
        pick = np.linspace(0, len(table_k)-1, plot_num)
        pick = [int(i) for i in pick]

    v_cont = []

    plt.figure(figsize=(20, 20))
    for i, idx in enumerate(pick):
        ax = plt.subplot(plot_num, 1, '{}'.format(i+1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for table_k in [table_k1, table_k2]:
            _table = table_k[idx]
            tmp = _table[-1]
            k = _table[0]
            f = _table[1]
            
            P = np.abs(tmp[:N//display_ratio])/np.max(tmp[:N//display_ratio])
            ignore = sum(P<MINVAL)
            P[P<MINVAL] = 0.0
            freqs = [i*(sr/N) for i in range(N//display_ratio)]
            plt.plot(freqs, P)

        plt.ylabel("k = {}".format(k))
        
        v_cont.append([k, f, round(ignore/P.shape[0]*100, 2)])
    plt.tight_layout()
    #print(tabulate(v_cont, headers=["Channel(k)", "Frequency(Hz)", "Discard %"]))

def plotCqt(cqt, freqs, sr, hop_size=512, title=""):
    """
    Input: 
        cqt: ndarray with shape (#k, sr*sec/hop_size)
        freqs : list with length #k [Hz of k=0, Hz of k=1, ...]
        sr : Sample Rate
        hop_size : Hop size
    """
    db_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    x = np.array([i*hop_size/sr for i in range(db_cqt.shape[1])])
    y = np.array(freqs)
    ax.pcolormesh(x, y, db_cqt, cmap='jet')

    ax.set_title("CQT {}".format(title))
    ax.set_xlabel("Time")
    ax.set_yscale("log", basey=2)

    #ax.yaxis.set_major_formatter(LogHzFormatter())
    ax.yaxis.set_major_formatter(NoteFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=2.0))
    #ax.yaxis.set_minor_formatter(LogHzFormatter(major=False))
    ax.yaxis.set_minor_formatter(NoteFormatter(major=False))
    ax.yaxis.set_minor_locator(LogLocator(base=2.0,
                                      subs=2.0**(np.arange(1, 12)/12.0)))
    ax.yaxis.set_label_text('Hz')
