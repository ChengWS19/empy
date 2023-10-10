import empymod
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

###############################################################################
def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
    """Apply a source waveform to the signal.

    Parameters
    ----------
    times : ndarray
        Times of computed input response; should start before and end after
        `times_wanted`.

    resp : ndarray
        EM-response corresponding to `times`.

    times_wanted : ndarray
        Wanted times.

    wave_time : ndarray
        Time steps of the wave.

    wave_amp : ndarray
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].

    nquad : int
        Number of Gauss-Legendre points for the integration. Default is 3.

    Returns
    -------
    resp_wanted : ndarray
        EM field for `times_wanted`.

    """

    # Interpolate on log.
    PP = iuSpline(np.log10(times), resp)

    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI/dt

    # Gauss-Legendre Quadrature; 3 is generally good enough.
    # (Roots/weights could be cached.)
    g_x, g_w = roots_legendre(nquad)

    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)

    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):

        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue

        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if ind_a.sum() == 0:
            continue

        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i+1] > times_wanted[ind_a]

        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a]-wave_time[i]
        tb = times_wanted[ind_a]-wave_time[i+1]
        tb[ind_b] = 0.0  # Cut elements

        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        logt = np.log10(np.outer((tb-ta)/2, g_x)+(ta+tb)[:, None]/2)
        fact = (tb-ta)/2*cdIdt
        resp_wanted[ind_a] += fact*np.sum(np.array(PP(logt)*g_w), axis=1)

    return resp_wanted

###############################################################################
def get_time(time, r_time):
    """Additional time for ramp.

    Because of the arbitrary waveform, we need to compute some times before and
    after the actually wanted times for interpolation of the waveform.

    Some implementation details: The actual times here don't really matter. We
    create a vector of time.size+2, so it is similar to the input times and
    accounts that it will require a bit earlier and a bit later times. Really
    important are only the minimum and maximum times. The Fourier DLF, with
    `pts_per_dec=-1`, computes times from minimum to at least the maximum,
    where the actual spacing is defined by the filter spacing. It subsequently
    interpolates to the wanted times. Afterwards, we interpolate those again to
    compute the actual waveform response.

    Note: We could first call `waveform`, and get the actually required times
          from there. This would make this function obsolete. It would also
          avoid the double interpolation, first in `empymod.model.time` for the
          Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
          Probably not or marginally faster. And the code would become much
          less readable.

    Parameters
    ----------
    time : ndarray
        Desired times

    r_time : ndarray
        Waveform times

    Returns
    -------
    time_req : ndarray
        Required times
    """
    tmin = np.log10(max(time.min()-r_time.max(), 1e-10))
    tmax = np.log10(time.max()-r_time.min())
    return np.logspace(tmin, tmax, time.size+2)

###############################################################################
def walktem(time_in, waveflag, signal,src, rec, depth, res, mrec, strength, srcpts, caltype):
    """Custom wrapper of empymod.model.bipole.

    Here, we compute WalkTEM data using the ``empymod.model.bipole`` routine as
    an example. We could achieve the same using ``empymod.model.dipole`` or
    ``empymod.model.loop``.

    We model the big source square loop by computing only half of one side of
    the electric square loop and approximating the finite length dipole with 3
    point dipole sources. The result is then multiplied by 8, to account for
    all eight half-sides of the square loop.

    The implementation here assumes a central loop configuration, where the
    receiver (1 m2 area) is at the origin, and the source is a 40x40 m electric
    loop, centered around the origin.

    Note: This approximation of only using half of one of the four sides
          obviously only works for central, horizontal square loops. If your
          loop is arbitrary rotated, then you have to model all four sides of
          the loop and sum it up.


    Parameters
    ----------
    moment : str {'lm', 'hm'}
        Moment. If 'lm', above defined ``lm_off_time``, ``lm_waveform_times``,
        and ``lm_waveform_current`` are used. Else, the corresponding
        ``hm_``-parameters.

    depth : ndarray
        Depths of the resistivity model (see ``empymod.model.bipole`` for more
        info.)

    res : ndarray
        Resistivities of the resistivity model (see ``empymod.model.bipole``
        for more info.)

    Returns
    -------
    WalkTEM : EMArray
        WalkTEM response (dB/dt).

    """

    # Get the measurement time and the waveform corresponding to the provided
    # moment.

    # Off times (when measurement happens)
    # off_time = np.array([
    #     9.810e-05, 1.216e-04, 1.506e-04, 1.876e-04, 2.341e-04, 2.921e-04,
    #     3.656e-04, 4.581e-04, 5.746e-04, 7.211e-04, 9.056e-04, 1.138e-03,
    #     1.431e-03, 1.799e-03, 2.262e-03, 2.846e-03, 3.580e-03, 4.505e-03,
    #     5.670e-03, 7.135e-03
    # ])
    off_time = np.logspace(time_in[0], time_in[1], int(time_in[2]))
    if waveflag == 1:
        waveform_times = np.r_[-8.333E-03, -8.033E-03, 0.000E+00, 5.600E-06]
        waveform_current = np.r_[0.0, 1.0, 1.0, 0.0]
        plt.figure()
        plt.title('Waveforms')
        plt.plot(np.r_[-9, waveform_times*1e3, 2], np.r_[0, waveform_current, 0],
                '-.', label='Moment')
        plt.xlabel('Time (ms)')
        plt.xlim([-9, 0.5])
        plt.legend()
        plt.show()

    # === GET REQUIRED TIMES ===
    if waveflag == 1:
        time = get_time(off_time, waveform_times)
    else:
        time = off_time

    # === GET REQUIRED FREQUENCIES ===
    time, freq, ft, ftarg = empymod.utils.check_time(
        time=time,          # Required times
        signal=signal,          # 1 Switch-on response; -1 Switch-off response; 0 Impulse response
        ft='dlf',           # Use DLF
        ftarg={'dlf': 'key_201_CosSin_2012'},  # 'key_81_CosSin_2009' Short, fast filter; 
        verb=2,                 # if you need higher accuracy choose a longer filter.
    )

    # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
    # We only define a few parameters here. You could extend this for any
    # parameter possible to provide to empymod.model.bipole.
    EM = empymod.model.bipole(
        src=src,                    # El. bipole source; half of one side.
        rec=rec,                    # Receiver at the origin, vertical.
        depth=depth,                # Depth-model, adding air-interface.
        res=res,                    # Provided resistivity model, adding air.
        # aniso=aniso,                # Here you could implement anisotropy...
        #                             # ...or any parameter accepted by bipole.
        freqtime=freq,                # Required frequencies.
        mrec=mrec,                    # It is an el. source, but a magn. rec.
        strength=strength,            # To account for 4 sides of square loop.
        srcpts=srcpts,                # Approx. the finite dip. with 3 points.
        verb=3,                       # Verbosity.
        htarg={'dlf': 'key_201_2012'}, # 'key_101_2009'Short filter, so fast.
    )
    # Note: If the receiver wouldn't be in the center, we would have to model
    # the actual complete loop (no symmetry to take advantage of).
    #
    #     EM = empymod.model.bipole(
    #         src=[[20, 20, -20, -20],  # x1
    #              [20, -20, -20, 20],  # x2
    #              [-20, 20, 20, -20],  # y1
    #              [20, 20, -20, -20],  # y2
    #              0, 0],               # z1, z2
    #         strength=1,
    #         # ... all other parameters remain the same
    #     )
    #     EM = EM.sum(axis=1)  # Sum all source bipoles
    if EM.ndim > 1:
        EM = EM.sum(axis=1)  # Sum all source bipoles

    # Multiply the frequecny-domain result with
    # \mu for H->B, and i\omega for B->dB/dt.
    if caltype == 'dB':
        EM *= 2j*np.pi*freq*4e-7*np.pi
    elif caltype == 'B':    
        EM *= 4e-7*np.pi
    elif caltype == 'dE':
        EM *= 2j*np.pi*freq
    

    # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
    # Note: Here we just apply one filter. But it seems that WalkTEM can apply
    #       two filters, one before and one after the so-called front gate
    #       (which might be related to ``delay_rst``, I am not sure about that
    #       part.)
    cutofffreq = 4.5e5               # As stated in the WalkTEM manual
    h = (1+1j*freq/cutofffreq)**-1   # First order type
    h *= (1+1j*freq/3e5)**-1
    EM *= h

    # === CONVERT TO TIME DOMAIN ===
    delay_rst = 1.8e-7               # As stated in the WalkTEM manual
    EM, _ = empymod.model.tem(EM[:, None], np.array([1]),
                              freq, time+delay_rst, 1, ft, ftarg)
    EM = np.squeeze(EM)

    # === APPLY WAVEFORM ===
    if waveflag == 1:
        EM = waveform(time, EM, off_time, waveform_times, waveform_current)     
    
    return off_time,EM

##########################-----MAIN-----####################################
# 按行读取参数文件并忽略该行#后的文件说明
file_path = "model.txt"  # 替换为实际的文件路径
with open(file_path, 'r',encoding='utf-8') as file:
    lines = file.readlines()
data = []
for line in lines:
    line = line.strip()
    if line:
        if line[0] == '#':
            continue
        else:
            values = line.split(',')
            data.append(values)    
# 将model转换为numpy数组
model = np.array([np.array(row, dtype=np.float64) for row in data[:-2]], dtype=object)
src=[
model[0],         # x1
model[1],         # x2
model[2],         # y1
model[3],         # y2
model[4],         # z1
model[5]]         # z2
rec = model[-8]
depth = model[-7]
res = model[-6]
srcpts = int(model[-5])
time_in = model[-4]
strength = model[-3]
signal = model[-2]
waveflag = model[-1]

caltype = data[-2][0]
if caltype == 'E' or caltype == 'dE':     # 计算电场响应
    mrec = False
elif caltype == 'B' or caltype == 'dB':   # 计算磁场响应
    mrec = True
else:
    print('caltype error!please reset')  # 计算电场响应
    exit()
outprefix = data[-1][0]
# 输出时间和结果到指定目录文件
off_time,EMfield = walktem(time_in, waveflag, signal, src, rec, depth, res, mrec, strength, srcpts, caltype)
EM_out = np.column_stack((off_time, EMfield))
np.savetxt( outprefix + '.dat', EM_out, fmt='%.6e', delimiter=',')