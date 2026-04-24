import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def RRC(sps, gDelay, r):
    """
    Generates a Root Raised Cosine (RRC) pulse shaping filter.

    Parameters
    ----------
    sps : int
        Samples per symbol (oversampling factor). Defines how many samples
        represent one symbol period.

    gDelay : int
        Group delay of the filter in symbol durations. The total filter length
        will be (gDelay * sps + 1) samples, meaning the pulse spans gDelay symbols.

    r : float
        Roll-off factor (0 <= r <= 1). Controls the excess bandwidth of the filter:
        - r = 0   → ideal sinc (brick-wall spectrum)
        - r = 1   → maximum bandwidth expansion

    Returns
    -------
    hpulse : numpy.ndarray
        The impulse response of the Root Raised Cosine filter (1D array of length ntaps).
        This can be used for pulse shaping at the transmitter or matched filtering
        at the receiver in digital communication systems.
    """

    ntaps = gDelay * sps + 1  # Total number of filter taps

    # Time axis in symbol durations (centered at 0)
    st = np.arange(-np.floor(ntaps/2), np.floor(ntaps/2) + 1) / sps

    # Compute the RRC formula (vectorized)
    numerator = np.sin((1 - r) * np.pi * st) + 4 * r * st * np.cos((1 + r) * np.pi * st)
    denominator = np.pi * st * (1 - (4 * r * st) ** 2)

    hpulse = numerator / denominator
    # hpulse = np.zeros_like(st)

    # valid = np.abs(denominator) > 1e-10
    # hpulse[valid] = numerator[valid] / denominator[valid]

    # Handle the singularity at t = 0 (center tap)
    center_idx = int(np.ceil(ntaps / 2)) - 1  # Adjust for 0-based indexing
    hpulse[center_idx] = 1 - r + 4 * r / np.pi

    # Handle singularities at t = ±1/(4r), where denominator → 0
    sing_idx = np.where(np.abs(1 - (4 * r * st) ** 2) < 1e-8)[0]

    if len(sing_idx) > 0:
        value = (r / np.sqrt(2)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * r)) +
            (1 - 2 / np.pi) * np.cos(np.pi / (4 * r))
        )
        hpulse[sing_idx] = value

    return hpulse

def delta(bit, samples):
    p = [0] * samples
    if(bit ==  1):
        p[0] = 1
    elif(bit == 0):
        p[0] = -1
    return p

def get_bits_delta(bits, sps):
    signal = []
    for b in bits:
        signal.extend(delta(b, sps))
    return signal

def get_received_bits(y):
    b = []
    for i in range(len(y)):
        if(y[i] >= 0):
            b.append(1)
        else:
            b.append(0)
    return b

def question_1(bits, sps, rrc, n, start):
    signal = np.array(get_bits_delta(bits, sps))
    y = np.convolve(signal, rrc)
    y = y[:n*sps]

    y_received = np.convolve(y, rrc/sps)
    y_received = y_received[:n*sps]
    ds = y_received[start::sps]
    ds = ds[14:]
    power = np.mean(ds**2)
    print("Signal power:", power)

def question_2(sps, rrc, n, start):
    snr = 10**(5 / 10)
    noise = np.sqrt(1/snr) * np.random.normal(size=n*sps)
    y_noise = np.convolve(noise, rrc/sps)
    y_noise = y_noise[:n*sps]
    ds_noise = y_noise[start::sps]
    ds_noise = ds_noise[14:]
    power = np.mean(ds_noise**2)
    print("Noise power K:", power)

def question_3(bits, sps, rrc):
    signal = np.array(get_bits_delta(bits, sps))
    tx = np.convolve(signal, rrc)

    N = 16 * sps
    signal_cut = signal[:N]
    delay = (len(rrc) - 1) // 2
    tx = tx[delay:]

    tx_cut = tx[:N]
    print(bits[:16])
    t = np.arange(N)

    plt.figure(figsize=(10, 5))
    plt.title("Transmit Pulses and Impulses")
    plt.stem(t, signal_cut, linefmt='r-', markerfmt='ro', basefmt='k-', label="Impulses")
    plt.plot(t, tx_cut, 'b', label="Tx Signal (RRC shaped)")

    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def question_4(bits, sps, rrc, n):
    snr = 10**(11 / 10)
    signal = np.array(get_bits_delta(bits, sps))

    y = np.convolve(signal, rrc)
    y = y[:n*sps]
    noise = np.sqrt(1/snr) * np.random.normal(size=len(y))

    r_noisy = y + noise
    r_clean = y
    y_clean = np.convolve(r_clean, rrc/sps)
    y_noisy = np.convolve(r_noisy, rrc/sps)

    y_clean = y_clean[14*sps:]
    y_noisy = y_noisy[14*sps:]
    print(bits[:16])
    N = 16 * sps
    t = np.arange(N)
    impulses = np.array(signal[:N])
    rx_clean = y_clean[:N]
    rx_noisy = y_noisy[:N]

    plt.figure(figsize=(10,5))

    plt.stem(t, impulses, linefmt='r-', markerfmt='ro', basefmt='k-', label="Impulses")
    plt.plot(t, rx_clean, 'b', label="Rx without noise")
    plt.plot(t, rx_noisy, 'g', alpha=0.7, label="Rx with noise (11 dB)")

    plt.title("Received Pulses (with and without noise)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.show()

def question_5(bits, sps, rrc, n):
    signal = np.array(get_bits_delta(bits, sps))

    y = np.convolve(signal, rrc)
    y = y[:n*sps]
    y_r = np.convolve(y, rrc/sps)
    y_r = y_r[:n*sps]
    total_delay = 2 * 14 * sps
    y_r = y_r[total_delay:]
    y_r = y_r[:1000]
    print(len(y_r))
    L = 2 * sps
    plt.figure(figsize=(8,5))
    for i in range(0, len(y_r) - L, sps):
        segment = y_r[i:i+L]
        plt.plot(segment.real, color='blue', alpha=0.3)

    plt.title("Eye Diagram (No Noise) - First 100 Bits")
    plt.xlabel("Samples per Symbol")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def question_6(bits, sps, rrc, n):
    signal = np.array(get_bits_delta(bits, sps))

    y = np.convolve(signal, rrc)
    y = y[:n*sps]
    snr = 10**(11/10)
    noise = np.sqrt(1/snr) * np.random.normal(size=len(y))
    r = y + noise
    y_r = np.convolve(r, rrc/sps)
    y_r = y_r[:n*sps]
    total_delay = 2 * 14 * sps
    y_r = y_r[total_delay:] 
    y_r = y_r[:1000]
    L = 2 * sps
    plt.figure(figsize=(8,5))
    for i in range(0, len(y_r) - L, sps):
        segment = y_r[i:i+L]
        plt.plot(segment.real, color='blue', alpha=0.3)

    plt.title("Eye Diagram (With Noise) - First 100 Bits")
    plt.xlabel("Samples per Symbol")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def question_7(bits, sps, rrc, n, start):
    signal = np.array(get_bits_delta(bits, sps))
    rrc = rrc / np.sqrt(np.sum(rrc**2))
    y = np.convolve(signal, rrc)
    snr = 10**(11/10)
    y = y[:n*sps]
    noise = np.sqrt(1/snr) * np.random.normal(size=len(y))
    r = y + noise
    y_r = np.convolve(r, rrc)

    total_delay = 2 * 14 * sps
    y_r = y_r[total_delay:]
    ds = y_r[start::sps]

    ds = ds[:100]

    plt.figure(figsize=(6,6))
    plt.scatter(ds, np.zeros_like(ds), alpha=0.7)

    plt.title("Constellation Diagram (SNR = 11 dB)")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid()
    plt.axhline(0)
    plt.axvline(0)
    plt.show()

def question_8_10(bits, sps, snr_db, rrc, n, start):
    pbe = np.zeros(len(snr_db))
    ber_theory = np.zeros(len(snr_db))
    signal = np.array(get_bits_delta(bits, sps))
    rrc = rrc / np.sqrt(np.sum(rrc**2))
    y = np.convolve(signal, rrc)
    y = y[:n*sps]

    for idx, snr_dB in enumerate(snr_db):
        snr = 10**(snr_dB/10)
        noise = np.sqrt(1/snr) * np.random.normal(size=len(y))
        r = y + noise
        y_r = np.convolve(r, rrc/sps)
        total_delay = 2 * 14 * sps

        y_r = y_r[total_delay:]
        ds = y_r[start::sps]

        received_bits = (ds >= 0).astype(int)
        bits_valid = bits[28:28+len(received_bits)]
        errors = bits_valid != received_bits[:len(bits_valid)]
        pbe[idx] = np.mean(errors)
        ber_theory[idx] = norm.sf(np.sqrt(snr))

    plt.figure(figsize=(8,5))
    plt.semilogy(snr_db, ber_theory, '-', label="Theoretical BER")
    pbe = np.maximum(pbe, 1e-6)
    plt.semilogy(snr_db, pbe, '*', label="Simulated BER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Error")
    plt.title("BER vs SNR")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
n = 3000
spers = 11
start = 14*spers
bits = np.random.randint(0, 2, size=n)
snr_db = np.arange(-5, 14, 2.5)
rrc = RRC(spers, 14, 0.22)

# question_1(bits, spers, rrc, n, start)
# question_2(spers, rrc, n, start)
# question_3(bits, spers, rrc)
# question_4(bits, spers, rrc, n)
# question_5(bits, spers, rrc, n)
# question_6(bits, spers, rrc, n)
# question_7(bits, spers, rrc, n, start)
# question_8_10(bits, spers, snr_db, rrc, n, start)
