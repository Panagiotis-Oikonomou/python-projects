import random
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np
from scipy.signal import fftconvolve
import itertools

def get_symbols(b):
    b = np.asarray(b)
    val = 1 / np.sqrt(2)

    pairs = b.reshape(-1, 2)
    symbols = np.empty(len(pairs), dtype=complex)

    # create masks for each pair
    mask00 = (pairs[:,0] == 0) & (pairs[:,1] == 0)
    mask01 = (pairs[:,0] == 0) & (pairs[:,1] == 1)
    mask10 = (pairs[:,0] == 1) & (pairs[:,1] == 0)
    mask11 = (pairs[:,0] == 1) & (pairs[:,1] == 1)

    symbols[mask00] = complex(+val, +val)
    symbols[mask01] = complex(-val, +val)
    symbols[mask10] = complex(+val, -val)
    symbols[mask11] = complex(-val, -val)

    return symbols

def get_noise(pn, n, sps):
    value = 1/np.sqrt(2)
    total = sps*n//2

    re = np.random.normal(0, 1, total)
    im = np.random.normal(0, 1, total)

    noise = (re * value * np.sqrt(pn)) + 1j * (im * value * np.sqrt(pn))
    return noise

def detector(ds):
    ds = np.asarray(ds, dtype=complex)
    val = 1/math.sqrt(2)

    detected = np.zeros_like(ds, dtype=complex)

    mask = (ds.real > 0) & (ds.imag > 0)
    detected[mask] = complex(val, val)

    mask = (ds.real < 0) & (ds.imag >= 0)
    detected[mask] = complex(-val, val)

    mask = (ds.real < 0) & (ds.imag < 0)
    detected[mask] = complex(-val, -val)

    mask = (ds.real >= 0) & (ds.imag < 0)
    detected[mask] = complex(val, -val)

    return detected

def get_received_bits(symbols):
    symbols = np.asarray(symbols, dtype=complex)
    N = len(symbols)
    
    # Preallocate array for 2 bits per symbol
    r_bits = np.zeros(2 * N, dtype=int)
    
    # First quadrant: real > 0, imag > 0 → bits 00
    mask = (symbols.real > 0) & (symbols.imag > 0)
    r_bits[2*mask.nonzero()[0]] = 0  # first bit
    r_bits[2*mask.nonzero()[0]+1] = 0  # second bit

    mask = (symbols.real < 0) & (symbols.imag >= 0)
    r_bits[2*mask.nonzero()[0]] = 0
    r_bits[2*mask.nonzero()[0]+1] = 1

    mask = (symbols.real < 0) & (symbols.imag < 0)
    r_bits[2*mask.nonzero()[0]] = 1
    r_bits[2*mask.nonzero()[0]+1] = 1

    mask = (symbols.real >= 0) & (symbols.imag < 0)
    r_bits[2*mask.nonzero()[0]] = 1
    r_bits[2*mask.nonzero()[0]+1] = 0

    return r_bits

def get_power(s):
    s = np.asarray(s)
    return np.mean(np.abs(s)**2)

def show_y(y):
    # t = np.arange(len(y))

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(y.real)
    plt.title('Real Part')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(y.imag)
    plt.title('Imaginary Part')
    plt.grid()

    plt.tight_layout()
    plt.show()

def eye_diagram(signal, sps):
    L = 3 * sps
    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    for i in range(0, len(signal) - L, sps):
        segment = signal[i:i+L]
        plt.plot(segment.real, color='blue', alpha=0.3)
    plt.title("Eye Diagram - Real (I) part")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2,1,2)
    for i in range(0, len(signal) - L, sps):
        segment = signal[i:i+L]
        plt.plot(segment.imag, color='red', alpha=0.3)
    plt.title("Eye Diagram - Imag (Q) part")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def constellation(ds):
    plt.figure(figsize=(6,6))

    plt.scatter(ds.real, ds.imag, color='blue', s=50, alpha=0.7)

    # Draw axes
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.title("Constellation Diagram (QPSK)")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis('equal')

    plt.show()

def fix_h(sps, n):
    h = np.zeros(sps*n//2)
    h[:sps] = 1 / sps
    return h

def show_pulses(pulses):
    fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax_real.plot(pulses.real, 'o', color='blue', label='Real Part')
    ax_real.set_ylabel('Real')
    ax_real.grid(True)
    ax_real.legend()

    ax_imag.plot(pulses.imag, 'o', color='red', label='Imag Part')
    ax_imag.set_ylabel('Imag')
    ax_imag.set_xlabel('Sample Index')
    ax_imag.grid(True)
    ax_imag.legend()

    plt.tight_layout()
    plt.show()

def question_2(h, sps, n):
    snr = 10**(0/10)
    noise = get_noise(1/snr, n, sps)
    y = fftconvolve(noise, h, mode='full')[:sps*n//2]
    y_samples = y[sps-1::sps]
    print(f"Noise Power: {np.mean(np.abs(y_samples)**2)}")

def question_3(pulses, h, sps, n):
    num = 10
    snr = 10**(0/10)
    noise = get_noise(1/snr, n, sps)
    s_n = pulses + noise
    y = fftconvolve(pulses, h, mode='full')[:sps*n//2]
    y_noise = fftconvolve(s_n, h, mode='full')[:sps*n//2]

    m1 = np.abs(y[:num])
    m2 = np.abs(y_noise[:num])

    plt.figure(figsize=(10,5))
    plt.plot(np.arange(num), m1, 'b-o', label='Καθαρό σήμα (|y|)')
    plt.plot(np.arange(num), m2, 'r-o', label='Σήμα με θόρυβο (|y|)')

    plt.title("Μέγεθος ληφθέντος σήματος για τα πρώτα 10 σύμβολα")
    plt.xlabel("Δείγματα")
    plt.ylabel("Μέγεθος |y|")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def question_4(pulses, h, sps, n, bits):
    snr = 10**(0/10)
    noise = get_noise(1/snr, n, sps)
    s_n = pulses + noise
    y = fftconvolve(s_n, h, mode='full')[:sps*n//2]
    ds = y[sps-1::sps]

    detected = detector(ds)
    received_bits = get_received_bits(detected)
    bit_errors = bits != received_bits
    num = np.sum(bit_errors)
    print(f"There where {num} errors")

def question_5(pulses, h, sps, n):
    snr = 10**(12.5/10)
    noise = get_noise(1/snr, n, sps)
    s_n = pulses + noise
    y = fftconvolve(s_n, h, mode='full')[:sps*n//2]
    ds = y[sps-1::sps]
    ds = ds[:100]
    constellation(ds)

def question_6(h, sps):
    all_bits = list(itertools.product([0,1], repeat=6))
    plt.figure(figsize=(10,6))

    for bits in all_bits:
        bits = np.array(bits)
        
        symbols = get_symbols(bits)
        
        pulses = np.repeat(symbols, sps)
        
        y = fftconvolve(pulses, h, mode='full')[:len(pulses)]
        
        L = 3*sps
        plt.plot(y.real[:L], color='blue', alpha=0.3)
        plt.plot(y.imag[:L], color='red', alpha=0.3)
    
    plt.title("Eye Diagram για όλα τα δυνατά bit 0/1 (3 σύμβολα)")
    plt.xlabel("Δείγματα")
    plt.ylabel("Πλάτος")
    plt.grid(True)
    plt.show()

def question_7(pulses, h, sps, n):
    snr = 10**(12.5/10)
    noise = get_noise(1/snr, n, sps)
    s_n = pulses + noise
    y = fftconvolve(s_n, h, mode='full')[:sps*n//2]
    y = y[:100]
    eye_diagram(y, sps)

def question_8(symbols, pulses, h, sps, n, snr_db):
    pse = np.zeros(len(snr_db))
    ser = np.zeros(len(snr_db))
    for idx, i in enumerate(snr_db):
        snr = 10**(i/10)
        noise = get_noise(1/snr, n, sps)
        s_n = pulses + noise
        y = fftconvolve(s_n, h, mode='full')[:sps*n//2]
        ds = y[sps-1::sps]
        detected = detector(ds)
        symbol_errors = symbols != detected
        num = np.sum(symbol_errors)
        # pse.append(num/n)
        pse[idx] = num / len(symbols)
        x = np.sqrt(snr)
        q = norm.sf(x)
        ser[idx] = 2*q - q**2

    plt.figure(figsize=(8,5))
    plt.semilogy(snr_db, ser, '-', label="Theoretical SER")
    pse = np.maximum(pse, 1e-6)
    plt.semilogy(snr_db, pse, '*', label="Simulated SER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Error")
    plt.title("SER vs SNR")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def question_9(bits, pulses, h, sps, n, snr_db):
    pbe = np.zeros(len(snr_db))
    ber = np.zeros(len(snr_db))
    for idx, i in enumerate(snr_db):
        snr = 10**(i/10)
        noise = get_noise(1/snr, n, sps)
        s_n = pulses + noise
        y = fftconvolve(s_n, h, mode='full')[:sps*n//2]
        ds = y[sps-1::sps]
        detected = detector(ds)
        received_bits = get_received_bits(detected)
        bit_errors = bits != received_bits
        num = np.sum(bit_errors)
        pbe[idx] = num / len(bits)
        x = np.sqrt(snr)
        q = norm.sf(x)
        pse = 2*q - q**2
        ber[idx] = pse/2

    plt.figure(figsize=(8,5))
    plt.semilogy(snr_db, ber, '-', label="Theoretical BER")
    pbe = np.maximum(pbe, 1e-6)
    plt.semilogy(snr_db, pbe, '*', label="Simulated BER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Error")
    plt.title("BER vs SNR")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


n = 10
spers = 4
bits = np.random.randint(0, 2, size=n)
symbols = get_symbols(bits)
pulses = np.repeat(symbols, spers)
# show_pulses(pulses)
h = fix_h(spers, n)
snr_db = np.arange(-5, 14, 2.5)

# question_2(h, spers, n)
# question_3(pulses, h, spers, n)
# question_4(pulses, h, spers, n, bits)
# question_5(pulses, h, spers, n)
# question_6(h, spers)
# question_7(pulses, h, spers, n)
# question_8(symbols, pulses, h, spers, n, snr_db)
# question_9(bits, pulses, h, spers, n, snr_db)