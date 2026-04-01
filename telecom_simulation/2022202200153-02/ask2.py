import random
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np
import itertools
from scipy.signal import fftconvolve

def fix_h(sps, n):
    h = np.zeros(sps * n)
    h[:sps] = 1 / sps
    return h

def get_power(pulses, h, sps, n):
    y = fftconvolve(pulses, h, mode='full')[:sps*n]
    d = y[sps-1::sps]
    print(np.mean(d**2))

def get_errors_no_noise(pulses, h, sps, n):
    y = fftconvolve(pulses, h, mode='full')[:sps*n]
    received_pulses = y[sps-1::sps]
    received_symbols = np.where(received_pulses >= 0, 1, -1)
    received_bits = (received_symbols == 1).astype(int)
    errors = np.sum(bits != received_bits)
    print(f"Erros without noise: {errors}")

def question_3(h, q, sps, n):
    snr = 10**(0/10)
    noise = np.sqrt(1/snr)*q
    y = fftconvolve(noise, h, mode='full')[:sps*n]
    y_samples = y[sps-1::sps]
    print(f"Noise Power: {np.mean(y_samples**2)}")

def question_4(pulses, h, sps, q):
    m = 10
    snr = 10**(0/10)
    noise = np.sqrt(1/snr)*q
    s_n = pulses + noise
    y = fftconvolve(pulses, h, mode='full')[:sps*m]
    y_noise = fftconvolve(s_n, h, mode='full')[:sps*m]

    t = np.arange(m*sps)

    plt.figure(figsize=(10,5))
    plt.plot(t, y, label="Χωρίς θόρυβο", linewidth=2)
    plt.plot(t, y_noise, label="Με θόρυβο", linestyle='--')

    for i in range(m):
        plt.axvline(i*sps, color='gray', alpha=0.3)

    plt.title("Πρώτα 10 σύμβολα")
    plt.xlabel("Δείγματα")
    plt.ylabel("Τιμή σήματος")
    plt.legend()
    plt.grid()
    plt.show()

def question_5(bits, pulses, h, sps, n, q):
    snr = 10**(0/10)
    noise = np.sqrt(1/snr)*q
    s_n = pulses + noise
    y = fftconvolve(s_n, h, mode='full')[:sps*n]
    received_pulses = y[sps-1::sps]
    received_symbols = np.where(received_pulses >= 0, 1, -1)
    received_bits = (received_symbols == 1).astype(int)
    errors = np.sum(bits != received_bits)
    print(f"Πλήθος σμαλμάτων {errors}")
    
def question_6(pulses, h, sps, n, q):
    snr = 10**(12.5/10)
    noise = np.sqrt(1/snr)*q
    s_n = pulses + noise
    y = fftconvolve(s_n, h, mode='full')[:sps*n]
    received_pulses = y[sps-1::sps]

    I = received_pulses[:100]
    Q = np.zeros_like(I)

    plt.figure(figsize=(5,5))
    plt.scatter(I, Q, alpha=0.7)

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    plt.title("Constellation Diagram")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid()
    plt.show()

def question_7(sps):
    h = np.zeros(sps * 3)
    h[:sps] = 1 / sps 
    bit_combinations = list(itertools.product([0,1], repeat=3))
    axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)[1]

    for i, Nsym in enumerate([1,2,3]):
        ax = axes[i]
        for bits in bit_combinations:
            bits = np.array(bits)
            symbols = 2*bits - 1
            x = np.repeat(symbols, sps)
            y = np.convolve(x, h)
            segment = y[:Nsym*spers]
            t = np.arange(len(segment)) / sps
            ax.plot(t, segment, alpha=0.7)
        
        ax.set_ylabel("Πλάτος")
        ax.grid()
        ax.set_title(f"{Nsym} Σύμβολο(-α)")

    axes[-1].set_xlabel("Χρόνος (σε σύμβολα)")
    plt.tight_layout()
    plt.show()

def question_8(pulses, h, sps, n, q):
    snr = 10**(12.5/10)
    noise = np.sqrt(1/snr)*q
    s_n = pulses + noise
    y = fftconvolve(s_n, h, mode='full')[:100]
    L = 3 * sps
    plt.figure(figsize=(8,5))
    for i in range(0, len(y) - L, sps):
        segment = y[i:i+L]
        plt.plot(segment.real, color='blue', alpha=0.3)

    plt.title("Eye Diagram for N=1000")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.show()

def question_9(pulses, symbols, h, sps, n, q, snr_db):
    pse = []
    pbe = []
    for i in np.linspace(-5, 12.5, 8):
        snr = 10**(i/10)
        noise = np.sqrt(1/snr)*q
        s_n = pulses + noise
        y = fftconvolve(s_n, h, mode='full')[:sps*n]
        received_pulses = y[sps-1::sps]
        received_symbols = np.where(received_pulses >= 0, 1, -1)
        
        errors = np.sum(symbols != received_symbols)
        pse.append(errors/n)
        pbe.append(float(norm.sf(np.sqrt(snr))))

    plt.figure(figsize=(8,5))
    plt.semilogy(snr_db, pbe, '-', label="Theoretical BER")
    pse = np.maximum(pse, 1e-6)
    plt.semilogy(snr_db, pse, '*', label="Simulated SER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Error")
    plt.title("SER vs SNR")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

def question_10(pulses, bits, h, sps, n, q, snr_db):
    be = []
    pbe = []
    for i in np.linspace(-5, 12.5, 8):
        snr = 10**(i/10)
        noise = np.sqrt(1/snr)*q
        s_n = pulses + noise
        y = fftconvolve(s_n, h, mode='full')[:sps*n]
        received_pulses = y[sps-1::sps]
        received_symbols = np.where(received_pulses >= 0, 1, -1)
        received_bits = (received_symbols == 1).astype(int)
        errors = np.sum(bits != received_bits)
        be.append(errors/n)
        pbe.append(float(norm.sf(np.sqrt(snr))))

    plt.figure(figsize=(8,5))
    plt.semilogy(snr_db, pbe, '-', label="Theoretical BER")
    be = np.maximum(be, 1e-6)
    plt.semilogy(snr_db, be, '*', label="Simulated BER")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Probability of Error")
    plt.title("BER vs SNR")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

n = 100
spers = 4
bits = np.random.randint(0, 2, size=n)
symbols = 2*bits -1
pulses = np.repeat(symbols, spers)
pulses = pulses.astype(np.float32)
h = fix_h(spers, n)
h = h.astype(np.float32)
q = np.random.normal(size=spers*n).astype(np.float32)
snr_db = np.arange(-5, 14, 2.5)

# get_power(pulses, h, spers, n)
# get_errors_no_noise(pulses, h, spers, n)
# question_3(h, q, spers, n)
# question_4(pulses, h, spers, q)
# question_5(bits, pulses, h, spers, n, q)
# question_6(pulses, h, spers, n, q)
# question_7(spers)
# question_8(pulses, h, spers, n, q)
# question_9(pulses, symbols, h, spers, n, q, snr_db)
# question_10(pulses, bits, h, spers, n, q, snr_db)