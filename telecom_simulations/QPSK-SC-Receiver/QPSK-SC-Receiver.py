import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import quad

def get_symbols(b):
    b = np.asarray(b)
    val = 1 / np.sqrt(2)

    pairs = b.reshape(-1, 2)
    symbols = np.empty(len(pairs), dtype=complex)

    mask00 = (pairs[:,0] == 0) & (pairs[:,1] == 0)
    mask01 = (pairs[:,0] == 0) & (pairs[:,1] == 1)
    mask10 = (pairs[:,0] == 1) & (pairs[:,1] == 0)
    mask11 = (pairs[:,0] == 1) & (pairs[:,1] == 1)

    symbols[mask00] = complex(+val, +val)
    symbols[mask01] = complex(-val, +val)
    symbols[mask10] = complex(+val, -val)
    symbols[mask11] = complex(-val, -val)

    return symbols

def get_rayleigh(length):
    value = 1/np.sqrt(2)

    re = np.random.normal(0, value, length)
    im = np.random.normal(0, value, length)

    return re + 1j*im

def get_noise(pn, n):
    value = 1/np.sqrt(2)

    re = np.random.normal(0, 1, n)
    im = np.random.normal(0, 1, n)

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
    
    r_bits = np.zeros(2 * N, dtype=int)
    
    mask = (symbols.real > 0) & (symbols.imag > 0)
    r_bits[2*mask.nonzero()[0]] = 0
    r_bits[2*mask.nonzero()[0]+1] = 0

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

def scSER(snr, L): #snr: SNR in linear scale, L: Number of antennas
    ser = 0.0
    for l in range(L):  # 0 to L-1
        denom = l + 1
        mu = math.sqrt((snr / denom) / (2 + (snr / denom)))
        ser += ((-1)**l) * math.comb(L - 1, l) / denom * ((1 - mu) / 2)
    return ser * 2 * L

def question_2(ray1, ray2, symb):
    snr = 10**(15/10)
    N = len(symb)
    rays = np.vstack([ray1, ray2])

    ys = np.vstack([
        symb * ray1 + get_noise(1/snr, N),
        symb * ray2 + get_noise(1/snr, N)
    ])

    idx_best = np.argmax(np.abs(rays), axis=0)

    selected_y = ys[idx_best, np.arange(N)]
    selected_h = rays[idx_best, np.arange(N)]

    selected = selected_y / selected_h

    detected = detector(selected)

    ser = np.mean(symb != detected)
    print(f"SER for L=2: {ser}")
    
def question_3(ray1, ray2, ray3, ray4, symb):
    snr = 10**(10/10)
    N = len(symb)

    rays = np.vstack([ray1, ray2, ray3, ray4])

    ys = np.vstack([
        symb * ray1 + get_noise(1/snr, N),
        symb * ray2 + get_noise(1/snr, N),
        symb * ray3 + get_noise(1/snr, N),
        symb * ray4 + get_noise(1/snr, N)
    ])

    idx_best = np.argmax(np.abs(rays), axis=0)

    selected_y = ys[idx_best, np.arange(N)]
    selected_h = rays[idx_best, np.arange(N)]

    selected = selected_y / selected_h

    detected = detector(selected)

    ser = np.mean(symb != detected)
    print(f"SER for L=4: {ser}")

def question_4(ray1, ray2, ray3, ray4, symb, db):
    theoretical_ser1 = np.zeros(len(db))
    theoretical_ser2 = np.zeros(len(db))
    theoretical_ser4 = np.zeros(len(db))
    simulation_ser1 = np.zeros(len(db))
    simulation_ser2 = np.zeros(len(db))
    simulation_ser4 = np.zeros(len(db))
    rays4 = np.vstack([ray1, ray2, ray3, ray4])
    rays2 = np.vstack([ray1, ray2])
    N = len(symb)

    idx_best2 = np.argmax(np.abs(rays2), axis=0)
    idx_best4 = np.argmax(np.abs(rays4), axis=0)

    for idx, i in enumerate(db):
        snr = 10**(i/10)
        ys1 = symb * ray1 + get_noise(1/snr, N)

        ys2 = np.vstack([
            symb * ray1 + get_noise(1/snr, N),
            symb * ray2 + get_noise(1/snr, N)
        ])

        ys4 = np.vstack([
            symb * ray1 + get_noise(1/snr, N),
            symb * ray2 + get_noise(1/snr, N),
            symb * ray3 + get_noise(1/snr, N),
            symb * ray4 + get_noise(1/snr, N)
        ])

        selected_y2 = ys2[idx_best2, np.arange(N)]
        selected_h2 = rays2[idx_best2, np.arange(N)]
        selected_y4 = ys4[idx_best4, np.arange(N)]
        selected_h4 = rays4[idx_best4, np.arange(N)]

        selected1 = ys1 / ray1
        selected2 = selected_y2 / selected_h2
        selected4 = selected_y4 / selected_h4
        
        detected1 = detector(selected1)
        detected2 = detector(selected2)
        detected4 = detector(selected4)

        errors1 = np.count_nonzero(symb != detected1)
        errors2 = np.count_nonzero(symb != detected2)
        errors4 = np.count_nonzero(symb != detected4)

        simulation_ser1[idx] = errors1 / N if errors1 >= 10 else np.nan
        simulation_ser2[idx] = errors2 / N if errors2 >= 10 else np.nan
        simulation_ser4[idx] = errors4 / N if errors4 >= 10 else np.nan

        theoretical_ser1[idx] = scSER(snr, 1)
        theoretical_ser2[idx] = scSER(snr, 2)
        theoretical_ser4[idx] = scSER(snr, 4)

    plt.figure(figsize=(8,5))
    plt.semilogy(db, theoretical_ser1, '-', linewidth=2, label="Theoretical SER 1-Antenna")
    plt.semilogy(db, theoretical_ser2, '-', linewidth=2, label="Theoretical SER 2-Antennas")
    plt.semilogy(db, theoretical_ser4, '-', linewidth=2, label="Theoretical SER 4-Antennas")
    plt.semilogy(db, simulation_ser1, '*', markersize=8, label="Simulated SER 1-Antenna")
    plt.semilogy(db, simulation_ser2, '*', markersize=8, label="Simulated SER 2-Antennas")
    plt.semilogy(db, simulation_ser4, '*', markersize=8, label="Simulated SER 4-Antennas")

    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title(f"SER vs SNR for 1, 2, 4-branch SC")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()
    plt.ylim(1e-8, 1)
    plt.legend()
    plt.show()

def question_5(ray1, ray2, ray3, ray4, symb, b, db):
    theoretical_ser1 = np.zeros(len(db))
    theoretical_ser2 = np.zeros(len(db))
    theoretical_ser4 = np.zeros(len(db))
    simulation_ser1 = np.zeros(len(db))
    simulation_ser2 = np.zeros(len(db))
    simulation_ser4 = np.zeros(len(db))
    rays4 = np.vstack([ray1, ray2, ray3, ray4])
    rays2 = np.vstack([ray1, ray2])
    N = len(symb)

    idx_best2 = np.argmax(np.abs(rays2), axis=0)
    idx_best4 = np.argmax(np.abs(rays4), axis=0)

    for idx, i in enumerate(db):
        snr = 10**(i/10)
        ys1 = symb * ray1 + get_noise(1/snr, N)

        ys2 = np.vstack([
            symb * ray1 + get_noise(1/snr, N),
            symb * ray2 + get_noise(1/snr, N)
        ])

        ys4 = np.vstack([
            symb * ray1 + get_noise(1/snr, N),
            symb * ray2 + get_noise(1/snr, N),
            symb * ray3 + get_noise(1/snr, N),
            symb * ray4 + get_noise(1/snr, N)
        ])

        selected_y2 = ys2[idx_best2, np.arange(N)]
        selected_h2 = rays2[idx_best2, np.arange(N)]
        selected_y4 = ys4[idx_best4, np.arange(N)]
        selected_h4 = rays4[idx_best4, np.arange(N)]

        selected1 = ys1 / ray1
        selected2 = selected_y2 / selected_h2
        selected4 = selected_y4 / selected_h4
        
        detected1 = detector(selected1)
        detected2 = detector(selected2)
        detected4 = detector(selected4)

        r1 = get_received_bits(detected1)
        r2 = get_received_bits(detected2)
        r4 = get_received_bits(detected4)

        errors1 = np.count_nonzero(b != r1)
        errors2 = np.count_nonzero(b != r2)
        errors4 = np.count_nonzero(b != r4)

        simulation_ser1[idx] = errors1 / len(b) if errors1 >= 10 else np.nan
        simulation_ser2[idx] = errors2 / len(b) if errors2 >= 10 else np.nan
        simulation_ser4[idx] = errors4 / len(b) if errors4 >= 10 else np.nan

        theoretical_ser1[idx] = scSER(snr, 1) / 2
        theoretical_ser2[idx] = scSER(snr, 2) / 2
        theoretical_ser4[idx] = scSER(snr, 4) / 2

    plt.figure(figsize=(8,5))
    plt.semilogy(db, theoretical_ser1, '-', linewidth=2, label="Theoretical BER 1-Antenna")
    plt.semilogy(db, theoretical_ser2, '-', linewidth=2, label="Theoretical BER 2-Antennas")
    plt.semilogy(db, theoretical_ser4, '-', linewidth=2, label="Theoretical BER 4-Antennas")
    plt.semilogy(db, simulation_ser1, '*', markersize=8, label="Simulated BER 1-Antenna")
    plt.semilogy(db, simulation_ser2, '*', markersize=8, label="Simulated BER 2-Antennas")
    plt.semilogy(db, simulation_ser4, '*', markersize=8, label="Simulated BER 4-Antennas")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER vs SNR for 1, 2, 4-branch SC")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()
    plt.ylim(1e-8, 1)
    plt.legend()
    plt.show()

n = 2**20
snr_db = np.arange(-5, 51, 5)
bits = np.random.randint(0, 2, size=n)
symbols = get_symbols(bits)
rayleigh1 = get_rayleigh(len(symbols))
rayleigh2 = get_rayleigh(len(symbols))
rayleigh3 = get_rayleigh(len(symbols))
rayleigh4 = get_rayleigh(len(symbols))

question_2(rayleigh1, rayleigh2, symbols)
question_3(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols)
question_4(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols, snr_db)
question_5(rayleigh1, rayleigh2, rayleigh3, rayleigh4, symbols, bits, snr_db)