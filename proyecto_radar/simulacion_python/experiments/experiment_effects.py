"""
Experimento: experiment_effects.py
----------------------------------
Objetivo: Generar gráficas demostrativas de fenómenos clásicos de DSP
(Aliasing, Spectral Leakage y Zero Padding) requeridas para la documentación
del radar acústico.
"""

import os
import numpy as np

from radar.signals.generators import generate_sine
from radar.transforms.fft import compute_fft
from radar.analysis.spectrum import compute_spectrum
from radar.utils.validation import pad_to_power_of_two
from radar.visualization.plots import plot_time_domain, plot_magnitude_spectrum

def run_aliasing_experiment():
    """
    Demuestra la violación del teorema de Nyquist.
    Fs = 1000 Hz (Nyquist = 500 Hz). Inyectamos una señal de 700 Hz.
    Debería aparecer un alias o frecuencia fantasma en |1000 - 700| = 300 Hz.
    """
    print("Ejecutando experimento de Aliasing...")
    fs = 1000.0
    N = 1024
    duration = N / fs
    
    # Generamos una frecuencia de 700 Hz (ilegal para Fs=1000)
    signal = generate_sine(frequency=700.0, sample_rate=fs, duration=duration)
    spectrum = compute_spectrum(signal, compute_fft(signal.samples))
    
    out_dir = "results/figures/aliasing"
    plot_magnitude_spectrum(spectrum, filepath=f"{out_dir}/espectro_aliasing_700hz.png")

def run_leakage_experiment():
    """
    Demuestra el "goteo" de energía cuando una frecuencia no coincide exactamente
    con un bin (intervalo) de la FFT.
    Con Fs=1024 y N=1024, los bins están separados por exactamente 1 Hz.
    """
    print("Ejecutando experimento de Spectral Leakage...")
    fs = 1024.0
    N = 1024
    duration = N / fs
    out_dir = "results/figures/leakage"

    # Caso 1: Frecuencia entera (50 Hz). Coincide perfecto con el bin 50.
    sig_perfect = generate_sine(frequency=50.0, sample_rate=fs, duration=duration)
    spec_perfect = compute_spectrum(sig_perfect, compute_fft(sig_perfect.samples))
    plot_magnitude_spectrum(
        spec_perfect, 
        filepath=f"{out_dir}/leakage_01_perfect_50hz.png", 
        zoom_freq=100
    )
    
    # Caso 2: Frecuencia fraccionaria (50.5 Hz). Cae entre el bin 50 y 51.
    sig_leaking = generate_sine(frequency=50.5, sample_rate=fs, duration=duration)
    spec_leaking = compute_spectrum(sig_leaking, compute_fft(sig_leaking.samples))
    plot_magnitude_spectrum(
        spec_leaking, 
        filepath=f"{out_dir}/leakage_02_leaking_50_5hz.png", 
        zoom_freq=100
    )

def run_zero_padding_experiment():
    """
    Demuestra cómo el relleno con ceros aumenta artificialmente los puntos 
    de la FFT (interpolación visual) pero no mejora la resolución física real.
    """
    print("Ejecutando experimento de Zero Padding...")
    fs = 1024.0
    # Señal original muy corta: N=64. Delta_f = 1024/64 = 16 Hz (Resolución pobre)
    # Frecuencias a 40 Hz y 50 Hz (están a menos de 16 Hz de distancia, se mezclarán)
    duration_short = 64 / fs
    
    # Generamos señal compuesta (Suma de dos senos)
    t = np.arange(64) / fs
    samples = np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 50 * t)
    
    from radar.signals.signal import Signal
    sig_short = Signal(samples=samples, sample_rate=fs, name="Senos_40_50")
    
    out_dir = "results/figures/zero_padding"
    
    # 1. FFT de la señal corta original (N=64)
    spec_short = compute_spectrum(sig_short, compute_fft(sig_short.samples))
    plot_magnitude_spectrum(spec_short, filepath=f"{out_dir}/zp_01_corta_N64.png", zoom_freq=150)
    
    # 2. FFT aplicando Zero Padding hasta N=1024
    # El algoritmo rellena con 960 ceros al final de las 64 muestras
    sig_padded = pad_to_power_of_two(sig_short) # Primero intentaría N=64, debemos forzar N=1024
    
    # Forzamos manualmente el padding a 1024 para el experimento
    padded_samples = np.concatenate((sig_short.samples, np.zeros(1024 - 64)))
    sig_forced_pad = Signal(samples=padded_samples, sample_rate=fs, name="Padded_N1024")
    
    spec_padded = compute_spectrum(sig_forced_pad, compute_fft(sig_forced_pad.samples))
    plot_magnitude_spectrum(spec_padded, filepath=f"{out_dir}/zp_02_rellena_N1024.png", zoom_freq=150)

if __name__ == "__main__":
    print("--- INICIANDO GENERACIÓN DE EFECTOS DSP ---")
    run_aliasing_experiment()
    run_leakage_experiment()
    run_zero_padding_experiment()
    print("--- TODAS LAS GRÁFICAS HAN SIDO GENERADAS EN results/figures/ ---")