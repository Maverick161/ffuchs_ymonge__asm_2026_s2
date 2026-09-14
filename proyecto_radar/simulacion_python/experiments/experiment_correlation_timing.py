"""
Experimento: experiment_correlation_timing.py
---------------------------------------------
Objetivo: Comparar el rendimiento computacional entre la correlación directa O(N^2)
y la correlación por FFT O(N log N) utilizando la mediana de múltiples ejecuciones.
"""
import time
import os
import numpy as np

from radar.signals.signal import Signal
from radar.analysis.correlation import cross_correlate_direct, cross_correlate_fft
from radar.visualization.plots import plot_execution_times

def run_timing_benchmark():
    print("--- Experimento: Benchmark de Correlación (Robusto) ---")
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    repetitions = 10  # Recomendación 10: reducir ruido del OS mediante repeticiones
    
    direct_times_median = []
    fft_times_median = []
    
    rng = np.random.default_rng(2026)
    
    for n in sizes:
        print(f"[*] Evaluando N = {n} ({repetitions} repeticiones)...")
        d_times = []
        f_times = []
        
        for _ in range(repetitions):
            tx = Signal(samples=rng.random(n), sample_rate=1000.0)
            rx = Signal(samples=rng.random(n * 2), sample_rate=1000.0)
            
            # Medir Directa
            start = time.perf_counter()
            cross_correlate_direct(tx, rx)
            d_times.append(time.perf_counter() - start)
            
            # Medir FFT
            start = time.perf_counter()
            cross_correlate_fft(tx, rx)
            f_times.append(time.perf_counter() - start)
            
        direct_times_median.append(np.median(d_times))
        fft_times_median.append(np.median(f_times))
        
    out_dir = "results/figures/timing"
    os.makedirs(out_dir, exist_ok=True)
    filepath = f"{out_dir}/benchmark_correlacion_comparativo.png"
    
    plot_execution_times(sizes, 
        direct_times_median, 
        fft_times_median, 
        filepath=filepath,
        label_1=r"Correlación Directa $\mathcal{O}(N^2)$",
        label_2=r"Correlación FFT $\mathcal{O}(N \log N)$",
        title="Comparativa de Rendimiento: Correlación Temporal vs Espectral",
        ylabel="Tiempo Mediano de Ejecución (s)")
    
    print("\n[+] Resultados de Tiempos (Mediana en segundos):")
    print(f"{'N':<8} | {'Directa O(N^2)':<15} | {'FFT O(N log N)':<15}")
    print("-" * 45)
    for i, n in enumerate(sizes):
        print(f"{n:<8} | {direct_times_median[i]:<15.6f} | {fft_times_median[i]:<15.6f}")
        
    print(f"\n[+] Gráfica de rendimiento guardada en {filepath}")

if __name__ == "__main__":
    run_timing_benchmark()