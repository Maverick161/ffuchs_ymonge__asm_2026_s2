"""
Experimento: timing.py
----------------------
Objetivo: Orquestar el flujo de medición de tiempos para demostrar empíricamente 
la diferencia entre O(N^2) y O(N log N) para potencias de 2 crecientes.
"""

import os
import csv
import numpy as np

from radar.signals.generators import generate_sine
from radar.transforms.dft import compute_dft
from radar.transforms.fft import compute_fft
from radar.analysis.timing import benchmark_algorithm
from radar.visualization.plots import plot_execution_times

def run_timing_experiment():
    print("Iniciando Benchmark de Tiempos (DFT vs FFT)...\n")
    
    # Tamaños a evaluar (potencias de 2). Limitamos a 2048 porque 
    # la DFT propia en Python puro se vuelve insoportablemente lenta después.
    N_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
    
    dft_avg_times = []
    fft_avg_times = []
    csv_data = []
    
    for N in N_values:
        print(f"Perfilando N = {N}...")
        
        # 1. Generar una señal aleatoria simulada de longitud N
        # La frecuencia y sample_rate no importan para el benchmark de procesamiento
        signal = generate_sine(frequency=10, sample_rate=N, duration=1.0)
        data = signal.samples
        
        # 2. Benchmarks (10 repeticiones por cada tamaño para promediar)
        res_dft = benchmark_algorithm("DFT", compute_dft, data, repetitions=10)
        res_fft = benchmark_algorithm("FFT", compute_fft, data, repetitions=10)
        
        # 3. Almacenar datos para gráficas
        dft_avg_times.append(res_dft.average_time)
        fft_avg_times.append(res_fft.average_time)
        
        # 4. Preparar fila para el CSV
        csv_data.append({
            'N': N, 
            'DFT_Avg_Time_s': f"{res_dft.average_time:.6f}", 
            'FFT_Avg_Time_s': f"{res_fft.average_time:.6f}"
        })

    # --- GUARDAR DATOS EN CSV ---
    csv_path = "results/timing/dft_fft_benchmark.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['N', 'DFT_Avg_Time_s', 'FFT_Avg_Time_s'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nDatos numéricos exportados a: {csv_path}")

    # --- GENERAR GRÁFICA ---
    img_path = "results/figures/timing/benchmark_timing_comparativo.png"
    plot_execution_times(N_values, dft_avg_times, fft_avg_times, filepath=img_path)
    
    print(f"Gráfica exportada a: {img_path}")
    print("¡Experimento completado exitosamente!")

if __name__ == "__main__":
    run_timing_experiment()