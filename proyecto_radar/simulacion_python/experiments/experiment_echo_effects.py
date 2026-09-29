"""
Experimento: experiment_echo_effects.py
---------------------------------------
Objetivo: Generar una tabla resumen evaluando cuantitativamente el 
error de detección (tau_real vs tau_estimado) en 3 niveles de ruido,
y generar las gráficas de correlación correspondientes.
"""

import os
import numpy as np
from radar.signals.generators import generate_chirp
from radar.simulation.echoes import EchoConfig, simulate_echoes
from radar.analysis.correlation import cross_correlate_fft
from radar.analysis.detection import estimate_delay
from radar.visualization.plots import plot_correlation

def run_echo_effects():
    print("--- Experimento Final: Robustez de Detección bajo Ruido ---")
    fs = 48000.0
    duration = 0.05
    
    tx_signal = generate_chirp(2000.0, 8000.0, fs, duration)
    
    # Configuramos un retardo conocido que queremos encontrar
    true_delay_samples = 7500
    echoes = [EchoConfig(delay_samples=true_delay_samples, attenuation=0.6)]
    
    # 3 niveles de ruido (Recomendación 1)
    scenarios = [
        ("Limpio", 0.0),
        ("Ruido Moderado", 0.05),
        ("Ruido Fuerte", 0.4)
    ]
    
    rng = np.random.default_rng(2026)
    out_dir = "results/figures/correlation"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[i] Retardo Físico Real: {true_delay_samples} muestras ({true_delay_samples/fs*1000:.2f} ms)\n")
    print(f"{'Escenario':<15} | {'Varianza':<10} | {'Detectado (muestras)':<22} | {'Error (muestras)':<18}")
    print("-" * 72)
    
    for name, variance in scenarios:
        # 1. Simular recepción
        rx_signal = simulate_echoes(tx_signal, echoes, noise_variance=variance, rng=rng)
        
        # 2. Correlación cruzada
        corr_result = cross_correlate_fft(tx_signal, rx_signal)
        
        # 3. Detección (Recomendación 3 y 14)
        det_samples, det_seconds = estimate_delay(corr_result)
        error_samples = abs(true_delay_samples - det_samples)
        
        print(f"{name:<15} | {variance:<10.2f} | {det_samples:<22} | {error_samples:<18}")
        
        # 4. Gráfica de correlación por nivel de ruido (Recomendación 2)
        safe_name = name.lower().replace(" ", "_")
        plot_correlation(
            corr_result, 
            peak_lag=det_samples, 
            filepath=f"{out_dir}/deteccion_pico_{safe_name}.png"
        )

    print(f"\n[+] Evidencia gráfica guardada en {out_dir}/")
    print("--- Fin de la Etapa de Detección ---")

if __name__ == "__main__":
    run_echo_effects()