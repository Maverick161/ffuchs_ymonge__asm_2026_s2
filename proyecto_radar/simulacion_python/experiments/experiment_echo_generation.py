"""
Experimento: experiment_echo_generation.py
------------------------------------------
Objetivo: Generar la señal transmitida (Chirp) y simular el canal acústico 
con múltiples ecos conocidos, atenuación y distintos niveles de ruido.
"""

import os
import csv
import numpy as np

from radar.signals.generators import generate_chirp
from radar.simulation.echoes import EchoConfig, simulate_echoes
from radar.visualization.plots import plot_transmitted_received, plot_chirp_zoom

def run_echo_generation():
    print("--- Iniciando Experimento de Generación de Ecos ---")
    
    # 1. Parámetros del Radar
    fs = 48000.0
    duration = 0.05 
    
    tx_signal = generate_chirp(
        start_frequency=2000.0, 
        end_frequency=8000.0, 
        sample_rate=fs, 
        duration=duration
    )
    print(f"[*] Señal TX generada: {tx_signal.name} ({tx_signal.number_of_samples} muestras)")

    # 2. Configurar el escenario físico
    echoes = [
        EchoConfig(delay_samples=2000, attenuation=0.8),
        EchoConfig(delay_samples=7000, attenuation=0.4),
        EchoConfig(delay_samples=15000, attenuation=0.15)
    ]
    
    # Crear directorios de salida
    fig_dir = "results/figures/echoes"
    data_dir = "results/data"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 3. Exportar parámetros y mostrar en consola (Mejoras 13.1 y 13.4)
    csv_path = f"{data_dir}/echo_scenario.csv"
    print("\n[*] Resumen Teórico de Ecos:")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["echo_id", "delay_samples", "delay_seconds", "attenuation"])
        
        for i, echo in enumerate(echoes):
            delay_sec = echo.delay_samples / fs
            writer.writerow([i+1, echo.delay_samples, delay_sec, echo.attenuation])
            print(f"  Eco {i+1}:")
            print(f"    - Delay: {echo.delay_samples} muestras")
            print(f"    - Delay temporal: {delay_sec * 1000:.2f} ms")
            print(f"    - Atenuación: {echo.attenuation}")
            
    print(f"\n[+] Parámetros exportados a {csv_path}")

    # 4. Generar gráfica de zoom del Chirp (Mejora 13.3)
    plot_chirp_zoom(tx_signal, zoom_ms=5.0, filepath=f"{fig_dir}/tx_chirp_zoom_5ms.png")
    print(f"[+] Gráfica de zoom del chirp guardada en {fig_dir}/tx_chirp_zoom_5ms.png")

    # 5. Generar escenarios con diferentes niveles de ruido (Mejora 13.5)
    noise_scenarios = [
        ("limpia", 0.0),
        ("ruido_moderado", 0.02),
        ("ruido_fuerte", 0.2)
    ]

    rng = np.random.default_rng(2026)

    for name, variance in noise_scenarios:
        print(f"\n[*] Simulando escenario: {name} (Varianza: {variance})")
        rx_signal = simulate_echoes(tx_signal, echoes, noise_variance=variance, rng=rng)
        
        filepath = f"{fig_dir}/recibida_{name}.png"
        plot_transmitted_received(tx_signal, rx_signal, echoes=echoes, filepath=filepath)
        print(f"[+] Gráfica guardada en {filepath}")

    print("\n--- Experimento Finalizado ---")

if __name__ == "__main__":
    run_echo_generation()