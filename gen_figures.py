"""
gen_figures.py
==============
Genera todas las figuras y estadísticos de batch para el informe.

Para cada escenario:
  1. Corre una réplica individual (semilla 42) y guarda el plot de stats.
  2. Corre 30 réplicas (batch) y guarda el plot de batch.
  3. Imprime los estadísticos en formato listo para copiar al LaTeX.

Uso:
    python gen_figures.py
"""

from __future__ import annotations

import os
import sys

# Asegura que el directorio del proyecto esté en el path
sys.path.insert(0, os.path.dirname(__file__))

from simulation import Simulation
from batch_run import run_batch, plot_batch, _desc

OUT_DIR = os.path.join(os.path.dirname(__file__), "informe", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_RUNS   = 30
END_TIME = 100.0
SEED     = 42

# Escenarios: (H, M, etiqueta)
SCENARIOS = [
    ( 50, 100, "h050_m100"),
    (100, 100, "h100_m100"),
    (200, 100, "h200_m100"),
    (400, 100, "h400_m100"),
    (100,  50, "h100_m050"),
    (100, 200, "h100_m200"),
    (100, 400, "h100_m400"),
]

METRICS = [
    ("final_pop",       "Población final"),
    ("total_births",    "Nacimientos totales"),
    ("total_deaths",    "Fallecimientos totales"),
    ("total_couples",   "Parejas formadas"),
    ("total_breakups",  "Rupturas"),
    ("avg_solo_time",   "Tiempo medio soltería (a)"),
]


def run_single(h: int, m: int, label: str) -> None:
    """Corre una réplica con semilla 42 y guarda el plot de stats."""
    print(f"  Corrida individual  H={h:3d} M={m:3d} ...", end=" ", flush=True)
    sim = Simulation(num_men=h, num_women=m, end_time=END_TIME, seed=SEED)
    s   = sim.run()
    path = os.path.join(OUT_DIR, f"single_{label}.png")
    s.plot(show=False, save_path=path)
    print(f"→ {os.path.basename(path)}")


def run_batch_scenario(h: int, m: int, label: str) -> list[dict]:
    """Corre 30 réplicas, guarda el plot de batch y devuelve los resultados."""
    print(f"  Batch ({N_RUNS} réplicas) H={h:3d} M={m:3d} ...", flush=True)
    results = run_batch(N_RUNS, num_men=h, num_women=m,
                        end_time=END_TIME, base_seed=0)
    path = os.path.join(OUT_DIR, f"batch_{label}.png")
    plot_batch(results, end_time=END_TIME, show=False, save_path=path)
    print(f"  → {os.path.basename(path)}")
    return results


def print_latex_row(h: int, m: int, results: list[dict]) -> None:
    """Imprime una fila LaTeX con los estadísticos del batch."""
    pop   = _desc([float(r["final_pop"])    for r in results])
    birth = _desc([float(r["total_births"]) for r in results])
    death = _desc([float(r["total_deaths"]) for r in results])
    coup  = _desc([float(r["total_couples"])  for r in results])
    solo  = _desc([float(r["avg_solo_time"]) for r in results])

    print(
        f"    {h:3d} & {m:3d} "
        f"& {pop['mean']:6.1f} & [{pop['ci_lo']:5.1f},{pop['ci_hi']:5.1f}] "
        f"& {birth['mean']:6.1f} & [{birth['ci_lo']:5.1f},{birth['ci_hi']:5.1f}] "
        f"& {death['mean']:6.1f} "
        f"& {coup['mean']:6.1f} "
        f"& {solo['mean']:4.2f} \\\\"
    )


def main() -> None:
    print("\n=== Generando figuras y estadísticos ===\n")

    all_batch: dict[str, list[dict]] = {}

    for h, m, label in SCENARIOS:
        print(f"--- Escenario H={h} M={m} ---")
        run_single(h, m, label)
        results = run_batch_scenario(h, m, label)
        all_batch[label] = results
        print()

    # ---- Tabla LaTeX resumen por escenario --------------------------------
    print("\n" + "=" * 70)
    print("TABLA LaTeX — variación de H (M=100 fijo):")
    print("=" * 70)
    print(r"% H & M & Pob. media & IC 95% & Nacim. media & IC 95% & Fallec. & Parejas & t_solo")
    for h, m, label in SCENARIOS:
        if m == 100:
            print_latex_row(h, m, all_batch[label])

    print("\n" + "=" * 70)
    print("TABLA LaTeX — variación de M (H=100 fijo):")
    print("=" * 70)
    for h, m, label in SCENARIOS:
        if h == 100:
            print_latex_row(h, m, all_batch[label])

    # ---- Tabla detallada del escenario base (100,100) ----------------------
    base = all_batch["h100_m100"]
    print("\n" + "=" * 70)
    print("TABLA LaTeX — estadísticos detallados escenario base (H=100, M=100):")
    print("=" * 70)
    for key, label in METRICS:
        d = _desc([float(r[key]) for r in base])
        print(
            f"    {label:<35} & {d['mean']:7.2f} & {d['std']:7.2f} "
            f"& {d['ci_lo']:7.2f} & {d['ci_hi']:7.2f} "
            f"& {d['min']:7.0f} & {d['max']:7.0f} \\\\"
        )

    print(f"\nFiguras guardadas en: {OUT_DIR}")
    print("=== Listo ===\n")


if __name__ == "__main__":
    main()
