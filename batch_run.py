"""
batch_run.py
============
Herramienta de réplicas múltiples (batch) para el Poblado en Evolución.

Ejecuta N simulaciones independientes con semillas distintas y analiza la
variabilidad de los resultados. Permite responder preguntas como:
  - ¿Produce el modelo siempre la misma evolución poblacional?
  - ¿Cuánta dispersión existe en los resultados finales?
  - ¿Hay riesgo de extinción bajo los parámetros elegidos?

Métricas recolectadas por réplica
----------------------------------
  final_pop           Población viva al final de la simulación
  final_males         Hombres vivos al final
  final_females       Mujeres vivas al final
  final_couples       Parejas activas al final
  total_births        Total de nacimientos en toda la simulación
  total_deaths        Total de fallecimientos en toda la simulación
  total_couples       Parejas formadas durante toda la simulación
  total_breakups      Rupturas durante toda la simulación
  avg_solo_time       Tiempo medio en soltería (años) hasta emparejarse
  median_solo_time    Mediana del tiempo en soltería (años)

Salida
------
  - Tabla de resumen: media, desviación estándar, IC 95 %, mínimo y máximo
    de cada métrica a lo largo de todas las réplicas.
  - Gráficas con:
      1. Trayectorias poblacionales individuales + banda media ± 1σ.
      2-6. Histogramas de distribución de cada métrica escalar.

Uso
---
    python batch_run.py --runs 30
    python batch_run.py --runs 50 --men 100 --women 100 --years 100 --base-seed 0
    python batch_run.py --runs 20 --no-plot --output batch.png
"""

from __future__ import annotations

import argparse
import math
import time
from statistics import mean, median, stdev

from simulation import Simulation


# ---------------------------------------------------------------------------
# Tabla de valores críticos t (two-tailed, α = 0.05) para IC al 95 %
# Clave: grados de libertad (df = n - 1).
# Para df fuera de la tabla se interpola linealmente entre las entradas más
# cercanas; para df > 120 se usa la aproximación normal z = 1.96.
# ---------------------------------------------------------------------------
_T_TABLE: dict[int, float] = {
    1: 12.706, 2: 4.303,  3: 3.182,  4: 2.776,  5: 2.571,
    6:  2.447, 7: 2.365,  8: 2.306,  9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}
_T_KEYS = sorted(_T_TABLE)


def _t_critical(n: int) -> float:
    """Valor crítico t (IC 95 %, dos colas) para n observaciones (df = n-1).

    Args:
        n: Número de observaciones.

    Returns:
        Valor de t; devuelve inf si n < 2 y 1.96 para df muy grande.
    """
    df = n - 1
    if df <= 0:
        return float("inf")
    if df in _T_TABLE:
        return _T_TABLE[df]
    if df > _T_KEYS[-1]:
        return 1.96  # aproximación normal para df grande
    # Interpolación lineal entre las dos entradas más cercanas
    lo = max(k for k in _T_KEYS if k <= df)
    hi = min(k for k in _T_KEYS if k >= df)
    frac = (df - lo) / (hi - lo)
    return _T_TABLE[lo] + frac * (_T_TABLE[hi] - _T_TABLE[lo])


# ---------------------------------------------------------------------------
# Estadísticos descriptivos
# ---------------------------------------------------------------------------

def _desc(values: list[float]) -> dict:
    """Calcula estadísticos descriptivos de una lista numérica.

    Args:
        values: Lista de valores numéricos.

    Returns:
        Dict con claves: n, mean, std, ci_half, ci_lo, ci_hi, min, max, median.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0,
                "ci_half": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "min": 0.0, "max": 0.0, "median": 0.0}
    mu  = mean(values)
    sig = stdev(values) if n >= 2 else 0.0
    t   = _t_critical(n)
    ci_half = t * sig / math.sqrt(n)
    return {
        "n":       n,
        "mean":    mu,
        "std":     sig,
        "ci_half": ci_half,
        "ci_lo":   mu - ci_half,
        "ci_hi":   mu + ci_half,
        "min":     min(values),
        "max":     max(values),
        "median":  median(values),
    }


# ---------------------------------------------------------------------------
# Ejecución de réplicas
# ---------------------------------------------------------------------------

def run_batch(
    n_runs:    int,
    num_men:   int,
    num_women: int,
    end_time:  float,
    base_seed: int | None,
) -> list[dict]:
    """Ejecuta n_runs réplicas independientes y devuelve sus métricas.

    Cada réplica usa una semilla distinta (base_seed + i si se especifica,
    o semilla aleatoria en caso contrario).

    Args:
        n_runs:    Número de réplicas a ejecutar.
        num_men:   Número inicial de hombres.
        num_women: Número inicial de mujeres.
        end_time:  Duración de cada simulación en años.
        base_seed: Semilla base. La réplica i usará base_seed + i.
                   None → semillas aleatorias (no reproducible).

    Returns:
        Lista de dicts, uno por réplica, con todas las métricas escalares y
        la trayectoria poblacional anual (clave ``trajectory``).
    """
    results: list[dict] = []

    for i in range(n_runs):
        seed = (base_seed + i) if base_seed is not None else None

        # Feedback de progreso en la misma línea
        print(f"  Ejecutando réplica {i + 1}/{n_runs}...   ", end="\r", flush=True)

        sim  = Simulation(num_men=num_men, num_women=num_women,
                          end_time=end_time, seed=seed)
        s    = sim.run()

        results.append({
            "run":              i + 1,
            "seed":             seed,
            "final_pop":        s.final_males + s.final_females,
            "final_males":      s.final_males,
            "final_females":    s.final_females,
            "final_couples":    s.final_couples,
            "total_births":     s.total_births,
            "total_deaths":     s.total_deaths,
            "total_couples":    s.total_couples_formed,
            "total_breakups":   s.total_breakups,
            "avg_solo_time":    s.avg_solo_time,
            "median_solo_time": s.median_solo_time,
            "trajectory":       s.yearly_population(),
        })

    # Limpiar la línea de progreso
    print(" " * 50, end="\r")
    return results


# ---------------------------------------------------------------------------
# Informe textual
# ---------------------------------------------------------------------------

# Pares (clave_dict, etiqueta_legible) de las métricas a reportar
_METRICS: list[tuple[str, str]] = [
    ("final_pop",         "Población final"),
    ("final_males",       "Hombres finales"),
    ("final_females",     "Mujeres finales"),
    ("final_couples",     "Parejas activas al final"),
    ("total_births",      "Nacimientos totales"),
    ("total_deaths",      "Fallecimientos totales"),
    ("total_couples",     "Parejas formadas (total)"),
    ("total_breakups",    "Rupturas (total)"),
    ("avg_solo_time",     "Tiempo medio soltería (a)"),
    ("median_solo_time",  "Mediana soltería (a)"),
]


def print_report(
    results:   list[dict],
    n_runs:    int,
    num_men:   int,
    num_women: int,
    end_time:  float,
) -> None:
    """Imprime la tabla de resumen estadístico de todas las réplicas.

    Muestra media, desviación estándar, intervalo de confianza al 95 %,
    mínimo y máximo para cada métrica.

    Args:
        results:   Lista de dicts devuelta por run_batch().
        n_runs:    Número de réplicas ejecutadas.
        num_men:   Hombres iniciales (para cabecera).
        num_women: Mujeres iniciales (para cabecera).
        end_time:  Duración de simulación en años (para cabecera).
    """
    W = 86
    print("=" * W)
    print("  ESTUDIO DE RÉPLICAS MÚLTIPLES — POBLADO EN EVOLUCIÓN")
    print(f"  Réplicas: {n_runs}  |  H: {num_men}  |  M: {num_women}  |  Años: {end_time:.0f}")
    print("=" * W)
    print(
        f"  {'Métrica':<30}  {'Media':>9}  {'Desv.Std':>9}  "
        f"{'IC 95% inf':>10}  {'IC 95% sup':>10}  {'Mín':>8}  {'Máx':>8}"
    )
    print("  " + "─" * (W - 2))

    for key, label in _METRICS:
        vals = [float(r[key]) for r in results]
        d = _desc(vals)
        print(
            f"  {label:<30}  {d['mean']:>9.2f}  {d['std']:>9.2f}  "
            f"{d['ci_lo']:>10.2f}  {d['ci_hi']:>10.2f}  "
            f"{d['min']:>8.2f}  {d['max']:>8.2f}"
        )

    print("=" * W)
    print("  Nota: IC calculado con distribución t de Student (α = 0.05, bilateral).")


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def plot_batch(
    results:   list[dict],
    end_time:  float,
    show:      bool = True,
    save_path: str | None = None,
) -> None:
    """Genera las gráficas de análisis de réplicas múltiples.

    Produce 6 subplots en una figura 3×2:
      (0,0) Trayectorias individuales (gris) + media (azul) + banda ±1σ.
      (0,1) Histograma de la población final.
      (1,0) Histograma de nacimientos totales.
      (1,1) Histograma de fallecimientos totales.
      (2,0) Histograma de parejas formadas.
      (2,1) Histograma del tiempo medio en soltería.

    Args:
        results:   Lista de dicts devuelta por run_batch().
        end_time:  Horizonte de simulación (para el eje X de la trayectoria).
        show:      Si True, abre la ventana interactiva de matplotlib.
        save_path: Ruta de archivo PNG donde guardar la figura (opcional).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[batch_run] matplotlib no disponible. Instálalo con: pip install matplotlib")
        return

    n_runs = len(results)
    years  = list(range(int(math.floor(end_time)) + 1))
    trajs  = [r["trajectory"] for r in results]

    # Estadísticos por año para la banda
    traj_mean: list[float] = []
    traj_lo:   list[float] = []
    traj_hi:   list[float] = []
    for yi in range(len(years)):
        vals = [t[yi] for t in trajs if yi < len(t)]
        if not vals:
            traj_mean.append(0.0); traj_lo.append(0.0); traj_hi.append(0.0)
            continue
        mu  = mean(vals)
        sig = stdev(vals) if len(vals) >= 2 else 0.0
        traj_mean.append(mu)
        traj_lo.append(mu - sig)
        traj_hi.append(mu + sig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        f"Análisis de Réplicas ({n_runs} réplicas) — Poblado en Evolución",
        fontsize=13,
    )

    # --- 1. Trayectorias poblacionales ---
    ax = axes[0, 0]
    for traj in trajs:
        ax.plot(years[:len(traj)], traj,
                color="steelblue", alpha=0.15, linewidth=0.7)
    ax.plot(years, traj_mean, color="navy", linewidth=2.0, label="Media")
    ax.fill_between(years, traj_lo, traj_hi,
                    alpha=0.25, color="navy", label="±1 σ")
    ax.set_title("Trayectorias de Población")
    ax.set_xlabel("Año de simulación")
    ax.set_ylabel("Personas vivas")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Histogramas reutilizables ---
    def _hist(ax_: object, key: str, label: str, color: str) -> None:
        vals = [r[key] for r in results]
        ax_.hist(vals, bins="auto", color=color, edgecolor="white", alpha=0.85)
        mu = mean(vals)
        ax_.axvline(mu, color="black", linestyle="--", linewidth=1.4,
                    label=f"Media = {mu:.1f}")
        ax_.set_title(f"Distribución: {label}")
        ax_.set_xlabel(label)
        ax_.set_ylabel("Número de réplicas")
        ax_.legend(fontsize=9)
        ax_.grid(True, alpha=0.3, axis="x")

    _hist(axes[0, 1], "final_pop",      "Población final",              "steelblue")
    _hist(axes[1, 0], "total_births",   "Nacimientos totales",          "mediumseagreen")
    _hist(axes[1, 1], "total_deaths",   "Fallecimientos totales",       "tomato")
    _hist(axes[2, 0], "total_couples",  "Parejas formadas",             "orchid")
    _hist(axes[2, 1], "avg_solo_time",  "Tiempo medio en soltería (a)", "darkorange")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Gráfica guardada en: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos para el batch runner."""
    parser = argparse.ArgumentParser(
        description="Réplicas múltiples — Simulación del Poblado en Evolución",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=30,
        help="Número de réplicas independientes a ejecutar.",
    )
    parser.add_argument(
        "--men", "-m", type=int, default=100,
        help="Número inicial de hombres en cada réplica.",
    )
    parser.add_argument(
        "--women", "-w", type=int, default=100,
        help="Número inicial de mujeres en cada réplica.",
    )
    parser.add_argument(
        "--years", "-y", type=float, default=100.0,
        help="Duración de cada simulación en años.",
    )
    parser.add_argument(
        "--base-seed", "-s", type=int, default=None,
        help="Semilla base para reproducibilidad. La réplica i usa base-seed + i.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Ruta donde guardar la figura como PNG (ej: batch.png).",
    )
    parser.add_argument(
        "--no-plot", action="store_true", default=False,
        help="Suprimir la ventana interactiva de matplotlib.",
    )
    return parser.parse_args()


def main() -> None:
    """Función principal: ejecuta el batch y muestra resultados."""
    args = parse_args()

    print("=" * 60)
    print("  RÉPLICAS MÚLTIPLES — POBLADO EN EVOLUCIÓN")
    print("=" * 60)
    print(f"  Réplicas     : {args.runs}")
    print(f"  Hombres      : {args.men}")
    print(f"  Mujeres      : {args.women}")
    print(f"  Duración     : {args.years:.0f} años")
    if args.base_seed is not None:
        print(f"  Semilla base : {args.base_seed}  (réplica i → semilla {args.base_seed} + i)")
    else:
        print("  Semilla base : aleatoria  (usa --base-seed para reproducibilidad)")
    print()

    wall_start = time.perf_counter()
    results    = run_batch(
        n_runs=args.runs,
        num_men=args.men,
        num_women=args.women,
        end_time=args.years,
        base_seed=args.base_seed,
    )
    elapsed = time.perf_counter() - wall_start

    print(
        f"  {args.runs} réplicas completadas en {elapsed:.2f} s "
        f"({elapsed / args.runs:.2f} s/réplica).\n"
    )

    print_report(results, args.runs, args.men, args.women, args.years)
    print()

    show_plot = not args.no_plot
    if show_plot or args.output:
        plot_batch(results, args.years, show=show_plot, save_path=args.output)
    else:
        print(
            "  (Gráficas suprimidas. Usa --output para guardar o elimina "
            "--no-plot para ver la ventana interactiva.)"
        )


if __name__ == "__main__":
    main()
