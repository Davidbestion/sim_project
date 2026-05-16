"""
main.py
=======
Punto de entrada de la simulación del Poblado en Evolución.

Permite configurar los parámetros clave desde la línea de comandos:
  - Número inicial de hombres (--men, default 100)
  - Número inicial de mujeres (--women, default 100)
  - Duración de la simulación en años (--years, default 100)
  - Semilla aleatoria para reproducibilidad (--seed, default None)
  - Ruta para guardar la gráfica (--output, opcional)
  - Flag para suprimir la gráfica interactiva (--no-plot)

Ejemplo de uso:
    python main.py
    python main.py --men 200 --women 200 --seed 42
    python main.py --men 50 --women 60 --years 80 --output resultado.png --no-plot
"""

import argparse
import time

from simulation import Simulation


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los valores de todos los parámetros configurables.
    """
    parser = argparse.ArgumentParser(
        description="Simulación de Eventos Discretos — Poblado en Evolución",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--men", "-m",
        type=int,
        default=100,
        help="Número inicial de hombres en la población.",
    )
    parser.add_argument(
        "--women", "-w",
        type=int,
        default=100,
        help="Número inicial de mujeres en la población.",
    )
    parser.add_argument(
        "--years", "-y",
        type=float,
        default=100.0,
        help="Duración total de la simulación en años.",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Semilla para el generador de números aleatorios (reproducibilidad).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Ruta donde guardar la gráfica como imagen PNG (ej: resultado.png).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Suprimir la ventana interactiva de matplotlib.",
    )
    return parser.parse_args()


def main() -> None:
    """Función principal: configura y ejecuta la simulación, muestra resultados.

    Pasos:
      1. Parsear argumentos de la CLI.
      2. Crear e iniciar la simulación.
      3. Imprimir resumen textual de métricas.
      4. Generar gráficas (si no se suprime con --no-plot).
    """
    args = parse_args()

    print("=" * 50)
    print("  SIMULACIÓN DEL POBLADO EN EVOLUCIÓN")
    print("=" * 50)
    print(f"  Hombres iniciales : {args.men}")
    print(f"  Mujeres iniciales : {args.women}")
    print(f"  Duración          : {args.years:.0f} años")
    print(f"  Semilla aleatoria : {args.seed if args.seed is not None else 'aleatoria'}")
    print("  Iniciando simulación...")
    print()

    # Crear la simulación con los parámetros dados
    sim = Simulation(
        num_men=args.men,
        num_women=args.women,
        end_time=args.years,
        seed=args.seed,
    )

    # Ejecutar y medir tiempo de cómputo
    start_wall = time.perf_counter()
    stats = sim.run()
    elapsed = time.perf_counter() - start_wall

    print(f"  Simulación completada en {elapsed:.2f} segundos.")
    print()

    # Mostrar resumen textual
    print(stats.summary())
    print()

    # Generar gráficas
    show_plot = not args.no_plot
    if show_plot or args.output:
        stats.plot(show=show_plot, save_path=args.output)
    else:
        print("  (Gráficas suprimidas con --no-plot)")


if __name__ == "__main__":
    main()
