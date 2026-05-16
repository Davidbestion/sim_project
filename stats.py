"""
stats.py
========
Módulo de recolección de estadísticas y visualización de resultados para la
simulación del Poblado en Evolución.

La clase SimulationStats acumula métricas clave a lo largo de la simulación:
  - Evolución temporal de la población (total, hombres, mujeres).
  - Nacimientos y muertes por año.
  - Formación y ruptura de parejas.
  - Distribución de edades al final de la simulación.

Al finalizar, el método plot() genera gráficas con matplotlib.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from person import Person


class SimulationStats:
    """Acumulador de métricas de la simulación del Poblado en Evolución.

    Registra eventos demográficos a medida que ocurren y los organiza
    en series temporales anuales para su posterior análisis y visualización.

    Attributes:
        end_time (float): Horizonte de simulación en años.
        births_per_year (dict[int, int]): Nacimientos totales por año.
        deaths_per_year (dict[int, int]): Muertes totales por año.
        couples_formed_per_year (dict[int, int]): Parejas formadas por año.
        breakups_per_year (dict[int, int]): Rupturas por año.
        population_snapshots (list[tuple[float, int]]): Snapshots (tiempo, tamaño)
            tomados en cada evento relevante.
        final_ages (list[float]): Edades de los sobrevivientes al final.
        total_births (int): Total acumulado de nacimientos.
        total_deaths (int): Total acumulado de muertes.
        total_couples_formed (int): Total de parejas formadas.
        total_breakups (int): Total de rupturas.
    """

    def __init__(self, end_time: float) -> None:
        """Inicializa el acumulador de estadísticas.

        Args:
            end_time: Horizonte de simulación en años (para etiquetar gráficas).
        """
        self.end_time = end_time

        # Métricas anuales (clave: año entero)
        self.births_per_year: dict[int, int] = defaultdict(int)
        self.deaths_per_year: dict[int, int] = defaultdict(int)
        self.couples_formed_per_year: dict[int, int] = defaultdict(int)
        self.breakups_per_year: dict[int, int] = defaultdict(int)

        # Evolución de la población: lista de (tiempo, delta) para reconstruir
        # la curva de tamaño poblacional con eficiencia O(1) por evento.
        self._pop_events: list[tuple[float, int]] = []  # (tiempo, +1 o -1)

        # Datos finales
        self.final_ages: list[float] = []
        self.final_males: int = 0
        self.final_females: int = 0
        self.final_couples: int = 0

        # Totales acumulados
        self.total_births: int = 0
        self.total_deaths: int = 0
        self.total_couples_formed: int = 0
        self.total_breakups: int = 0

        # Tamaño inicial de la población (se actualiza con record_person_added)
        self._current_pop: int = 0

    # ------------------------------------------------------------------
    # Métodos de registro (llamados por los handlers de eventos)
    # ------------------------------------------------------------------

    def record_person_added(self, time: float, person: Person) -> None:
        """Registra la incorporación de una persona a la población.

        Debe llamarse al nacer un bebé o al inicializar la población.

        Args:
            time: Tiempo de simulación en que ocurre.
            person: Persona incorporada (no se usa actualmente, preparado
                    para extensiones futuras).
        """
        self._current_pop += 1
        self._pop_events.append((time, 1))

    def record_death(self, time: float, person: Person) -> None:
        """Registra el fallecimiento de una persona.

        Args:
            time: Tiempo de simulación del fallecimiento.
            person: Persona fallecida.
        """
        year = int(math.floor(time))
        self.deaths_per_year[year] += 1
        self.total_deaths += 1
        self._current_pop -= 1
        self._pop_events.append((time, -1))

    def record_birth(self, time: float, num_babies: int) -> None:
        """Registra el nacimiento de uno o más bebés.

        Nota: record_person_added se llama por separado por cada bebé en
        handle_birth; este método sólo acumula el conteo de nacimientos.

        Args:
            time: Tiempo de simulación del parto.
            num_babies: Número de bebés nacidos.
        """
        year = int(math.floor(time))
        self.births_per_year[year] += num_babies
        self.total_births += num_babies

    def record_couple_formed(self, time: float) -> None:
        """Registra la formación de una nueva pareja.

        Args:
            time: Tiempo de simulación de la formación de pareja.
        """
        year = int(math.floor(time))
        self.couples_formed_per_year[year] += 1
        self.total_couples_formed += 1

    def record_breakup(self, time: float) -> None:
        """Registra la ruptura de una pareja.

        Args:
            time: Tiempo de simulación de la ruptura.
        """
        year = int(math.floor(time))
        self.breakups_per_year[year] += 1
        self.total_breakups += 1

    def finalize(self, population: set) -> None:
        """Calcula métricas finales a partir de la población sobreviviente.

        Debe llamarse al terminar la simulación.

        Args:
            population: Conjunto de personas vivas al final de la simulación.
        """
        from person import RelationshipStatus
        counted_couples = set()

        for person in population:
            self.final_ages.append(person.age)
            if person.is_male:
                self.final_males += 1
            else:
                self.final_females += 1
            # Contar parejas únicas (cada pareja aparece dos veces)
            if person.is_in_couple and person.partner is not None:
                pair_id = frozenset({person.pid, person.partner.pid})
                counted_couples.add(pair_id)

        self.final_couples = len(counted_couples)

    # ------------------------------------------------------------------
    # Construcción de series temporales
    # ------------------------------------------------------------------

    def _build_population_series(self) -> tuple[list[float], list[int]]:
        """Construye la serie temporal del tamaño poblacional.

        Ordena los eventos de población por tiempo y reconstruye el tamaño
        acumulado en cada instante.

        Returns:
            Tupla (tiempos, tamaños) con listas paralelas.
        """
        # Ordenar por tiempo
        events = sorted(self._pop_events, key=lambda x: x[0])
        times = []
        sizes = []
        size = 0
        for t, delta in events:
            size += delta
            times.append(t)
            sizes.append(size)
        return times, sizes

    def _annual_series(self, data: dict[int, int]) -> tuple[list[int], list[int]]:
        """Convierte un dict {año: conteo} en listas ordenadas para graficar.

        Rellena los años sin datos con 0 para obtener una serie continua.

        Args:
            data: Diccionario con conteos anuales.

        Returns:
            Tupla (años, conteos) como listas ordenadas.
        """
        max_year = int(math.ceil(self.end_time))
        years = list(range(max_year))
        counts = [data.get(y, 0) for y in years]
        return years, counts

    # ------------------------------------------------------------------
    # Resumen en texto
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Genera un resumen en texto de las métricas principales.

        Returns:
            Cadena con el resumen formateado.
        """
        final_pop = self.final_males + self.final_females
        avg_age = (
            sum(self.final_ages) / len(self.final_ages)
            if self.final_ages else 0.0
        )
        lines = [
            "=" * 50,
            "   RESUMEN DE LA SIMULACIÓN - POBLADO EN EVOLUCIÓN",
            "=" * 50,
            f"  Horizonte simulado     : {self.end_time:.0f} años",
            f"  Población final total  : {final_pop}",
            f"    Hombres              : {self.final_males}",
            f"    Mujeres              : {self.final_females}",
            f"    Parejas activas      : {self.final_couples}",
            f"  Edad promedio final    : {avg_age:.1f} años",
            f"  Total nacimientos      : {self.total_births}",
            f"  Total fallecimientos   : {self.total_deaths}",
            f"  Parejas formadas       : {self.total_couples_formed}",
            f"  Rupturas ocurridas     : {self.total_breakups}",
            "=" * 50,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------

    def plot(self, show: bool = True, save_path: str | None = None) -> None:
        """Genera y muestra (o guarda) las gráficas de resultados.

        Produce 4 subplots:
          1. Evolución del tamaño poblacional a lo largo del tiempo.
          2. Nacimientos y muertes por año.
          3. Formación y ruptura de parejas por año.
          4. Histograma de edades de la población final.

        Args:
            show: Si True, muestra la ventana interactiva de matplotlib.
            save_path: Si se provee una ruta, guarda la figura como imagen PNG.
                       Ejemplo: "resultados.png".
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[stats] matplotlib no está instalado. Instálalo con: pip install matplotlib")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Simulación del Poblado en Evolución — Resultados", fontsize=14)

        # --- 1. Evolución poblacional ---
        ax1 = axes[0, 0]
        times, sizes = self._build_population_series()
        ax1.plot(times, sizes, color="steelblue", linewidth=1.2)
        ax1.set_title("Evolución de la Población")
        ax1.set_xlabel("Tiempo (años)")
        ax1.set_ylabel("Número de personas vivas")
        ax1.grid(True, alpha=0.3)

        # --- 2. Nacimientos y muertes por año ---
        ax2 = axes[0, 1]
        years_b, counts_b = self._annual_series(self.births_per_year)
        years_d, counts_d = self._annual_series(self.deaths_per_year)
        ax2.bar(years_b, counts_b, alpha=0.7, color="mediumseagreen", label="Nacimientos")
        ax2.bar(years_d, counts_d, alpha=0.7, color="tomato", label="Muertes")
        ax2.set_title("Nacimientos y Muertes por Año")
        ax2.set_xlabel("Año de simulación")
        ax2.set_ylabel("Cantidad")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # --- 3. Formación y ruptura de parejas por año ---
        ax3 = axes[1, 0]
        years_c, counts_c = self._annual_series(self.couples_formed_per_year)
        years_r, counts_r = self._annual_series(self.breakups_per_year)
        ax3.plot(years_c, counts_c, color="orchid", label="Parejas formadas", marker="o", markersize=3)
        ax3.plot(years_r, counts_r, color="darkorange", label="Rupturas", marker="s", markersize=3)
        ax3.set_title("Dinámica de Parejas por Año")
        ax3.set_xlabel("Año de simulación")
        ax3.set_ylabel("Cantidad")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- 4. Histograma de edades finales ---
        ax4 = axes[1, 1]
        if self.final_ages:
            ax4.hist(self.final_ages, bins=20, color="cornflowerblue", edgecolor="white")
        ax4.set_title("Distribución de Edades — Población Final")
        ax4.set_xlabel("Edad (años)")
        ax4.set_ylabel("Número de personas")
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[stats] Gráfica guardada en: {save_path}")

        if show:
            plt.show()
