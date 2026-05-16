"""
simulation.py
=============
Motor principal de la simulación de eventos discretos del Poblado en Evolución.

Implementa la clase Simulation, que:
  1. Inicializa la población de M mujeres y H hombres con edades aleatorias.
  2. Puebla la cola de prioridad con los eventos iniciales.
  3. Ejecuta el bucle principal DES: extraer evento → despachar handler.
  4. Finaliza y devuelve las estadísticas recolectadas.

La cola de prioridad se implementa con heapq (min-heap), lo que garantiza
O(log n) por inserción y extracción.
"""

from __future__ import annotations

import heapq
from typing import List

from distributions import (
    uniform,
    time_to_death,
    wants_partner,
)
from events import Event, EventType, EVENT_HANDLERS
from person import Person
from stats import SimulationStats


class Simulation:
    """Motor de simulación de eventos discretos para el Poblado en Evolución.

    Gestiona la población, la cola de eventos y delega el procesamiento
    de cada evento al handler correspondiente de events.py.

    Attributes:
        num_men (int): Número inicial de hombres.
        num_women (int): Número inicial de mujeres.
        end_time (float): Horizonte de simulación en años (100 por defecto).
        population (set[Person]): Conjunto de personas vivas en todo momento.
        singles (set[Person]): Subconjunto de personas vivas y solteras
            disponibles para buscar pareja.
        stats (SimulationStats): Objeto acumulador de métricas.
        _event_queue (list): Heap de eventos pendientes.
        current_time (float): Tiempo actual de simulación.
    """

    def __init__(
        self,
        num_men: int = 100,
        num_women: int = 100,
        end_time: float = 100.0,
        seed: int | None = None,
    ) -> None:
        """Inicializa la simulación con los parámetros dados.

        Args:
            num_men: Número inicial de hombres en la población (default 100).
            num_women: Número inicial de mujeres en la población (default 100).
            end_time: Duración total de la simulación en años (default 100).
            seed: Semilla para el generador de números aleatorios. Si es None,
                  la simulación no es reproducible.
        """
        import random
        if seed is not None:
            random.seed(seed)

        self.num_men = num_men
        self.num_women = num_women
        self.end_time = end_time
        self.current_time: float = 0.0

        # Estructuras de datos de la población
        self.population: set[Person] = set()
        self.singles: set[Person] = set()   # solteros disponibles

        # Cola de prioridad (min-heap) de eventos
        self._event_queue: List[Event] = []

        # Recolector de estadísticas
        self.stats = SimulationStats(end_time)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def schedule(self, event: Event) -> None:
        """Inserta un evento en la cola de prioridad.

        Args:
            event: El evento a programar. Se ignorará si su tiempo es mayor
                   o igual al horizonte de simulación.
        """
        if event.time >= self.end_time:
            return  # No procesar eventos fuera del horizonte
        heapq.heappush(self._event_queue, event)

    def run(self) -> SimulationStats:
        """Ejecuta la simulación completa.

        Inicializa la población, puebla la cola con eventos iniciales y
        ejecuta el bucle DES hasta que la cola esté vacía o se alcance
        el horizonte temporal.

        Returns:
            SimulationStats con todas las métricas recolectadas durante
            la simulación.
        """
        self._initialize_population()
        self._seed_initial_events()

        # Bucle principal de eventos discretos
        while self._event_queue:
            event = heapq.heappop(self._event_queue)

            # Seguridad: no procesar más allá del horizonte
            if event.time > self.end_time:
                break

            self.current_time = event.time

            # Despachar al handler correspondiente
            handler = EVENT_HANDLERS.get(event.event_type)
            if handler is not None:
                handler(self, event)

        # Tomar snapshot final de la población
        self.stats.finalize(self.population)
        return self.stats

    # ------------------------------------------------------------------
    # Inicialización interna
    # ------------------------------------------------------------------

    def _initialize_population(self) -> None:
        """Crea la población inicial de hombres y mujeres.

        Cada persona recibe una edad inicial aleatoria U(0, 100) y es
        añadida a la población y al conjunto de solteros.
        """
        # Crear hombres
        for _ in range(self.num_men):
            age = uniform(0.0, 100.0)
            person = Person(is_male=True, age=age, birth_time=0.0)
            self.population.add(person)
            # Personas mayores de 12 pueden buscar pareja → añadir a solteros
            if age >= 12.0:
                self.singles.add(person)
            # Registrar en estadísticas
            self.stats.record_person_added(0.0, person)

        # Crear mujeres
        for _ in range(self.num_women):
            age = uniform(0.0, 100.0)
            person = Person(is_male=False, age=age, birth_time=0.0)
            self.population.add(person)
            if age >= 12.0:
                self.singles.add(person)
            self.stats.record_person_added(0.0, person)

    def _seed_initial_events(self) -> None:
        """Programa los eventos iniciales para toda la población de partida.

        Para cada persona se programa:
          - Su muerte (basada en la probabilidad según edad y sexo).
          - Su próximo cumpleaños (en 1 año).
          - Para mujeres en edad fértil con deseo de pareja: primer intento
            de embarazo (se activará sólo si tiene pareja cuando ocurra).

        También se programa el primer chequeo de formación de parejas.
        """
        for person in self.population:
            # Evento de muerte
            dt_death = time_to_death(person.age, person.is_male)
            self.schedule(Event(
                time=dt_death,
                event_type=EventType.DEATH,
                payload={"person": person}
            ))

            # Evento de cumpleaños (anual)
            self.schedule(Event(
                time=1.0,
                event_type=EventType.BIRTHDAY,
                payload={"person": person}
            ))
            # Los eventos de embarazo NO se agendan aquí: solo se agendan
            # en el momento en que la mujer forma pareja (handle_partner_check
            # y handle_birth), garantizando que siempre tiene pareja cuando
            # el evento ocurre.

        # Primer chequeo de formación de parejas (a los 6 meses)
        self.schedule(Event(
            time=0.5,
            event_type=EventType.PARTNER_CHECK,
            payload={}
        ))
