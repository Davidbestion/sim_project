"""
person.py
=========
Define la clase Person, que representa a un individuo dentro de la simulación
del Poblado en Evolución.

Cada persona tiene un estado interno que evoluciona a lo largo del tiempo:
edad, sexo, estado civil (soltero/en pareja/período de soledad), número de
hijos tenidos y número de hijos deseados.
"""

from __future__ import annotations

import itertools
from enum import Enum, auto
from typing import Optional

from distributions import desired_children, wants_partner_probability, bernoulli, PARTNER_WISH_THRESHOLDS


# Los umbrales de edad donde se re-sortea el deseo de pareja se importan
# directamente de la tabla en distributions.py (PARTNER_WISH_THRESHOLDS).


# Generador de IDs únicos para cada persona
_id_counter = itertools.count(1)


class RelationshipStatus(Enum):
    """Estados posibles de la vida sentimental de una persona."""
    SINGLE = auto()        # Soltera, disponible para buscar pareja
    IN_COUPLE = auto()     # Tiene pareja activa
    GRIEVING = auto()      # Período de soledad tras ruptura o viudez


class Person:
    """Representa a un individuo dentro de la simulación poblacional.

    Almacena toda la información relevante de una persona: su identidad,
    estado de salud (viva/muerta), edad, sexo, relación sentimental actual
    y datos reproductivos. Los eventos de la simulación modifican estos
    atributos a lo largo del tiempo.

    Attributes:
        pid (int): Identificador único de la persona.
        is_male (bool): True si es hombre, False si es mujer.
        age (float): Edad actual en años.
        alive (bool): True si la persona está viva.
        status (RelationshipStatus): Estado sentimental actual.
        partner (Optional[Person]): Referencia a la pareja actual, o None.
        children_had (int): Número de hijos que ha tenido.
        children_desired (int): Número de hijos que desea tener.
        pregnant (bool): True si la mujer está actualmente embarazada.
        available_for_partner (bool): True si puede buscar pareja (no en
            período de soledad y no tiene pareja).
        birth_time (float): Tiempo de simulación en que nació/fue creada.
    """

    def __init__(self, is_male: bool, age: float, birth_time: float = 0.0) -> None:
        """Inicializa una nueva persona con los atributos dados.

        Args:
            is_male: Sexo de la persona (True = hombre, False = mujer).
            age: Edad inicial en años.
            birth_time: Tiempo de simulación en que esta persona "nace"
                        (0.0 para la población inicial).
        """
        self.pid: int = next(_id_counter)
        self.is_male: bool = is_male
        self.age: float = age
        self.alive: bool = True
        self.birth_time: float = birth_time

        # Estado sentimental
        self.status: RelationshipStatus = RelationshipStatus.SINGLE
        self.partner: Optional[Person] = None
        # Tiempo absoluto de simulación en que la pareja actual se romperá.
        # Se fija al formarse la pareja; inf = sin ruptura prevista.
        self.couple_breakup_time: float = float("inf")
        # Deseo intrínseco de tener pareja. False por defecto; se sortea
        # mediante update_partner_desire() al cruzar cada umbral de tramo.
        # Para la población inicial adulta se llama desde _initialize_population().
        self.partner_desire: bool = False

        # Tiempo de simulación en que la persona entró a su período de soltería
        # más reciente. Se usa para medir cuánto tiempo estuvo sola antes de
        # formar pareja. Se actualiza en SOLO_END y cuando entra al pool a los 12.
        self.solo_since: float = birth_time

        # True mientras haya un evento SEEK_PARTNER pendiente en la cola.
        # Evita programar eventos duplicados que distorsionarían la tasa de encuentros.
        self.seek_in_progress: bool = False

        # Datos reproductivos

        self.children_had: int = 0
        self.children_desired: int = desired_children()
        self.pregnant: bool = False  # Sólo relevante para mujeres

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def is_female(self) -> bool:
        """True si la persona es mujer."""
        return not self.is_male

    @property
    def is_single(self) -> bool:
        """True si la persona está soltera y disponible (no en período de duelo)."""
        return self.status == RelationshipStatus.SINGLE

    @property
    def is_in_couple(self) -> bool:
        """True si la persona tiene pareja activa."""
        return self.status == RelationshipStatus.IN_COUPLE

    @property
    def is_grieving(self) -> bool:
        """True si la persona está en período de soledad obligatoria."""
        return self.status == RelationshipStatus.GRIEVING

    @property
    def wants_more_children(self) -> bool:
        """True si la persona aún no ha alcanzado el número de hijos deseados."""
        return self.children_had < self.children_desired

    @property
    def can_be_pregnant(self) -> bool:
        """True si la mujer cumple las condiciones básicas para quedar embarazada.

        Condiciones: estar viva, ser mujer, tener pareja, no estar ya
        embarazada, y no haber superado el mínimo de hijos deseado entre
        ella y su pareja (en cuanto uno de los dos queda satisfecho, paran).
        """
        if not self.alive or self.is_male or not self.is_in_couple:
            return False
        if self.pregnant or self.partner is None:
            return False
        # El umbral de hijos es el MENOR entre lo que desea ella y lo que
        # desea su pareja: en cuanto uno de los dos queda satisfecho, la
        # pareja deja de intentar tener más hijos.
        min_desired = min(self.children_desired, self.partner.children_desired)
        return self.children_had < min_desired

    # ------------------------------------------------------------------
    # Métodos de cambio de estado
    # ------------------------------------------------------------------

    def form_couple(self, other: Person) -> None:
        """Establece una relación de pareja entre esta persona y otra.

        Actualiza el estado sentimental de ambas personas a IN_COUPLE y
        crea la referencia mutua de pareja.

        Args:
            other: La otra persona que pasará a ser pareja.

        Raises:
            ValueError: Si alguna de las dos personas ya tiene pareja.
        """
        if self.is_in_couple:
            raise ValueError(
                f"No se puede formar pareja: persona {self.pid} ya tiene pareja."
            )
        if other.is_in_couple:
            raise ValueError(
                f"No se puede formar pareja: persona {other.pid} ya tiene pareja."
            )
        # if self.is_in_couple or other.is_in_couple:
        #     raise ValueError(
        #         f"No se puede formar pareja: persona {self.pid} o {other.pid} ya tiene pareja."
        #     )
        self.partner = other
        other.partner = self
        self.status = RelationshipStatus.IN_COUPLE
        other.status = RelationshipStatus.IN_COUPLE

    def break_up(self) -> None:
        """Disuelve la relación de pareja de esta persona.

        Establece el estado de ambos miembros de la pareja a GRIEVING
        (período de soledad obligatoria). Limpia las referencias mutuas y
        resetea couple_breakup_time en ambos, invalidando cualquier evento
        de embarazo que pudiera estar pendiente para tiempos posteriores.
        """
        if self.partner is not None:
            self.partner.status = RelationshipStatus.GRIEVING
            self.partner.couple_breakup_time = float("inf")
            self.partner.partner = None
        self.status = RelationshipStatus.GRIEVING
        self.couple_breakup_time = float("inf")
        self.partner = None

    def end_grieving(self) -> None:
        """Finaliza el período de soledad de la persona.

        La persona pasa a estar SINGLE y puede volver a buscar pareja.
        """
        if self.status == RelationshipStatus.GRIEVING:
            self.status = RelationshipStatus.SINGLE

    def die(self) -> None:
        """Marca a la persona como fallecida y disuelve su pareja si tenía.

        Si tenía pareja, la pareja queda en estado GRIEVING (viudez) y su
        couple_breakup_time se resetea, cancelando futuros intentos de embarazo
        que pudieran estar agendados.
        """
        self.alive = False
        if self.is_in_couple and self.partner is not None:
            # La pareja queda viuda → entra en período de soledad
            self.partner.status = RelationshipStatus.GRIEVING
            self.partner.couple_breakup_time = float("inf")
            self.partner.partner = None
            self.partner = None

    def update_partner_desire(self) -> None:
        """Re-sortea el deseo de pareja según el tramo de edad actual.

        Debe llamarse al cruzar un umbral de tramo (desde birthday()) o
        una vez al inicializar personas adultas de la población inicial.
        """
        self.partner_desire = bernoulli(wants_partner_probability(self.age))

    def birthday(self) -> None:
        """Incrementa la edad de la persona en un año.

        Si al cumplir años la persona cruza el umbral de un nuevo tramo de
        edad (según _PARTNER_DESIRE_THRESHOLDS), se re-sortea su deseo de
        tener pareja con la probabilidad del nuevo tramo.
        """
        old_age = self.age
        self.age += 1.0
        # Detectar si se cruzó algún umbral de tramo
        for threshold in PARTNER_WISH_THRESHOLDS:
            if old_age < threshold <= self.age:
                self.update_partner_desire()
                break  # solo puede cruzarse un umbral por año

    def register_birth(self, num_babies: int) -> None:
        """Registra que la mujer ha dado a luz y actualiza el conteo de hijos.

        También actualiza el conteo del padre (pareja) si sigue vivo y en pareja.

        Args:
            num_babies: Número de bebés nacidos en el parto.
        """
        self.pregnant = False
        self.children_had += num_babies
        # Actualizar también el contador del padre
        if self.partner is not None and self.partner.alive:
            self.partner.children_had += num_babies

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sex = "H" if self.is_male else "M"
        status_str = self.status.name
        return (
            f"Person(pid={self.pid}, sex={sex}, age={self.age:.1f}, "
            f"alive={self.alive}, status={status_str})"
        )
