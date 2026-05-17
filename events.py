"""
events.py
=========
Define los tipos de eventos y sus handlers para la simulación.

Cada evento tiene:
  - Un tiempo en que ocurre (usado como clave de prioridad en la cola).
  - Un tipo (EventType).
  - Datos asociados (payload) con la información necesaria para procesarlo.

El motor de simulación (simulation.py) extrae eventos de la cola de prioridad
y los despacha al handler correspondiente definido aquí.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Evitar importación circular: Simulation se importa sólo para type hints
    from simulation import Simulation


class EventType(Enum):
    """Tipos de eventos posibles en la simulación."""
    DEATH          = auto()   # Fallecimiento de una persona
    BIRTHDAY       = auto()   # Cumpleaños anual (actualiza edad)
    PREGNANCY      = auto()   # Intento de embarazo de una mujer en pareja
    BIRTH          = auto()   # Nacimiento de bebé(s) tras embarazo
    BREAKUP        = auto()   # Ruptura de una pareja
    SOLO_END       = auto()   # Fin del período de soledad post-ruptura/viudez
    SEEK_PARTNER   = auto()   # Una persona soltera busca pareja (Exp individual)


@dataclass(order=True)
class Event:
    """Representa un evento en la cola de prioridad de la simulación.

    Se ordena por tiempo: el evento con menor tiempo tiene mayor prioridad.

    Attributes:
        time (float): Tiempo de simulación en que ocurre el evento (en años).
        event_type (EventType): Tipo del evento.
        payload (Any): Datos adicionales necesarios para procesar el evento.
            Generalmente un dict con claves que dependen del tipo de evento.
    """
    time: float
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False, default=None)


# ---------------------------------------------------------------------------
# Handlers de eventos
# ---------------------------------------------------------------------------

def handle_death(sim: Simulation, event: Event) -> None:
    """Procesa el fallecimiento de una persona.

    Marca a la persona como muerta, la elimina del registro de personas vivas
    y actualiza las estadísticas. Si tenía pareja, la pareja queda viuda y se
    programa el fin de su período de soledad.

    Args:
        sim: Instancia de la simulación (acceso a población, cola y stats).
        event: Evento con payload {'person': Person}.
    """
    from distributions import sample_alone_period

    person = event.payload["person"]

    # Ignorar si ya murió (evento duplicado por cambio de tramo de edad)
    if not person.alive:
        return

    partner = person.partner  # guardar antes de llamar a die()
    person.die()              # actualiza estado y desvincula pareja

    # Eliminar de la lista de vivos y de solteros disponibles
    sim.population.discard(person)
    sim.singles.discard(person)

    # Registrar estadísticas
    sim.stats.record_death(event.time, person)

    # Si la pareja sobrevive, programar fin de su período de soledad (viudez)
    if partner is not None and partner.alive:
        # si era el partner de alguien, no era soltero disponible, no estaba en singles, no hace falta descartarlo de nuevo
        # sim.singles.discard(partner)  # ya no es soltero disponible todavía
        duration = sample_alone_period(partner.age)
        sim.schedule(Event(
            time=event.time + duration,
            event_type=EventType.SOLO_END,
            payload={"person": partner}
        ))


def handle_birthday(sim: Simulation, event: Event) -> None:
    """Procesa el cumpleaños anual de una persona.

    Incrementa su edad en un año y reprograma el próximo cumpleaños.
    Además gestiona dos efectos colaterales:
      - Si la persona cruza los 12 años, entra al pool de solteros.
      - Si tras actualizar el deseo de pareja (tramo) el resultado es True
        y no hay un SEEK_PARTNER pendiente, inicia la cadena de búsqueda.

    Args:
        sim: Instancia de la simulación.
        event: Evento con payload {'person': Person}.
    """
    person = event.payload["person"]

    if not person.alive:
        return

    old_age = person.age
    person.birthday()  # +1 año; actualiza partner_desire si cruza umbral de tramo

    # Si la persona acaba de cruzar los 12 años: incorporarla al pool de solteros
    if old_age < 12.0 <= person.age:
        person.solo_since = event.time
        sim.singles.add(person)

    # Si está soltera, desea pareja y no hay búsqueda activa: iniciar cadena.
    # Cubre el primer cruce del umbral 12, cambios de deseo False→True en
    # umbrales posteriores, y cualquier otro caso en que la cadena se detuvo.
    if person.is_single and person.partner_desire and not person.seek_in_progress:
        _schedule_seek_partner(sim, event.time, person)

    # Re-programar próximo cumpleaños
    next_birthday = event.time + 1.0
    if next_birthday < sim.end_time:
        sim.schedule(Event(
            time=next_birthday,
            event_type=EventType.BIRTHDAY,
            payload={"person": person}
        ))


def handle_pregnancy(sim: Simulation, event: Event) -> None:
    """Procesa un intento de embarazo de una mujer que está en pareja.

    Este evento SOLO se agenda en dos momentos garantizados:
      1. Al formarse la pareja (handle_partner_check), si el tiempo del
         primer intento cae antes del tiempo de ruptura previsto.
      2. Tras un parto (handle_birth), si el siguiente intento también cae
         antes del tiempo de ruptura.

    Por construcción, cuando este evento llega la mujer tiene pareja,
    por lo que NO es necesario verificar is_in_couple en el handler.
    Las únicas razones para no embarazarse son:
      - La mujer murió antes de que llegara el evento.
      - Ya está embarazada (embarazo en curso).
      - Ambos (ella y su pareja) ya tienen los hijos deseados.
    En esos casos se descarta el intento sin reprogramar (la reprogramación
    tras el parto es responsabilidad de handle_birth).

    Args:
        sim: Instancia de la simulación.
        event: Evento con payload {'woman': Person}.
    """
    from distributions import pregnancy_probability, time_to_pregnancy, \
        pregnancy_duration, bernoulli

    woman = event.payload["woman"]

    # Cancelar si la mujer murió antes de que llegara este evento
    if not woman.alive:
        return

    # Cancelar si ya está embarazada
    if woman.pregnant:
        return
    # Cancelar si ya no tiene pareja (puede haber muerto antes de este evento)
    if not woman.is_in_couple or woman.partner is None:
        return
    # Cancelar si ya han llegado al mínimo de los dos deseos: en cuanto uno
    # de los dos queda satisfecho, la pareja deja de intentar tener más hijos.
    min_desired = min(woman.children_desired, woman.partner.children_desired)
    if woman.children_had >= min_desired:
        return

    # Sortear si queda embarazada en este intento
    p = pregnancy_probability(woman.age)
    if bernoulli(p):
        woman.pregnant = True
        sim.schedule(Event(
            time=event.time + pregnancy_duration(),
            event_type=EventType.BIRTH,
            payload={"woman": woman}
        ))
    else:
        # No quedó embarazada; re-agendar solo si el próximo intento ocurre
        # antes de la ruptura prevista (sigue teniendo pareja en ese momento)
        dt = time_to_pregnancy(woman.age)
        t_next = event.time + dt
        if t_next < woman.couple_breakup_time:
            sim.schedule(Event(
                time=t_next,
                event_type=EventType.PREGNANCY,
                payload={"woman": woman}
            ))


def handle_birth(sim: Simulation, event: Event) -> None:
    """Procesa el nacimiento de uno o más bebés.

    Crea los objetos Person para cada bebé, los añade a la población y
    programa sus eventos futuros (DEATH, BIRTHDAY). Actualiza el conteo de
    hijos de los padres.

    Args:
        sim: Instancia de la simulación.
        event: Evento con payload {'woman': Person}.
    """
    from distributions import sample_num_babies, sample_baby_sex, time_to_death
    from person import Person

    woman = event.payload["woman"]

    # Si la madre murió cancelar el parto
    if not woman.alive:
        return

    num_babies = sample_num_babies()

    # Registrar el parto (incluso si num_babies == 0: embarazo sin nacidos vivos)
    woman.register_birth(num_babies)  # resetea pregnant y actualiza conteo
    sim.stats.record_birth(event.time, num_babies)

    for _ in range(num_babies):
        is_male = sample_baby_sex()
        baby = Person(is_male=is_male, age=0.0, birth_time=event.time)

        # Añadir bebé a la población viva
        sim.population.add(baby)
        sim.stats.record_person_added(event.time, baby)

        # Programar muerte del bebé
        dt_death = time_to_death(baby.age, baby.is_male)
        sim.schedule(Event(
            time=event.time + dt_death,
            event_type=EventType.DEATH,
            payload={"person": baby}
        ))

        # Programar primer cumpleaños
        if event.time + 1.0 < sim.end_time:
            sim.schedule(Event(
                time=event.time + 1.0,
                event_type=EventType.BIRTHDAY,
                payload={"person": baby}
            ))

    # Reprogramar siguiente intento de embarazo solo si:
    #   - La mujer sigue viva y en pareja (puede haber muerto durante el parto)
    #   - El próximo intento ocurre ANTES de la ruptura prevista
    # Así se mantiene la garantía de que handle_pregnancy nunca necesita
    # verificar si la mujer tiene pareja.
    from distributions import time_to_pregnancy
    if woman.alive and woman.is_in_couple:
        dt = time_to_pregnancy(woman.age)
        t_next = event.time + dt
        if t_next < woman.couple_breakup_time:
            sim.schedule(Event(
                time=t_next,
                event_type=EventType.PREGNANCY,
                payload={"woman": woman}
            ))


def handle_breakup(sim: Simulation, event: Event) -> None:
    """Procesa la ruptura de una pareja.

    Este evento fue agendado en el momento en que la pareja se formó,
    con el tiempo de ruptura ya sorteado. No hay chequeos periódicos:
    el evento simplemente ejecuta la separación si la pareja sigue activa
    (ambos vivos y todavía juntos).

    Si alguno falleció antes de que llegara este evento, handle_death
    ya habrá disuelto la pareja; en ese caso se ignora el evento.

    Args:
        sim: Instancia de la simulación.
        event: Evento con payload {'person': Person} (cualquier miembro de la pareja).
    """
    from distributions import sample_alone_period

    person = event.payload["person"]

    # Si la pareja ya no está activa (muerte previa de alguno), ignorar
    if not person.alive or not person.is_in_couple:
        return

    partner = person.partner
    p1, p2 = person, partner

    # Ejecutar la ruptura
    person.break_up()  # actualiza ambos a GRIEVING y limpia referencias

    # Ambos eran una pareja, no estaban en singles
    # sim.singles.discard(p1)
    # sim.singles.discard(p2)
    sim.stats.record_breakup(event.time)

    # Programar fin del período de soledad para cada sobreviviente
    for p in (p1, p2):
        if p is not None and p.alive:
            duration = sample_alone_period(p.age)
            sim.schedule(Event(
                time=event.time + duration,
                event_type=EventType.SOLO_END,
                payload={"person": p}
            ))


def handle_solo_end(sim: Simulation, event: Event) -> None:
    """Procesa el fin del período de soledad de una persona.

    La persona pasa a estado SINGLE y se añade al conjunto de solteros
    disponibles para buscar pareja.

    Args:
        sim: Instancia de la simulación.
        event: Evento con payload {'person': Person}.
    """
    person = event.payload["person"]

    if not person.alive:
        return

    # Sólo actuar si sigue en período de duelo (no fue recaptado antes)
    if person.is_grieving:
        person.end_grieving()
        person.solo_since = event.time  # nuevo período de soltería comienza aquí
        sim.singles.add(person)
        # Iniciar cadena de búsqueda de pareja si la persona lo desea
        if person.partner_desire:
            _schedule_seek_partner(sim, event.time, person)


def _schedule_seek_partner(
    sim: Simulation, current_time: float, person: "Person"
) -> None:
    """Agenda el próximo evento SEEK_PARTNER para una persona soltera.

    Establece seek_in_progress=True para evitar duplicados en la cola.
    No agenda nada si el tiempo resultante supera el horizonte.

    Args:
        sim: Instancia de la simulación.
        current_time: Tiempo actual de simulación.
        person: Persona soltera para la que se agenda la búsqueda.
    """
    from distributions import time_to_seek_partner
    t_next = current_time + time_to_seek_partner()
    if t_next < sim.end_time:
        person.seek_in_progress = True
        sim.schedule(Event(
            time=t_next,
            event_type=EventType.SEEK_PARTNER,
            payload={"person": person}
        ))


def handle_seek_partner(sim: Simulation, event: Event) -> None:
    """Procesa el intento de una persona soltera de encontrar pareja.

    Cada persona soltera que desea pareja agenda de forma autónoma sus propios
    eventos SEEK_PARTNER con tiempos sorteados de Exp(MEAN_SEEK_INTERVAL).
    Cuando el evento ocurre:
      - Se elige un candidato aleatorio del sexo opuesto que también desea pareja.
      - Se llama a will_form_couple(ages) para decidir si forman pareja.
      - Si forman pareja: se ejecuta la unión, se agenda BREAKUP y primer PREGNANCY.
      - Si no hay candidatos o el intento falla: se reagenda el próximo encuentro.

    Este modelo reemplaza el checkeo global periódico (PARTNER_CHECK) por un
    proceso continuo individual, eliminando la discretización temporal.

    Args:
        sim: Instancia de la simulación.
        event: Evento con payload {'person': Person}.
    """
    import random
    from distributions import will_form_couple, time_to_breakup, time_to_pregnancy

    person = event.payload["person"]
    person.seek_in_progress = False  # el evento se está procesando; resetear flag

    # Cancelar si la persona murió, ya no es soltera, o perdió el deseo de pareja
    if not person.alive or not person.is_single or not person.partner_desire:
        return  # no reagendar → cadena se detiene naturalmente

    # Candidatos: solteros vivos del sexo opuesto que también desean pareja
    candidates = [
        p for p in sim.singles
        if p.alive and p.is_single and p.partner_desire and p.is_male != person.is_male
    ]

    if not candidates:
        # Sin candidatos disponibles; reagendar e intentar más tarde
        _schedule_seek_partner(sim, event.time, person)
        return

    candidate = random.choice(candidates)
    man = person if person.is_male else candidate
    woman = candidate if person.is_male else person

    if will_form_couple(man.age, woman.age):
        # Registrar duración del período de soltería de ambos antes de unirlos
        sim.stats.record_solo_ended(event.time - man.solo_since)
        sim.stats.record_solo_ended(event.time - woman.solo_since)

        man.form_couple(woman)
        sim.singles.discard(man)
        sim.singles.discard(woman)
        sim.stats.record_couple_formed(event.time)

        # Sortear tiempo de ruptura (un único sorteo al formarse la pareja)
        dt_breakup = time_to_breakup()
        t_breakup_abs = event.time + dt_breakup
        man.couple_breakup_time = t_breakup_abs
        woman.couple_breakup_time = t_breakup_abs

        sim.schedule(Event(
            time=t_breakup_abs,
            event_type=EventType.BREAKUP,
            payload={"person": man}
        ))

        # Primer intento de embarazo, solo si ocurre antes de la ruptura prevista
        dt_preg = time_to_pregnancy(woman.age)
        t_preg = event.time + dt_preg
        if t_preg < t_breakup_abs:
            sim.schedule(Event(
                time=t_preg,
                event_type=EventType.PREGNANCY,
                payload={"woman": woman}
            ))
    else:
        # Intento fallido; reagendar próximo encuentro para esta persona
        _schedule_seek_partner(sim, event.time, person)


# ---------------------------------------------------------------------------
# Mapa de dispatch: EventType → handler
# ---------------------------------------------------------------------------

EVENT_HANDLERS = {
    EventType.DEATH:         handle_death,
    EventType.BIRTHDAY:      handle_birthday,
    EventType.PREGNANCY:     handle_pregnancy,
    EventType.BIRTH:         handle_birth,
    EventType.BREAKUP:       handle_breakup,
    EventType.SOLO_END:      handle_solo_end,
    EventType.SEEK_PARTNER:  handle_seek_partner,
}
