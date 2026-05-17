"""
distributions.py
================
Módulo de generación de variables aleatorias para la simulación del Poblado en Evolución.

Todas las distribuciones se implementan manualmente a partir de random.random() y
random.uniform(), sin usar bibliotecas estadísticas externas, tal como exigen las
orientaciones del proyecto.

Las tablas de probabilidades se basan directamente en los datos del enunciado.
"""

import random
import math


# ---------------------------------------------------------------------------
# Utilidades base
# ---------------------------------------------------------------------------

def uniform(a: float, b: float) -> float:
    """Genera una variable aleatoria con distribución Uniforme en [a, b).

    Args:
        a: Límite inferior del intervalo.
        b: Límite superior del intervalo.

    Returns:
        Un float en [a, b).
    """
    return a + (b - a) * random.random()


def exponential(mean: float) -> float:
    """Genera una variable aleatoria con distribución Exponencial.

    Usa el método de la transformada inversa: X = -mean * ln(U),
    donde U ~ Uniform(0, 1).

    Args:
        mean: Media (1/λ) de la distribución en las mismas unidades de tiempo
              que se use en la simulación (años).

    Returns:
        Un float positivo muestreado de la distribución exponencial.
    """
    u = random.random()
    # Evitar log(0)
    while u == 0.0:
        u = random.random()
    return -mean * math.log(u)


def bernoulli(p: float) -> bool:
    """Genera una variable aleatoria de Bernoulli.

    Args:
        p: Probabilidad de éxito (True).

    Returns:
        True con probabilidad p, False con probabilidad 1-p.
    """
    return random.random() < p


# ---------------------------------------------------------------------------
# Probabilidad de fallecimiento según edad y sexo
# ---------------------------------------------------------------------------

# Tabla: (edad_min, edad_max, prob_hombre, prob_mujer)
_DEATH_TABLE = [
    (0,   12,  0.25, 0.25),
    (12,  45,  0.10, 0.15),
    (45,  76,  0.30, 0.35),
    (76, 125,  0.70, 0.65),
]


def death_probability(age: float, is_male: bool) -> float:
    """Retorna la probabilidad anual de fallecimiento según edad y sexo.

    Basado en la tabla del enunciado. Si la edad cae fuera de todos los rangos
    se retorna 1.0 (certeza de muerte).

    Args:
        age: Edad actual de la persona en años.
        is_male: True si la persona es hombre, False si es mujer.

    Returns:
        Probabilidad (float en [0, 1]) de fallecer en el próximo año.
    """
    for age_min, age_max, prob_m, prob_f in _DEATH_TABLE:
        if age_min <= age < age_max:
            return prob_m if is_male else prob_f
    # Mayor de 125 años: fallece con certeza
    return 1.0


def time_to_death(age: float, is_male: bool) -> float:
    """Muestrea el tiempo restante de vida de una persona avanzando tramo a tramo.

    La tabla da la probabilidad de morir *dentro de cada tramo de edad*.
    Se simula secuencialmente cada tramo: en cada uno se sortea con Bernoulli
    si la persona muere. Si muere en el tramo, la muerte ocurre en un tiempo
    uniforme dentro de dicho tramo. Si sobrevive, se avanza al siguiente tramo.

    Args:
        age: Edad actual de la persona (puede estar en medio de un tramo).
        is_male: Sexo de la persona.

    Returns:
        Tiempo en años hasta el fallecimiento (siempre positivo).
    """
    current_age = age
    for age_min, age_max, prob_m, prob_f in _DEATH_TABLE:
        # Saltar tramos ya superados
        if current_age >= age_max:
            continue

        p = prob_m if is_male else prob_f
        # Parte del tramo que la persona aún tiene por vivir
        remaining_in_tranche = age_max - current_age

        if bernoulli(p):
            # Muere dentro de este tramo: tiempo uniforme en lo que queda del tramo
            time_in_tranche = uniform(0.0, remaining_in_tranche)
            return current_age - age + time_in_tranche  # tiempo desde ahora
        else:
            # Sobrevive el tramo completo; avanzar al inicio del siguiente
            current_age = age_max

    # Sobrevivió todos los tramos (muy raro): muere a los 125 años
    return 125.0 - age if 125.0 > age else 1.0


# ---------------------------------------------------------------------------
# Probabilidad de embarazo según edad
# ---------------------------------------------------------------------------

# Tabla: (edad_min, edad_max, prob_embarazo)
_PREGNANCY_TABLE = [
    (12,  15, 0.20),
    (15,  21, 0.45),
    (21,  35, 0.80),
    (35,  45, 0.40),
    (45,  60, 0.20),
    (60, 125, 0.05),
]


def pregnancy_probability(age: float) -> float:
    """Retorna la probabilidad de que una mujer quede embarazada según su edad.

    Args:
        age: Edad de la mujer en años.

    Returns:
        Probabilidad de embarazo (float en [0, 1]). Retorna 0.0 si la edad
        está fuera de los rangos de la tabla.
    """
    for age_min, age_max, prob in _PREGNANCY_TABLE:
        if age_min <= age < age_max:
            return prob
    return 0.0


def time_to_pregnancy(age: float) -> float:
    """Muestrea el tiempo hasta el próximo intento de embarazo.

    Se modela como una exponencial cuya media depende de la probabilidad de
    embarazo: a mayor probabilidad, menor tiempo de espera.

    Args:
        age: Edad actual de la mujer.

    Returns:
        Tiempo en años hasta el próximo intento de embarazo. Retorna un valor
        muy grande si la probabilidad es 0.
    """
    p = pregnancy_probability(age)
    if p <= 0.0:
        return 999.0  # No puede quedar embarazada
    # Media en años: cuanto más alta la probabilidad, más frecuente el evento
    return exponential(1.0 / p)


# ---------------------------------------------------------------------------
# Número de hijos deseados
# ---------------------------------------------------------------------------

# Cada fila de la tabla es un ensayo de Bernoulli independiente:
# la probabilidad representa si la persona desea ese hijo adicional.
# El número total deseado es la suma de todos los éxitos.
# Valor esperado: 0.60 + 0.75 + 0.35 + 0.20 + 0.10 + 0.05 = 2.05 hijos.
_CHILDREN_BERNOULLIS = [0.60, 0.75, 0.35, 0.20, 0.10, 0.05]


def desired_children() -> int:
    """Muestrea el número de hijos que una persona desea tener.

    Cada entrada de la tabla del enunciado es un ensayo de Bernoulli
    independiente: la probabilidad de la fila N indica si la persona
    desearía tener N hijos. Se corren los 6 Bernoullis de forma
    independiente y el resultado es el MAYOR número cuyo Bernoulli
    resultó exitoso.

    Ejemplo:
        B(1)=0.60 → éxito  ✓
        B(2)=0.75 → éxito  ✓
        B(3)=0.35 → fallo  ✗
        B(4)=0.20 → éxito  ✓  ← máximo éxito
        B(5)=0.10 → fallo  ✗
        B(6)=0.05 → fallo  ✗
        → desea 4 hijos

    Si ningún Bernoulli resulta éxito (todos fallan), la persona no
    desea tener hijos (retorna 0).

    Returns:
        Número entero de hijos deseados (entre 0 y 6, donde 6 representa
        "más de 5").
    """
    # Enumerar desde 1: el índice i+1 corresponde al número de hijos
    successes = [i + 1 for i, p in enumerate(_CHILDREN_BERNOULLIS) if bernoulli(p)]
    return max(successes) if successes else 0


# ---------------------------------------------------------------------------
# Deseo de tener pareja según edad
# ---------------------------------------------------------------------------

# Tabla: (edad_min, edad_max, prob_querer_pareja)
_PARTNER_WISH_TABLE = [
    (12,  15, 0.60),
    (15,  21, 0.65),
    (21,  35, 0.80),
    (35,  45, 0.60),
    (45,  60, 0.50),
    (60, 125, 0.20),
]

# Umbrales de edad donde cambia el tramo. Se derivan automáticamente de
# _PARTNER_WISH_TABLE para que ambos estén siempre sincronizados.
PARTNER_WISH_THRESHOLDS: list[int] = [row[0] for row in _PARTNER_WISH_TABLE]

# Tiempo medio (años) entre encuentros de un soltero con un posible candidato/a.
# Parámetro de la Exponencial usada en el evento SEEK_PARTNER de cada persona.
# Cambiar este valor para calibrar la frecuencia de búsqueda de pareja.
MEAN_SEEK_INTERVAL: float = 0.50   # ≈ 3 meses de media entre encuentros


def time_to_seek_partner() -> float:
    """Tiempo hasta el próximo encuentro con un posible candidato/a.

    Modela el tiempo que tarda un soltero en toparse con alguien del sexo
    opuesto con quien intentar formar pareja. Distribuido Exp(MEAN_SEEK_INTERVAL).

    Returns:
        Tiempo en años (float positivo).
    """
    return exponential(MEAN_SEEK_INTERVAL)


def wants_partner_probability(age: float) -> float:
    """Retorna la probabilidad de que una persona desee tener pareja según su edad.

    Args:
        age: Edad de la persona en años.

    Returns:
        Probabilidad (float en [0, 1]). Retorna 0.0 fuera de los rangos.
    """
    for age_min, age_max, prob in _PARTNER_WISH_TABLE:
        if age_min <= age < age_max:
            return prob
    return 0.0


def wants_partner(age: float) -> bool:
    """Determina aleatoriamente si una persona desea tener pareja.

    Args:
        age: Edad de la persona en años.

    Returns:
        True si desea pareja, False en caso contrario.
    """
    return bernoulli(wants_partner_probability(age))


# ---------------------------------------------------------------------------
# Probabilidad de establecer pareja según diferencia de edad
# ---------------------------------------------------------------------------

# Tabla: (dif_min, dif_max, prob_pareja)
_COUPLE_PROB_TABLE = [
    (0,   5,  0.45),
    (5,  10,  0.40),
    (10, 15,  0.35),
    (15, 20,  0.25),
    (20, 999, 0.15),  # "20 o más"
]


def couple_formation_probability(age1: float, age2: float) -> float:
    """Retorna la probabilidad de que dos personas formen pareja según su diferencia de edad.

    Args:
        age1: Edad de la primera persona.
        age2: Edad de la segunda persona.

    Returns:
        Probabilidad (float en [0, 1]) de formación de pareja.
    """
    diff = abs(age1 - age2)
    for dif_min, dif_max, prob in _COUPLE_PROB_TABLE:
        if dif_min <= diff < dif_max:
            return prob
    return 0.15  # fallback: diferencia >= 20


def will_form_couple(age1: float, age2: float) -> bool:
    """Determina aleatoriamente si dos personas formarán pareja.

    Args:
        age1: Edad de la primera persona.
        age2: Edad de la segunda persona.

    Returns:
        True si forman pareja, False en caso contrario.
    """
    return bernoulli(couple_formation_probability(age1, age2))


# ---------------------------------------------------------------------------
# Ruptura de pareja
# ---------------------------------------------------------------------------

BREAKUP_PROBABILITY = 0.2   # Probabilidad anual de ruptura (Bernoulli por año)


def time_to_breakup() -> float:
    """Muestrea el tiempo hasta la ruptura de una pareja usando la Transformada Inversa
    de la distribución Geométrica continua.

    Interpretación:
    Cada año existe una probabilidad p=0.2 de que la pareja se separe 
    (ensayo de Bernoulli anual independiente). El tiempo hasta el 
    primer éxito (primera separación) sigue una distribución
    Geométrica de parámetro p.

    Aplicando la Transformada Inversa sobre la CDF de la Geométrica continua:

        CDF: P(T ≤ t) = 1 - (1 - p)^t
        Despejando T:  T = ln(U) / ln(1 - p),  con U ~ Uniform(0, 1)

    Args:
        (ninguno)

    Returns:
        Tiempo en años (float positivo) hasta la ruptura de la pareja.
        Con p=0.2, el valor esperado es 1/p = 5 años.
    """
    u = random.random()
    while u == 0.0:          # evitar log(0)
        u = random.random()
    # Transformada inversa de la Geométrica continua
    return math.log(u) / math.log(1.0 - BREAKUP_PROBABILITY)


# ---------------------------------------------------------------------------
# Período de soledad post-ruptura / viudez
# ---------------------------------------------------------------------------

# Tabla: (edad_min, edad_max, media_en_años)
_ALONE_PERIOD_TABLE = [
    (12,  15, 3  / 12),   # 3 meses
    (15,  21, 6  / 12),   # 6 meses
    (21,  35, 6  / 12),   # 6 meses
    (35,  45, 1.0),       # 1 año
    (45,  60, 2.0),       # 2 años
    (60, 125, 4.0),       # 4 años
]


def alone_period_mean(age: float) -> float:
    """Retorna la media del período de soledad post-ruptura según la edad.

    Args:
        age: Edad de la persona al momento de quedar sola.

    Returns:
        Media en años del período de soledad. Retorna 1.0 si fuera de rango.
    """
    for age_min, age_max, mean in _ALONE_PERIOD_TABLE:
        if age_min <= age < age_max:
            return mean
    return 1.0  # fallback


def sample_alone_period(age: float) -> float:
    """Muestrea el período de soledad de una persona tras una ruptura o viudez.

    Distribuye exponencial con media dependiente de la edad.

    Args:
        age: Edad de la persona al momento de la ruptura o viudez.

    Returns:
        Duración en años del período de soledad.
    """
    mean = alone_period_mean(age)
    return exponential(mean)


# ---------------------------------------------------------------------------
# Embarazo múltiple
# ---------------------------------------------------------------------------

# Cada fila de la tabla es un ensayo de Bernoulli independiente:
# la probabilidad indica si ese bebé adicional nace en el parto.
# El total de bebés = suma de éxitos, con mínimo 1 (es un parto real).
# Valor esperado: 0.70 + 0.18 + 0.08 + 0.04 + 0.02 = 1.02 bebés.
_BIRTH_BERNOULLIS = [0.70, 0.18, 0.08, 0.04, 0.02]


def sample_num_babies() -> int:
    """Muestrea el número de bebés nacidos en un parto.

    Cada entrada de la tabla del enunciado es un ensayo de Bernoulli
    independiente: la probabilidad de la fila N indica si nacen N bebés.
    Se corren los 5 Bernoullis de forma independiente y el resultado es
    el MAYOR número cuyo Bernoulli resultó éxito (Opción A: máximo éxito).

    Ejemplo:
        B(1)=0.70 → éxito  ✓
        B(2)=0.18 → fallo  ✗
        B(3)=0.08 → éxito  ✓  ← máximo éxito
        B(4)=0.04 → fallo  ✗
        B(5)=0.02 → fallo  ✗
        → nacen 3 bebés

    Si todas las Bernoullis fallan (todos los ensayos fallan), se considera 
    un embarazo sin nacidos vivos y se retorna 0.

    Returns:
        Número entero de bebés nacidos (entre 0 y 5).
    """
    # Enumerar desde 1: el índice i+1 corresponde al número de bebés
    successes = [i + 1 for i, p in enumerate(_BIRTH_BERNOULLIS) if bernoulli(p)]
    return max(successes) if successes else 0  # 0 = embarazo sin nacidos vivos


def sample_baby_sex() -> bool:
    """Determina el sexo de un bebé recién nacido.

    Probabilidad uniforme 0.5 para cada sexo.

    Returns:
        True si el bebé es hombre, False si es mujer.
    """
    return bernoulli(0.5)


# ---------------------------------------------------------------------------
# Duración del embarazo
# ---------------------------------------------------------------------------

PREGNANCY_DURATION_YEARS = 9 / 12  # 9 meses en años


def pregnancy_duration() -> float:
    """Retorna la duración estándar de un embarazo en años (9 meses -> 9 / 12).

    Returns:
        Duración del embarazo en años.
    """
    return PREGNANCY_DURATION_YEARS
