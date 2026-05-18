# Poblado en Evolución — Simulación de Eventos Discretos

Proyecto de simulación basada en eventos discretos (SED/DES) que modela la evolución demográfica de una población hipotética a lo largo de 100 años.

**Asignatura:** Simulación · Facultad de Matemática y Computación, Universidad de La Habana  
**Autor:** David Sánchez Iglesias (C411)  
**Curso:** 2025–2026

---

## Descripción

El simulador reproduce la dinámica poblacional de una región partiendo de $H$ hombres y $M$ mujeres con edades uniformes en $[0, 100)$ años. Los eventos que dirigen la evolución son:

- **Fallecimiento** — probabilidad por tramos de edad y sexo (tablas del enunciado).
- **Cumpleaños** — actualiza la edad y re-evalúa atributos dependientes del tramo.
- **Búsqueda de pareja** — proceso continuo individual; cada soltero agenda su propio intento con $\Delta t \sim \text{Exp}(0.5)$.
- **Formación de pareja** — se acepta según la diferencia de edad entre los candidatos.
- **Ruptura** — ocurre con probabilidad dependiente del tiempo de convivencia.
- **Embarazo / Nacimiento** — condicionado a la edad de la madre, su pareja y el número de hijos deseados.

Toda la generación de variables aleatorias se implementa desde cero (sin bibliotecas externas de estadística) usando la transformada inversa sobre `random.random()`.

---

## Estructura del proyecto

```
.
├── distributions.py   # Generación de todas las variables aleatorias del modelo
├── person.py          # Clase Person: estado completo de un individuo
├── events.py          # EventType, Event y handlers del motor DES
├── simulation.py      # Motor principal: inicialización, cola de eventos y bucle DES
├── stats.py           # Recolección de métricas y generación de visualizaciones
├── main.py            # Punto de entrada con interfaz de línea de comandos
├── batch_run.py       # Réplicas múltiples para validación estadística
├── gen_figures.py     # Genera las 14 figuras del Anexo A del informe
└── informe/
    ├── main.tex       # Fuente LaTeX del informe
    ├── main.pdf       # Informe compilado (30 páginas)
    └── figures/       # Figuras PNG generadas automáticamente
```

---

## Requisitos

- Python 3.10+
- [matplotlib](https://matplotlib.org/) (solo para visualizaciones)

```bash
pip install matplotlib
```

---

## Uso

### Corrida única

```bash
python main.py                              # parámetros por defecto (H=100, M=100, 100 años)
python main.py --men 200 --women 200 --seed 42
python main.py --men 50 --women 60 --years 80 --output resultado.png --no-plot
```

| Argumento | Descripción | Por defecto |
|-----------|-------------|-------------|
| `--men`   | Número inicial de hombres | `100` |
| `--women` | Número inicial de mujeres | `100` |
| `--years` | Duración de la simulación (años) | `100` |
| `--seed`  | Semilla aleatoria (reproducibilidad) | `None` |
| `--output`| Ruta para guardar la gráfica | — |
| `--no-plot` | Suprime la gráfica interactiva | — |

### Réplicas múltiples (batch)

```bash
python batch_run.py
```

Ejecuta 30 réplicas independientes con el escenario base (H=100, M=100) y muestra estadísticos de resumen (media, IC 95 %, mín/máx) para cada métrica.

### Generar todas las figuras del informe

```bash
python gen_figures.py
```

Recorre los 7 escenarios del análisis (variación de H y de M), genera una corrida individual y un batch de 30 réplicas por escenario, e imprime los estadísticos en formato LaTeX. Las 14 figuras resultantes se guardan en `informe/figures/`.

---

## Informe

El informe completo (30 páginas) se encuentra en [`informe/main.pdf`](informe/main.pdf) e incluye:

- Descripción del modelo SED y sus eventos.
- Resultados de corrida única para cada escenario.
- Validación estadística con 30 réplicas e IC al 95 % (distribución *t* de Student).
- Anexo A con las 14 figuras de simulación.

Para recompilar el PDF desde las fuentes LaTeX:

```bash
cd informe
rm -f main.aux main.toc main.out main.lof main.lot
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # segunda pasada para TOC y referencias
```