# Proyectos de Simulación basada en Eventos Discretos
#### Colectivo de Simulación


## Orientaciones Generales
Cada estudiante debe realizar la implementación de uno de los problemas presentes (seleccionado por el estudiante).
Esta implementación puede ser realizada en cualquier lenguaje de programación y debe entregarse por correo electrónico al profesor de conferencia (<yudy@matcom.uh.cu>) antes del domingo 21 de noviembre a las 12:00 de la noche. Junto a la implementación (en la implementación se deben programar también toda la generación de variables aleatorias)-que debe estar en un repositorio de Github-, se debe enviar un informe de trabajo (un documento en formato pdf).

El informe de trabajo debe contener los siguientes elementos:

1. Generales del Estudiante (Nombre y apellidos, grupo, etc)

2. Orden del Problema Asignado
      
3. Principales Ideas seguidas para la solución del problema
      
4. Modelo de Simulación de Eventos Discretos desarrollado para resolver el problema

5. Consideraciones obtenidas a partir de la ejecución de las simulaciones del problema

6. El enlace al repositorio del proyecto en Github

# Problema escogido: Poblado en Evolución
Se dese conocer la evolución de la población de una determinada región.
Se conoce que la probabilidad de fallecer de una persona distribuye uniforme y se corresponde, según su edad y sexo, con la siguiente tabla:

   | Edad    |  Hombre   |  Mujer
   |---------|-----------|-------
   | 0-12    | 0.25      | 0.25
   | 12-45   | 0.1       | 0.15
   | 45-76   | 0.3       | 0.35
   | 76-125  | 0.7       | 0.65 

Del mismo modo, se conoce que la probabilidad de una mujer se embarace es uniforme y está relacionada con la edad:

   | Edad    | Probabilidad Embarazarce
   |---------|----------------------
   | 12-15   | 0.2
   | 15-21   | 0.45
   | 21-35   | 0.8
   | 35-45   | 0.4
   | 45-60   | 0.2
   | 60-125  | 0.05

Para que una mujer quede embarazada debe tener pareja y no haber tenido el número máximo de hijos que deseaba tener ella o su pareja en ese momento.
El número de hijos que cada persona desea tener distribuye uniforme según la tabla siguiente:

    | Número  | Probabilidad |
    |---------|--------------|
    | 1       | 0.6          |
    | 2       | 0.75         |
    | 3       | 0.35         |
    | 4       | 0.2          |
    | 5       | 0.1          |
    |más de 5 | 0.05         |

Para que dos personas sean pareja deben estar solas en ese instante y deben desear tener pareja. El desear tener pareja está relacionado con la edad:

    |Edad   |   Probabilidad Querer Pareja
    |-------|---------------------------
    | 12-15 | 0.6
    | 15-21 | 0.65
    | 21-35 | 0.8
    | 35-45 | 0.6
    | 45-60 | 0.5
    | 60-125 | 0.2

Si dos personas de diferente sexo están solas y ambas desean tener pareja entonces la probabilidad de volverse pareja está relacionada con la diferencia de edad:

    | Diferencia de Edad | Probabilidad Establecer Pareja |
    |-------------------|-------------------------------|
    | 0-5               | 0.45                          |
    | 5-10              | 0.4                           |
    | 10-15             | 0.35                          |
    | 15-20             | 0.25                          |
    | 20 o más          | 0.15                          |

Cuando dos personas están en pareja la probabilidad de que ocurra una ruptura distribuye uniforme y es de 0.2. Cuando una persona se separa, o enviuda, necesita estar sola por un perı́odo de tiempo que distribuye exponencial con un parámetro que está relacionado con la edad:
    |Edad  |           λ
    |-------|----------------
    | 12-15 | 3 meses
    | 15-21 | 6 meses
    | 21-35 | 6 meses
    | 35-45 | 1 año
    | 45-60 | 2 años
    | 60-125 | 4 años

Cuando están dadas todas las condiciones y una mujer queda embarazada puede tener o no un embarazo múltiple y esto distribuye uniforme acorde a las
probabilidades siguientes:

    |Número de Bebés|   Probabilidad
    |----------------|--------------
    | 1              | 0.7
    | 2              | 0.18
    | 3              | 0.08
    | 4              | 0.04
    | 5              | 0.02

La probabilidad del sexo de cada bebé nacido es uniforme 0,5.

Asumiendo que se tiene una población inicial de M mujeres y H hombres y que cada poblador, en el instante incial, tiene una edad que distribuye uniforme (U(0,100)). Realice un proceso de simulación para determinar como evoluciona
la población en un perı́odo de 100 años.