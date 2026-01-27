# HOWTO: Pandemic Experiment Scenario (PES)

## Introducción

El **Pandemic Experiment Scenario (PES)** es una simulación interactiva de mitigación de pandemias donde un **Agente de Aprendizaje por Refuerzo (RL-Agent)** toma decisiones autónomas para asignar recursos limitados a ciudades en peligro, minimizando la severidad de una pandemia.

Este documento explica el funcionamiento completo de la simulación para científicos de datos sin experiencia en simulación de pandemias.

---

## 1. Concepto Fundamental

### El Problema del Agente

El agente enfrenta un dilema clásico de **asignación de recursos limitados**:

- **Objetivo**: Minimizar la severidad total de una pandemia distribuida en múltiples ciudades
- **Restricción**: Tiene un presupuesto fijo de recursos por secuencia
- **Decisión**: En cada trial (ciudad), decidir cuántos recursos asignar (0-10)

### Fórmula Fundamental

La severidad actualizada de cada ciudad se calcula:

$$\text{NewSeverity}_c = \text{SEVERITY\_MULTIPLIER} \times \text{InitialSeverity}_c - \text{RESPONSE\_MULTIPLIER} \times \text{ResourcesAllocated}_c$$

Donde:
- `SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER` (amplifica el crecimiento natural)
- `RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER` (efecto de recursos asignados)
- Por defecto: `PANDEMIC_PARAMETER = 0.4`

**Interpretación**: Más recursos → menor severidad final (efecto negativo), pero la severidad base crece sin intervención.

---

## 2. Estructura Jerárquica de la Simulación

### Diagrama de Nidificación

```
Experimento (1)
  ├─ Bloque (8)
  │    ├─ Secuencia / Mapa (8)
  │    │    ├─ Trial / Ciudad (3-10)
  │    │    │    └─ Decision de Recursos (0-10)
```

### Detalles por Nivel

#### 2.1 Experimento Completo
- **Duración Total**: ~360 trials (64 secuencias × ~5-7 trials promedio)
- **Archivo de Salida**: `PES_responses_{SubjectId}.csv`
- **Archivo de Resultados**: `PES_performance_plot_{SubjectId}.png`, `PES_performance_stats_{SubjectId}.png`

#### 2.2 Bloque (Block)
- **Cantidad**: 8 bloques en el experimento
- **Propósito**: Agrupación organizativa (puede representar fases)
- **Referencia en Código**: 
  ```python
  for CurrentBlockIndex in range(NUM_BLOCKS):  # En __main__.py línea ~390
  ```

#### 2.3 Secuencia (Sequence) / Mapa (Map)
- **Cantidad**: 8 secuencias por bloque = 64 secuencias totales
- **Duración Variable**: 3-10 trials por secuencia
- **Selección de Mapa**: Asignación aleatoria de mapas del 0 al 8 (indices `MapIndices__blocks_x_sequences__2darray`)
- **Referencia en Código**:
  ```python
  for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate(CurrentBlockMapIndices):
      # Procesar secuencia (línea ~420)
  ```

#### 2.4 Trial (City)
- **Definición**: Presentación de una ciudad individual durante una secuencia
- **Variable por Secuencia**: Cada secuencia tiene un número diferente de ciudades (trials)
- **Severidad Inicial**: Valor aleatorio 2-8 (inclusive)
- **Recursos Disponibles**: 49 por secuencia (9 pre-asignados a ciudades iniciales + 40 libres)
- **Referencia en Código**:
  ```python
  for AbsoluteTrialCount in range(number_of_trials):
      # Procesar trial individual (línea ~550)
  ```

---

## 3. Flujo de Ejecución Detallado

### 3.1 Inicialización (main() - Líneas 100-370)

```
1. Validación de paquete (__main__.py:15-25)
   └─ Verifica que el módulo se ejecute con: python3 -m PES

2. Cargar configuración (__init__.py:60-130)
   ├─ Leer PES/config/CONFIG.py
   ├─ Inicializar variables globales
   └─ Configurar rutas de entrada/salida

3. Crear archivo de respuestas
   └─ __main__.py:350: Respuestas_filehandle = open(..., 'w')

4. Generar índices de mapas y secuencias (__main__.py:230-290)
   ├─ MapIndices__blocks_x_sequences__2darray[blk, seq] = random(0-8)
   └─ NumTrials__blocks_x_sequences__2darray[blk, seq] = random(3-10)

5. Generar severidades iniciales (__main__.py:310-330)
   ├─ Si RANDOM_INITIAL_SEVERITY: generar random
   ├─ Si no: cargar de CSV (INPUTS_PATH/initial_severity.csv)
   └─ Rango: 2-8 (inclusive) para 360 trials

6. Inicializar listas de rendimiento (__main__.py:360-365)
   ├─ MyPerformances = [] (rendimiento del agente)
   └─ AllPerformances[0] y AllPerformances[1] (agent + aggregated)
```

**Código Clave**:
```python
# Línea 240-320 en __main__.py
NumTrials__blocks_x_sequences__2darray = numpy.full((NUM_BLOCKS, NUM_SEQUENCES), 0)
MapIndices__blocks_x_sequences__2darray = numpy.full((NUM_BLOCKS, NUM_SEQUENCES), 0)
first_severity = numpy.loadtxt(InitialSeverityCsv)  # O generar random
```

### 3.2 Loop Principal: Bloque → Secuencia → Trial

```python
# Línea ~390 en __main__.py
for CurrentBlockIndex in range(NUM_BLOCKS):                        # 8 bloques
    for CurrentSequenceIndex in range(NUM_SEQUENCES):              # 8 secuencias
        number_of_trials = NumTrials__blocks_x_sequences__2darray[CurrentBlockIndex, CurrentSequenceIndex]
        
        for AbsoluteTrialCount in range(number_of_trials):         # N trials
            # === AQUÍ SUCEDE LA MAGIA ===
```

### 3.3 Decisión del Agente por Trial (Línea ~550-660)

#### Paso 1: Obtener Observación

```python
# Línea 550-580 en __main__.py
InitialSeveritiesInSequence = first_severity[AbsoluteTrialCount]
MyMessage, rt_hold, rt_release, confidence = pygameMediator.provide_rl_agent_response(...)

# provide_rl_agent_response() → Agent.agent_meta_cognitive()
# Retorna: (response_allocation, confidence, rt_hold, rt_release)
```

**Estado Observado**:
- Severidad actual de la ciudad
- Recursos disponibles restantes
- Número de trial actual

#### Paso 2: Generar Respuesta del Agente

Referencia: `PES/src/Agent.py`

```python
def agent_meta_cognitive(action, output_value_range, resources_left, response_timeout):
    """
    El agente mapea la distancia a la frontera de decisión como confianza.
    
    Entrada:
        - action: salida continua de la red neuronal entrenada (0-10)
        - output_value_range: rango de acciones posibles (0-10)
        - resources_left: recursos disponibles aún
        - response_timeout: timeout en ms (10000 = 10s)
    
    Salida:
        - response: recursos enteros asignados (0-resources_left)
        - confidence: confianza en la decisión (0-1)
        - rt_hold: tiempo de presión del mouse (ms)
        - rt_release: tiempo de liberación del mouse (ms)
    """
    
    # 1. Calcular distancia a centro de decisión más cercano
    centers = [0, 1, 2, ..., 10]
    closer = |centers - action|
    closervalue = centers[argmin(closer)]
    
    # 2. Clip de acción a recursos disponibles
    action = clip(action, 1, resources_left)
    
    # 3. Calcular confianza como función de distancia
    distance = |closervalue - action|
    confidence = distance * (-2) + 1  # Escala 1 a 0 en rango [0, 0.5]
    
    # 4. Generar tiempos de respuesta realistas
    rt_hold = normal(distance*10, 3)
    rt_release = rt_hold + normal(distance, 1)
    
    return response, confidence, rt_hold, rt_release
```

#### Paso 3: Aplicar Ruido de Decaimiento

Referencia: `Agent.py:adjust_response_decay()`

```python
def adjust_response_decay(resp, decay, resources_left):
    """
    Añade ruido gaussiano para emular comportamiento menos óptimo con decaimiento.
    
    decay: parámetro de decaimiento (0-1)
           0 = sin ruido (óptimo)
           1 = máximo ruido (aleatorio)
    """
    variance = int((1.0 - decay) * AGENT_NOISE_VARIANCE)
    delta = gaussian(resp, variance)
    return clip(delta, 0, resources_left)
```

**Decaimiento de Boltzmann**:
```python
def boltzmann_decay(global_seq_no):
    """
    Reduce decaimiento a lo largo del experimento (aprende).
    T = exp(-75 / (4 * seq_number))
    """
    return exp(-75.0 / (4.0 * global_seq_no))
```

#### Paso 4: Guardar Respuesta

```python
# Línea 590-650 en __main__.py
Responses_filehandle.write(f"{InitialSeverity},{Response},{Confidence},{rt_hold},{rt_release}\n")
ResourceAllocationsAtCurrentlyVisibleCities.append(Response)
resources_left -= Response
```

### 3.4 Cálculo de Severidad Actualizada

Referencia: `PES/src/exp_utils.py:get_updated_severity()`

```python
def get_updated_severity(no_of_cities, resource_allocated, initial_severity):
    """
    Actualiza severidad para cada ciudad basado en recursos asignados.
    """
    for c in range(no_of_cities):
        NewSeverity_c = SEVERITY_MULTIPLIER * initial_severity[c] - RESPONSE_MULTIPLIER * resource_allocated[c]
        NewSeverity_c = max(NewSeverity_c, 0)  # No puede ser negativa
        UpdatedSeverity_list.append(NewSeverity_c)
    
    return UpdatedSeverity_list
```

**Ejemplo Numérico**:
```
Initial Severity = 6
Resources Allocated = 8
SEVERITY_MULTIPLIER = 1.4
RESPONSE_MULTIPLIER = 0.4

New Severity = 1.4 * 6 - 0.4 * 8
             = 8.4 - 3.2
             = 5.2 ✓
```

### 3.5 Cálculo de Rendimiento por Secuencia

Referencia: `exp_utils.py:calculate_normalised_final_severity_performance_metric()`

```python
def calculate_normalised_final_severity_performance_metric(final_severity, initial_severity):
    """
    Normaliza el rendimiento como: 1 - (final_severity / initial_severity)
    
    Rango: [0, 1]
    - 1.0 = severidad reducida a 0 (óptimo)
    - 0.0 = severidad sin cambios (worst)
    """
    
    performance = 1.0 - (sum(final_severity) / sum(initial_severity))
    performance = max(0, min(performance, 1.0))  # Clip [0, 1]
    
    return performance, final_severity, confidence_weighted_allocation
```

**Ejemplo**:
```
Initial Severity Total = 10
Final Severity Total = 3 (agente asignó recursos bien)
Performance = 1 - (3/10) = 0.7 (70% de mejora)
```

---

## 4. Parámetros de Configuración Clave

### Archivo: `PES/config/CONFIG.py`

| Parámetro | Valor | Significado |
|-----------|-------|------------|
| `NUM_BLOCKS` | 8 | Bloques en el experimento |
| `NUM_SEQUENCES` | 8 | Secuencias por bloque (64 total) |
| `NUM_MIN_TRIALS` | 3 | Mínimo trials por secuencia |
| `NUM_MAX_TRIALS` | 10 | Máximo trials por secuencia |
| `AVAILABLE_RESOURCES_PER_SEQUENCE` | 39 | Recursos libres por secuencia (49 total - 9 iniciales) |
| `MAX_ALLOCATABLE_RESOURCES` | 10 | Máximo recursos por asignación individual |
| `INIT_NO_OF_CITIES` | 2 | Ciudades pre-asignadas al inicio |
| `PANDEMIC_PARAMETER` | 0.4 | Factor de pandemia (amplificación/respuesta) |
| `RANDOM_INITIAL_SEVERITY` | False | Si generar severidades random o cargar de CSV |

### Variables Derivadas (calculadas en __init__.py)

```python
SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER = 1.4
RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER = 0.4
```

---

## 5. Flujos de Datos

### 5.1 Entrada → Salida

```
PES/inputs/
├─ initial_severity.csv        (360 severidades iniciales)
├─ sequence_lengths.csv        (duración de cada secuencia)
├─ optimal_resources.npy       (solución óptima para referencia)
└─ optimal_severity.npy        (severidad óptima para referencia)
                    ↓
         [EXPERIMENTO PRINCIPAL]
                    ↓
PES/outputs/
├─ PES_responses_{SubjectId}.csv      (respuestas por trial)
├─ PES_log_{SubjectId}.log            (log de ejecución)
├─ PES_performance_plot_{SubjectId}.png    (gráficos de rendimiento)
└─ PES_performance_stats_{SubjectId}.png   (estadísticas en barras)
```

### 5.2 Archivo de Respuestas CSV

```csv
#InitialSeverity, Response, Confidence, PressEvent_seconds, ReleaseEvent_seconds
6, 8, 0.75, 0.450, 1.230
4, 5, 0.82, 0.380, 0.950
7, 10, 0.60, 0.620, 1.450
...
```

---

## 6. Análisis de Resultados

### 6.1 Renderización de Resultados

Después de completar todas las secuencias, el sistema automáticamente genera:

#### Print en Consola: `print_experiment_results_summary()`

```
================================================================================
EXPERIMENT RESULTS SUMMARY
================================================================================

Agent Performance Statistics:
  - Mean Performance:        0.8234
  - Std Deviation:           0.0847
  - Minimum Performance:     0.5362
  - Maximum Performance:     1.0000
  - Median Performance:      0.8420
  - Number of Sequences:     64

Aggregated Performance Statistics:
  - Mean Performance:        0.8234  (igual en modo single-agent)
  - Std Deviation:           0.0847
  - Minimum Performance:     0.5362
  - Maximum Performance:     1.0000
================================================================================
```

**Referencia en Código**: `exp_utils.py:640-654`

#### Gráficos PNG: `plot_experiment_results()`

**Archivo 1: `PES_performance_plot_{SubjectId}.png`** (2×2 subplots)

| Subplot | Descripción |
|---------|-------------|
| (1,1) | Rendimiento a lo largo de secuencias (línea con área sombreada) |
| (1,2) | Distribución de rendimientos (histograma con media/mediana) |
| (2,1) | Curva de aprendizaje (media acumulativa) |
| (2,2) | Comparación Agent vs Agregado (barras lado a lado) |

**Archivo 2: `PES_performance_stats_{SubjectId}.png`** (gráfico de barras)

Muestra estadísticas resumidas (media, std, min, max, mediana) en formato visual.

**Referencia en Código**: `exp_utils.py:505-585`

---

## 7. Modos de Ejecución

### 7.1 Experimento Principal

```bash
python3 -m PES
```

**Qué sucede**:
1. Inicializa configuración desde CONFIG.py
2. Carga severidades iniciales
3. Ejecuta 64 secuencias (8 bloques × 8 secuencias)
4. Por cada secuencia: 3-10 trials de toma de decisión
5. Genera gráficos y estadísticas
6. Guarda resultados en outputs/

**Tiempo Estimado**: 2-5 minutos

### 7.2 Entrenamiento de Agente RL

```bash
python3 -m PES.ext.train_rl
```

**Qué sucede**:
1. Carga datos de entrada (sequence_lengths.csv, initial_severity.csv)
2. Ejecuta Q-Learning para entrenar el agente
3. Guarda Q-table en inputs/q.npy
4. Genera gráficos de rewards vs episodios

**Referencia**: `ext/train_rl.py`

---

## 8. Integración del Agente RL

### 8.1 Arquitectura del Agente

```
Entrenamiento (train_rl.py)
    ↓
    Q-Learning: aprende qué acción tomar en cada estado
    ↓
    Salida: Q-table(state, action) → reward esperado
    ↓
    Archivo: inputs/q.npy (tabla de decisión)
    
Ejecución (__main__.py)
    ↓
    Carga Q-table
    ↓
    Para cada trial:
        - Estado = (recursos_disponibles, trial_no, severity)
        - Acción = argmax(Q[state, :])  (decisión óptima)
        - Aplica ruido y confianza metacognitiva
    ↓
    Resultado: asignación de recursos por trial
```

### 8.2 Flujo de Decisión

```python
# En pygameMediator.provide_rl_agent_response() (Línea ~1800)

def provide_rl_agent_response(img, ..., sequence_no, trial_no):
    # 1. Cargar Q-table entrenado
    Q = numpy.load(os.path.join(INPUTS_PATH, 'q.npy'))
    
    # 2. Construir estado actual
    state = (resources_available, trial_number, current_severity)
    
    # 3. Obtener acción óptima de Q-table
    action = argmax(Q[state, :])  # Mejor acción según entrenamiento
    
    # 4. Aplicar transformaciones metacognitivas
    response, confidence, rt_hold, rt_release = agent_meta_cognitive(action, ...)
    response = adjust_response_decay(response, boltzmann_decay(global_seq_no), ...)
    
    # 5. Retornar respuesta
    return response, rt_hold, rt_release, confidence
```

---

## 9. Validación y Debugging

### 9.1 Verbosidad

Para ver logs detallados durante ejecución:

```python
# En CONFIG.py
VERBOSE = True
DEBUG = True
```

Esto imprimirá a consola y al archivo de log información sobre:
- Índices de bloque/secuencia procesados
- Severidades calculadas
- Recursos asignados
- Rendimientos por secuencia

### 9.2 Verificación de Archivos Generados

Después de `python3 -m PES`, verificar:

```bash
ls -lah PES/outputs/
# Debe contener:
# - PES_responses_{SubjectId}.csv (1-2 MB)
# - PES_log_{SubjectId}.log (100-500 KB)
# - PES_performance_plot_{SubjectId}.png (200-300 KB)
# - PES_performance_stats_{SubjectId}.png (150-250 KB)
```

### 9.3 Sintaxis y Errores

Compilar módulos para verificar sintaxis:

```bash
python3 -m py_compile PES/__main__.py
python3 -m py_compile PES/src/exp_utils.py
python3 -m py_compile PES/src/Agent.py
```

---

## 10. Ejemplos Prácticos

### Ejemplo 1: Entender un Trial Individual

**Secuencia 5, Trial 3**:

```
1. Severidad inicial ciudad 1: 6
2. Severidad inicial ciudad 2: 4
3. Severidad inicial ciudad 3: 7

Recursos disponibles al inicio: 49

Trial 1:
  - Agente asigna: 8 recursos
  - Nueva severidad: 1.4*6 - 0.4*8 = 5.2
  - Recursos restantes: 49 - 8 = 41

Trial 2:
  - Agente asigna: 5 recursos
  - Nueva severidad: 1.4*4 - 0.4*5 = 3.6
  - Recursos restantes: 41 - 5 = 36

Trial 3:
  - Agente asigna: 10 recursos (máximo)
  - Nueva severidad: 1.4*7 - 0.4*10 = 5.8
  - Recursos restantes: 36 - 10 = 26

Severidad Final: 5.2 + 3.6 + 5.8 = 14.6
Severidad Inicial: 6 + 4 + 7 = 17.0
Performance: 1 - (14.6/17.0) = 0.141 (14.1% mejora)
```

### Ejemplo 2: Trace de Ejecución

```
[14:30:15] --- Entering main execution block ---
[14:30:20] Initializing experiment...
[14:30:25] Loading initial severities from CSV
[14:30:30] --- Starting experiment ---
[14:30:35] Processing Block 0, Sequence 0 (map=3)
[14:30:36]   Trial 1: severity=6, response=8, confidence=0.75
[14:30:37]   Trial 2: severity=4, response=5, confidence=0.82
[14:30:38]   Trial 3: severity=7, response=10, confidence=0.60
[14:30:39] Sequence 0: Performance = 0.1410
[14:30:40] Processing Block 0, Sequence 1 (map=7)
...
[14:35:15] --- Experiment completed ---
[14:35:16] 
================================================================================
EXPERIMENT RESULTS SUMMARY
...
```

---

## 11. Preguntas Frecuentes

### P: ¿Qué significa "Aggregated Score" en modo single-agent?
**R**: En modo single-agent, es el mismo que el score del agente, porque solo hay un agente asignando recursos. La arquitectura permite múltiples agentes, pero actualmente solo se usa uno.

### P: ¿Por qué varían los trials por secuencia?
**R**: Simula ambientes reales donde pandemias afectan diferentes números de ciudades. El rango 3-10 es configurable en CONFIG.py.

### P: ¿Qué es PANDEMIC_PARAMETER?
**R**: Factor de amplificación de pandemia. Valor 0.4 significa:
- La severidad crece 40% sin intervención
- Cada recurso reduce severidad 40%

### P: ¿El agente aprende durante la ejecución?
**R**: **No**, usa un Q-table pre-entrenado. El aprendizaje ocurre en `train_rl.py`.

### P: ¿Puedo cambiar parámetros sin reentrenar?
**R**: Sí, los parámetros en CONFIG.py afectan inmediatamente la simulación. Pero si cambias parámetros radicales, el Q-table entrenado puede no ser óptimo.

---

## 12. Referencia de Código

### Archivos Principales

| Archivo | Líneas | Propósito |
|---------|--------|----------|
| `__main__.py` | 830 | Loop principal de experimento |
| `src/Agent.py` | 92 | Funciones de decisión del agente |
| `src/exp_utils.py` | 654 | Utilidades de experimento (cálculos) |
| `ext/train_rl.py` | 201 | Entrenamiento con Q-Learning |
| `ext/pandemic.py` | 330 | Modelo Gym del entorno |
| `config/CONFIG.py` | 100 | Configuración de parámetros |

### Funciones Críticas

```python
# Agent.py
agent_meta_cognitive(action, output_value_range, resources_left, response_timeout)
  └─ Genera respuesta e confianza del agente

# exp_utils.py
get_updated_severity(no_of_cities, resource_allocated, initial_severity)
  └─ Calcula nuevas severidades

calculate_normalised_final_severity_performance_metric(final_severity, initial_severity)
  └─ Calcula rendimiento (0-1)

# __main__.py
main()
  └─ Ejecuta experimento completo
```

---

## Conclusión

El PES es una simulación bien estructurada que permite estudiar cómo un agente RL toma decisiones de asignación de recursos bajo incertidumbre y restricciones. La arquitectura modular facilita experimentación, análisis y extensión.

Para profundizar en los aspectos teóricos del Reinforcement Learning usado en el agente, ver `RL_THEORY.md` y `RL_AGENT.md`.
