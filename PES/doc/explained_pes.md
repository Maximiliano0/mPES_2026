# PES: Explicación Detallada del Funcionamiento del Proyecto

## 1. Introducción

**mPES (Pandemic Experiment Scenario)** es un framework de investigación que simula escenarios de respuesta a pandemias, donde un agente de **Reinforcement Learning (Q-Learning)** aprende a optimizar la asignación limitada de recursos para minimizar la severidad de enfermedades en múltiples ciudades.

### Objetivo Principal
Entrenar y ejecutar un agente inteligente que tome decisiones estratégicas sobre distribución de recursos médicos/sanitarios bajo restricciones de disponibilidad.

---

## 2. Estructura Jerárquica del Experimento

El experimento sigue una estructura anidada estricta definida en `__main__.py`:

```
1 EXPERIMENTO
├─ NUM_BLOCKS = 8 BLOQUES
│  ├─ NUM_SEQUENCES = 8 SECUENCIAS por bloque
│  │  ├─ NUM_MIN_TRIALS a NUM_MAX_TRIALS (3-10) TRIALS por secuencia
│  │  │  └─ 1 ACCIÓN DE ASIGNACIÓN DE RECURSOS (0-10)
```

### Desglose de Números
- **Total de bloques**: 8
- **Total de secuencias**: 64 (8 bloques × 8 secuencias)
- **Trials por bloque**: ~45 (TOTAL_NUM_TRIALS_IN_BLOCK = 45)
- **Trials totales**: ~360 (64 secuencias × 5.6 trials promedio)
- **Total de decisiones**: ~360

### Configuración en `CONFIG.py`
```python
NUM_BLOCKS = 8                      # Número de bloques experimentales
NUM_SEQUENCES = 8                   # Secuencias por bloque
NUM_MIN_TRIALS = 3                  # Trials mínimos por secuencia
NUM_MAX_TRIALS = 10                 # Trials máximos por secuencia
TOTAL_NUM_TRIALS_IN_BLOCK = 45      # Suma exacta de trials en un bloque
```

---

## 3. Modelo Dinámico del Escenario Pandémico

### 3.1 La Fórmula de Progresión de Severidad

Implementada en `exp_utils.py` (`get_updated_severity()`):

```python
new_severity = max(0, SEVERITY_MULTIPLIER × initial_severity - RESPONSE_MULTIPLIER × allocated_resources)
```

Donde:
- **SEVERITY_MULTIPLIER** (β) = 1 + PANDEMIC_PARAMETER = 1.4
  - Representa el crecimiento natural de la pandemia sin intervención
  - Tasa de crecimiento: 40% por unidad de tiempo
  
- **RESPONSE_MULTIPLIER** (α) = PANDEMIC_PARAMETER = 0.4
  - Representa la efectividad de los recursos en combate
  - Cada unidad de recurso reduce 0.4 puntos de severidad

### 3.2 Ejemplo Numérico

```
Entrada inicial: severidad = 4, recursos_asignados = 5, α = 0.4

Step 1: new_sev = 1.4 × 4 - 0.4 × 5 = 5.6 - 2.0 = 3.6
Step 2: new_sev = 1.4 × 3.6 - 0.4 × 5 = 5.04 - 2.0 = 3.04
Step 3: new_sev = 1.4 × 3.04 - 0.4 × 5 = 4.256 - 2.0 = 2.256
```

La severidad evoluciona a través de los trials, acumulando efecto de recursos.

### 3.3 Evolución Temporal en una Secuencia

En `get_array_of_sequence_severities_from_allocations()`:
- Se itera sobre cada trial de la secuencia
- Para cada trial se aplica la fórmula
- El vector de severidades se actualiza secuencialmente
- Resultado: array de severidades finales [sev_1, sev_2, ..., sev_n]

---

## 4. Flujo Principal del Experimento

### 4.1 Inicialización (`__main__.py`)

**Código relevante:**
```python
def main():
    # Validación de archivos RL-AGENT
    if PLAYER_TYPE == 'RL_AGENT':
        q_file = os.path.join(INPUTS_PATH, 'q.npy')
        rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')
        
        # Verificar que los archivos entrenados existan
        if not os.path.isfile(q_file):
            terminal_utils.error("Q-Table file not found!")
            return
```

**Pasos**:
1. Valida que `q.npy` (Q-table entrenada) exista
2. Valida que `rewards.npy` (histórico de recompensas) exista
3. Carga ambos archivos en memoria
4. Si alguno falta, sugiere ejecutar `python3 -m PES.ext.train_rl`

### 4.2 Creación de Sesión (`__main__.py`)

**Identificación del experimento:**
```python
MySubjectId = f"{datetime.date.today().strftime('%Y-%m-%d')}_{PLAYER_TYPE}"
# Resultado: "2026-02-09_RL_AGENT"

session_outputs_path = os.path.join(OUTPUTS_PATH, MySubjectId)
os.makedirs(session_outputs_path, exist_ok=True)
```

**Logging inicializado:**
```python
log_utils.create_ConsoleLog_filehandle_singleton(MySubjectId)
# Crea: outputs/PES_log_2026-02-09_RL_AGENT.txt
```

### 4.3 Asignación de Mapas y Secuencias (`__main__.py`)

**Estructura de datos:**
```python
# Matriz blocks × sequences que almacena:
NumTrials__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))
MapIndices__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))
```

**Lógica de asignación de mapa:**
```python
for blk in range(NUM_BLOCKS):
    for seq in range(NUM_SEQUENCES):
        numpy.random.seed(100 + blk)  # Seed reproducible
        
        # Asignar índice de mapa único (0-8) sin repeticiones en el bloque
        while counter_seq > 0:
            b = numpy.random.randint(0, 9)
            if b not in MapIndices__blocks_x_sequences__2darray[blk, :]:
                MapIndices__blocks_x_sequences__2darray[blk, seq] = b
            counter_seq -= 1
```

**Asignación de número de trials:**
```python
if USE_FIXED_BLOCK_SEQUENCES:
    # Cargar desde CSV: sequence_lengths.csv
    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.next_seq_length(
        blk * NUM_SEQUENCES,
        NUM_SEQUENCES
    )
else:
    # Generar aleatoriamente respetando constraint de 45 trials/bloque
    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.sampler(
        NUM_SEQUENCES,
        TOTAL_NUM_TRIALS_IN_BLOCK,
        [NUM_MIN_TRIALS, NUM_MAX_TRIALS],
        ...
    )
```

### 4.4 Ejecución de Bloques y Secuencias (`__main__.py`)

**Loop principal:**
```python
for blk in range(STARTING_BLOCK_INDEX, NUM_BLOCKS):
    for seq in range(STARTING_SEQ_INDEX, NUM_SEQUENCES):
        
        # Obtener configuración de secuencia
        num_trials = NumTrials__blocks_x_sequences__2darray[blk, seq]
        initial_severities = get_initial_severities_for_sequence(...)
        
        # Ejecutar secuencia
        run_sequence(..., num_trials, initial_severities)
```

### 4.5 Ejecución de Trials (Ciudades)

**Para cada trial en la secuencia:**

```python
for trial in range(num_trials):
    # Estado actual
    current_severity = initial_severities[trial]
    resources_remaining = AVAILABLE_RESOURCES_PER_SEQUENCE - sum(previous_allocations)
    
    # Consultar Q-table
    action = Q[resources_remaining, trial, int(current_severity)]
    action = numpy.argmax(action)  # Greedy selection
    
    # Validar acción
    if action > resources_remaining:
        action = resources_remaining
    
    # Registrar decisión
    response[blk, seq, trial] = action
    confidence[blk, seq, trial] = calculate_confidence(...)
    
    # Actualizar severidad
    current_severity = get_updated_severity(
        len(severities_so_far),
        allocations_so_far + [action],
        initial_severities
    )
```

---

## 5. Cálculo de Severidades y Performance

### 5.1 Severidad Final por Secuencia

Implementado en `get_sequence_severity_from_allocations()`:
```python
FinalSequenceSeverity = numpy.sum(final_severities_array)
```

Suma todas las severidades finales de todos los trials en la secuencia.

### 5.2 Métrica de Performance Normalizado

Implementada en `calculate_normalised_final_severity_performance_metric()`:

```python
# Best case: máximo recursosasignados (10 cada trial)
BestCaseSequenceSeverity = get_sequence_severity_from_allocations(
    [10, 10, ..., 10],  # Max allocation
    initial_severities
)

# Worst case: mínimos recursos asignados (0 cada trial)
WorstCaseSequenceSeverity = get_sequence_severity_from_allocations(
    [0, 0, ..., 0],     # Min allocation
    initial_severities
)

# Performance normalizado en [0, 1]
Performance = (WorstCaseSequenceSeverity - FinalSequenceSeverity) / 
              (WorstCaseSequenceSeverity - BestCaseSequenceSeverity)
```

**Interpretación:**
- `Performance = 0`: Agente actuó como worst case (sin recursos)
- `Performance = 0.5`: Agente actuó en línea media
- `Performance = 1`: Agente actuó de forma óptima (máximos recursos)

### 5.3 Ejemplo Numérico

```
Initial severities: [3, 4, 8]
Allocations made: [5, 6, 4]
Final severities: [1.5, 3.1, 8.8]
FinalSequenceSeverity = 13.4

Worst case (zero allocation): [4.2, 5.6, 11.2] → 21.0
Best case (max allocation):   [0.8, 1.2, 3.4] → 5.4

Performance = (21.0 - 13.4) / (21.0 - 5.4) = 7.6 / 15.6 = 0.487
```

---

## 6. Módulo de Confianza Meta-cognitiva

Implementado en `pygameMediator.py` y `pandemic.py`:

### 6.1 `rl_agent_meta_cognitive()`

Esta función existe en dos ubicaciones:
- `pygameMediator.py`: usada durante la ejecución del experimento (`python3 -m PES`)
- `pandemic.py`: usada durante el entrenamiento (`python3 -m PES.ext.train_rl`)

```python
def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    """
    Genera tiempos de reacción humano-realistas basados en confianza
    """
    # options = Q-values para las acciones disponibles
    
    # Calcular entropía de opciones
    decision_entropy = entropy_from_pdf(options)
    
    # Normalizar confianza
    m_entropy = entropy_from_pdf([1, 0, 0, ...])  # Mín
    M_entropy = entropy_from_pdf([1, 1, 1, ...])  # Máx
    confidence = (decision_entropy - M_entropy) / (m_entropy - M_entropy)
    
    # Seleccionar acción
    response = numpy.argmax(options)
    
    # Mapear confianza a tiempos de reacción
    # Alta confianza → tiempos rápidos
    # Baja confianza → tiempos lentos
    map_to_response_time = lambda x: x * (-2) + 1
    mu = int(map_to_response_time(confidence) * 10)
    rt_hold = numpy.random.normal(mu, 3, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]
    
    # Clipear a rango válido [0, response_timeout]
    rt_hold = numpy.clip(rt_hold, 0, response_timeout/1000.0)
    rt_release = numpy.clip(rt_release, 0, response_timeout/1000.0)
    
    return response, confidence, rt_hold, rt_release
```

> **Nota**: Las funciones `entropy()` y `calculate_agent_response_and_confidence()` que
> existían previamente en `pygameMediator.py` fueron eliminadas por ser código muerto
> (no eran invocadas en ningún lugar del proyecto).

**Propósito**: Simular un agente que no solo toma decisiones óptimas, sino que refleja incertidumbre (baja confianza) con tiempos de reacción más largos.

---

## 7. Agregación de Decisiones (Multi-participante)

Implementado en `exp_utils.py`:

### 7.1 Métodos Disponibles

En `CONFIG.py`:
```python
AGGREGATION_METHOD = {
    1: 'confidence_weighted_median',    # Robusto a outliers
    2: 'confidence_weighted_mean',      # Promedio ponderado
    3: 'confidence_weighted_mode'       # Valor más frecuente
}[2]  # Seleccionar método 2
```

### 7.2 `get_confidence_weighted_mean()`

> **Nota sobre agente único**: Cuando el experimento se ejecuta con un solo agente RL
> (configuración actual), estas funciones de agregación no tienen efecto práctico porque
> `AllMessages` contiene un único participante. Se conservan por compatibilidad futura
> con escenarios multi-jugador.

```python
def get_confidence_weighted_mean(all_messages, first_severity, AbsoluteSequenceIndex, AbsoluteTrialCount):
    """
    Promedio ponderado por confianza
    
    decisions = [5, 7, 6]  (asignaciones de 3 participantes)
    confidences = [0.8, 0.6, 0.9]  (sus confianzas)
    
    weighted_mean = (5×0.8 + 7×0.6 + 6×0.9) / (0.8 + 0.6 + 0.9)
                 = (4.0 + 4.2 + 5.4) / 2.3
                 = 13.6 / 2.3
                 = 5.91
    """
    numerator = numpy.sum(decisions * confidences)
    denominator = numpy.sum(confidences)
    return numerator / denominator
```

### 7.3 `get_confidence_weighted_median()`

```python
def get_confidence_weighted_median(all_messages, first_severity, AbsoluteSequenceIndex, AbsoluteTrialCount):
    """
    Mediana robusta que ordena por confianza
    - Descarta opiniones con baja confianza
    - Robusto a valores extremos (outliers)
    """
    # Ordenar por confianza descendente
    sorted_indices = numpy.argsort(confidences)[::-1]
    sorted_decisions = decisions[sorted_indices]
    sorted_confidences = confidences[sorted_indices]
    
    # Calcular mediana ponderada
    cumsum = numpy.cumsum(sorted_confidences)
    median_idx = numpy.where(cumsum >= cumsum[-1]/2)[0][0]
    
    return sorted_decisions[median_idx]
```

---

## 8. Generación de Reportes

### 8.1 Archivos de Output

Implementado en `result_formatter.py`:

```python
def generate_results_report(subject_id, outputs_path, performances, all_performances, resources_data=None):
    """
    Genera dos tipos de reportes:
    1. JSON: Estadísticas numéricas
    2. PNG: Visualizaciones multi-panel
    """
    
    # Estadísticas
    stats = {
        'overall_mean': numpy.mean(performances),
        'overall_median': numpy.median(performances),
        'overall_std': numpy.std(performances),
        'first_block_mean': numpy.mean(performances[:8]),
        'last_block_mean': numpy.mean(performances[-8:]),
        'improvement': last_block_mean - first_block_mean,
        ...
    }
    
    # Generar JSON
    json_filepath = os.path.join(outputs_path, f'PES_results_{subject_id}.json')
    with open(json_filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generar PNG con múltiples subplots
    # - Tendencia de performance
    # - Distribución de performance
    # - Box plot por bloque
    # - Severidad acumulativa
    # etc.
    png_filepath = os.path.join(outputs_path, f'PES_results_{subject_id}.png')
    plt.savefig(png_filepath)
    
    return json_filepath, png_filepath
```

### 8.2 Archivos Generados

**Por experimento** (ej. "2026-02-09_RL_AGENT/"):
```
PES__2026-02-09_RL_AGENT.txt           # Información de sujeto
PES_responses_2026-02-09_RL_AGENT.txt  # Todas las decisiones (acción, confianza, tiempos)
PES_results_2026-02-09_RL_AGENT.json   # Estadísticas JSON
PES_results_2026-02-09_RL_AGENT.png    # Gráficas (6+ subplots)
PES_movement_log_2026-02-09_RL_AGENT.npy  # Matriz evolución severidades
```

**Global:**
```
PES_log_2026-02-09_RL_AGENT.txt        # Log completo (terminal + archivos)
```

---

## 9. Entrada de Datos

### 9.1 Datos Requeridos en `inputs/`

```
initial_severity.csv
─────────────────────
Formato: CSV de severidades iniciales
Estructura: 1 valor por línea o 1 fila × N columnas
Total: NUM_BLOCKS × NUM_SEQUENCES × promedio_trials
Ejemplo: 
  3.2
  4.1
  2.8
  ...
  (≈360 valores)

Cargado en: __main__.py (vía numpy.loadtxt)
```

```
sequence_lengths.csv
────────────────────
Formato: CSV con número de trials por secuencia
Estructura: 64 valores (8 bloques × 8 secuencias)
Rango: 3-10 (respetando 45 trials/bloque)
Ejemplo:
  5, 6, 4, 7, 5, 4, 7, 7  (bloque 1)
  4, 6, 5, 6, 4, 7, 5, 8  (bloque 2)
  ...

Cargado en: exp_utils.py (next_seq_length(), llamado desde __main__.py)
```

### 9.2 Archivos de Modelo Entrenado

```
q.npy
─────
Shape: (31, 11, 11, 11)
  - 31: recursos disponibles (0-30, con 9 pre-asignados = 39 total)
  - 11: número de trial (0-10)
  - 11: severidad actual (0-10)
  - 11: valores de acción (0-10 recursos posibles)

Type: float64
Contenido: Q-values para cada (estado, acción)

rewards.npy
───────────
Shape: (N/10000,)  [promedio cada 10k episodios; N = episodios de entrenamiento]
Contenido: Histórico de recompensas promedio durante entrenamiento
Uso: Para visualización de curva de aprendizaje
```

---

## 10. Tabla de Referencia: Código y Experimento

| Componente | Archivo | Función / Sección | Funcionalidad |
|-----------|---------|-------------------|---------------|
| **Inicialización** | `__main__.py` | `main()` inicio | Validación Q-table |
| **Creación sesión** | `__main__.py` | `main()` sesión | ID único, logging |
| **Asignación mapas** | `__main__.py` | `main()` asignación | Seeds reproducibles |
| **Asignación trials** | `__main__.py` | `main()` asignación | Constraint 45/bloque |
| **Loop bloques** | `__main__.py` | `main()` loop principal | Iteración 8 bloques |
| **Loop secuencias** | `__main__.py` | `main()` loop interno | Iteración 8 secuencias |
| **Loop trials** | `__main__.py` | `main()` loop trials | Iteración 3-10 trials |
| **Consulta Q-table** | `pygameMediator.py` | `provide_rl_agent_response()` | Obtener acción + confianza |
| **Update severidad** | `exp_utils.py` | `get_updated_severity()` | Aplicar fórmula |
| **Calc performance** | `exp_utils.py` | `calculate_normalised_...()` | Normalizar [0,1] |
| **Confianza** | `pygameMediator.py` | `rl_agent_meta_cognitive()` | Entropía meta-cognitiva |
| **Reportes** | `result_formatter.py` | `generate_results_report()` | JSON + PNG |
| **Logging** | `log_utils.py` | `create_ConsoleLog_...()` | Dual terminal+archivo |

---

## 11. Flujo Temporal Resumido

```
Inicio
  ↓
Cargar Q.npy y rewards.npy
  ↓
Inicializar logging ("PES_log_2026-02-09_RL_AGENT.txt")
  ↓
FOR bloque = 0 to 7:
  │
  ├─ Asignar 8 índices de mapa (0-8, sin repetición)
  ├─ Asignar 8 números de trials (sum = 45)
  │
  ├─ FOR secuencia = 0 to 7:
  │   │
  │   ├─ Cargar severidades iniciales
  │   ├─ Inicializar recursos = 39
  │   │
  │   ├─ FOR trial = 0 to num_trials:
  │   │   │
  │   │   ├─ Obtener estado: [recursos_left, trial_num, severidad]
  │   │   ├─ Consultar Q[estado] → obtener Q-values para acciones
  │   │   ├─ Seleccionar acción: argmax(Q-values)
  │   │   ├─ Validar: si acción > recursos_left, ajustar
  │   │   ├─ Registrar: response[trial] = acción
  │   │   ├─ Calcular: confidence = entropy(Q-values)
  │   │   ├─ Update: severidad = f(severidad_prev, acción)
  │   │   ├─ Update: recursos_left -= acción
  │   │   │
  │   │   └─ Escribir en responses.txt
  │   │
  │   ├─ FIN trials
  │   ├─ Calc finalidad_secuencia = sum(severidades)
  │   ├─ Calc perf_normalizado = (worst - final) / (worst - best)
  │   ├─ Guardar en performances[]
  │   └─ Escribir en results.json
  │
  └─ FIN secuencias
│
FIN bloques
  ↓
Calc estadísticas globales
  ↓
Generar result_formatter PNG multi-subplots
  ↓
Generar movement_log.npy (evolución severidades)
  ↓
Finalizar logging
  ↓
Fin
```

---

## 12. Reproducibilidad

### 12.1 Seeds Reproducibles

```python
# En asignación de mapas (__main__.py)
numpy.random.seed(100 + blk)

# Resultado: mismo bloque siempre genera mismos mapas
```

### 12.2 Datos Fijos

```python
USE_FIXED_BLOCK_SEQUENCES = True  # Cargar desde CSV, no generar aleatorio
RANDOM_INITIAL_SEVERITY = False   # Cargar desde CSV, no generar aleatorio
```

### 12.3 Configuración Registrada

Cada experimento guarda su archivo de configuración:
```
SubjectInfo_filename = f'PES__{MySubjectId}.txt'
# Contiene todos los CONFIG.* usados en la ejecución
```

---

## Conclusión

El experimento PES implementa un ciclo estructurado y repetible:
1. **Configuración** → Define parámetros fijos
2. **Asignación** → Mapea mapa e trials a bloques/secuencias
3. **Ejecución** → Ejecuta Q-Learning consultando Q-table entrenada
4. **Cálculo** → Actualiza severidades y calcula performance
5. **Logging** → Registra todas las decisiones y métricas
6. **Reportes** → Genera estadísticas y visualizaciones

Este diseño permite reproducir el mismo experimento múltiples veces y comparar variaciones de agentes o parámetros.
