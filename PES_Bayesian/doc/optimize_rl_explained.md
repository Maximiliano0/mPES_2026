# Optimización Bayesiana de Hiperparámetros Q-Learning

## Descripción general

`optimize_rl.py` automatiza la búsqueda de los mejores hiperparámetros para el algoritmo
Q-Learning del escenario pandémico.  En lugar de probar combinaciones a mano o recorrer
una grilla exhaustiva, usa **optimización Bayesiana** (librería Optuna) para elegir de
forma inteligente qué combinación de hiperparámetros evaluar en cada paso.

---

## 1. Fundamento teórico

### 1.1 El problema

Q-Learning depende de cinco hiperparámetros cuyo valor óptimo se desconoce a priori:

| Símbolo | Parámetro | Rango explorado | Escala |
|---------|-----------|-----------------|--------|
| $\alpha$ | `learning_rate` | $[0.10,\; 0.4]$ | logarítmica |
| $\gamma$ | `discount_factor` | $[0.80,\; 0.99]$ | lineal |
| $\varepsilon_0$ | `epsilon_initial` | $[0.4,\; 1.0]$ | lineal |
| $\varepsilon_{\min}$ | `epsilon_min` | $[0.05,\; 0.1]$ | lineal |
| $N$ | `num_episodes` | $[800\,000,\; 1\,000\,000]$ | paso = 100 000 |

Probar todas las combinaciones (grid search) requiere un número exponencial de
evaluaciones.  Cada evaluación implica **entrenar una Q-table completa** (con
semilla fija `SEED` de `CONFIG.py` para reproducibilidad) y después evaluar sobre
64 secuencias fijas, lo que puede tardar varios minutos por combinación
dependiendo del `num_episodes` muestreado.

### 1.2 Optimización Bayesiana

La optimización Bayesiana resuelve el problema de "caja negra":

$$\boldsymbol{\theta}^{*} = \arg\max_{\boldsymbol{\theta}\in\Theta}\; f(\boldsymbol{\theta})$$

donde $\boldsymbol{\theta} = (\alpha,\gamma,\varepsilon_0,\varepsilon_{\min},N)$ y
$f(\boldsymbol{\theta})$ es el rendimiento medio normalizado sobre 64 secuencias.

El algoritmo sigue un ciclo de tres pasos en cada *trial* $t$:

1. **Modelo sustituto (surrogate):** Construir un modelo probabilístico
   $p(f \mid \boldsymbol{\theta})$ a partir de los resultados de los trials
   $1,\dots,t-1$ anteriores.
2. **Función de adquisición:** Elegir el $\boldsymbol{\theta}_t$ que maximice la
   probabilidad de mejorar el mejor valor conocido (*expected improvement*).
3. **Evaluación:** Ejecutar $f(\boldsymbol{\theta}_t)$ (entrenar + evaluar) y agregar
   el resultado al historial.

Optuna usa el **muestreador TPE (Tree-structured Parzen Estimator)** como modelo
sustituto.  En vez de modelar $p(f \mid \boldsymbol{\theta})$ directamente, modela
dos distribuciones:

$$\ell(\boldsymbol{\theta}) = p(\boldsymbol{\theta} \mid f > f^{*}), \qquad
  g(\boldsymbol{\theta}) = p(\boldsymbol{\theta} \mid f \leq f^{*})$$

y elige el $\boldsymbol{\theta}$ que maximice la razón $\ell(\boldsymbol{\theta}) / g(\boldsymbol{\theta})$.

Esto permite concentrar la exploración en zonas del espacio que estadísticamente
producen mejor rendimiento, necesitando muchos menos trials que grid search.

### 1.3 Función objetivo

El valor que se maximiza es el **rendimiento medio normalizado** de la Q-table
entrenada, evaluado sobre las 64 secuencias fijas que constituyen el *benchmark*
del escenario pandémico:

$$f(\boldsymbol{\theta}) = \frac{1}{64}\sum_{i=1}^{64}\;
  \text{perf}\!\left(S^{(i)}_{\text{final}},\; S^{(i)}_{\text{inicial}}\right)$$

donde $\text{perf}$ es la métrica de severidad final normalizada
(`calculate_normalised_final_severity_performance_metric`), definida en
`exp_utils.py` e invocada desde `run_experiment` en `pandemic.py`.  Un valor
cercano a $1.0$ indica que el agente redujo la severidad al mínimo posible con
los recursos disponibles.

---

## 2. Estructura del código

El archivo `optimize_rl.py` tiene cuatro secciones principales:

```
optimize_rl.py
├── _best_artifacts             # Almacena la mejor Q-table en memoria
├── _load_evaluation_data()     # Datos de evaluación (carga una vez)
├── objective(trial)            # Función objetivo que Optuna llama
├── _save_report(study, ...)    # Reportes y gráficos
└── main()                      # Orquestación: CLI, estudio, guardado Q-table
```

### 2.1 Carga de datos de evaluación

```python
def _load_evaluation_data():
    global _trials_per_sequence, _sevs, _number_cities_prob, _severity_prob

    _trials_per_sequence = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=','
    )
    all_severities = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=','
    )
    _sevs = convert_globalseq_to_seqs(_trials_per_sequence, all_severities)

    val_cities, count_cities = numpy.unique(
        _trials_per_sequence, return_counts=True)
    _number_cities_prob = numpy.asarray(
        (val_cities, count_cities / len(_trials_per_sequence))).T

    val_severity, count_severity = numpy.unique(
        all_severities, return_counts=True)
    _severity_prob = numpy.asarray(
        (val_severity, count_severity / len(all_severities))).T
```

**Propósito:** Cargar los CSV de longitudes y severidades una sola vez al inicio del
programa y precalcular las distribuciones de probabilidad.  Estos cuatro arrays globales
se reutilizan en cada trial sin re-leerlos del disco.

| Variable | Contenido |
|----------|-----------|
| `_trials_per_sequence` | Vector con la longitud (número de ciudades) de cada una de las 64 secuencias. |
| `_sevs` | Lista de arrays con las severidades iniciales de cada ciudad en cada secuencia. |
| `_number_cities_prob` | Distribución empírica de longitudes de secuencia (para generar secuencias aleatorias durante el entrenamiento). |
| `_severity_prob` | Distribución empírica de severidades iniciales (ídem). |

**Relación con la teoría:** Para que $f(\boldsymbol{\theta})$ sea comparable entre
trials, la evaluación debe hacerse siempre sobre el mismo benchmark; por eso se
cargan secuencias fijas.  Las distribuciones se usan durante el *entrenamiento* para
que el agente vea secuencias representativas del mismo dominio.

### 2.2 Función objetivo

```python
def objective(trial: optuna.Trial) -> float:
    # (1) Muestrear hiperparámetros
    learning_rate    = trial.suggest_float('learning_rate',    0.10,  0.4,  log=True)
    discount_factor  = trial.suggest_float('discount_factor',  0.80,  0.99)
    epsilon_initial  = trial.suggest_float('epsilon_initial',  0.40,   1.0)
    epsilon_min      = trial.suggest_float('epsilon_min',      0.05,   0.10)
    num_episodes     = trial.suggest_int('num_episodes',       800000, 1000000, step=100000)

    # (2) Entrenar Q-table (semilla fija para comparabilidad entre trials)
    env = Pandemic()
    env.number_cities_prob = _number_cities_prob
    env.severity_prob      = _severity_prob
    env.verbose = False
    rewards, Q, _ = QLearning(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        seed=SEED
    )

    # (3) Evaluar sobre 64 secuencias fijas
    env_eval = Pandemic()
    env_eval.verbose = False

    def qf(_env, state, _seqid):
        s0 = min(int(state[0]), Q.shape[0] - 1)
        s1 = min(int(state[1]), Q.shape[1] - 1)
        s2 = min(int(state[2]), Q.shape[2] - 1)
        # Mask infeasible actions (consistent with rl_agent_meta_cognitive)
        options = Q[s0, s1, s2].copy()
        o = numpy.arange(len(options), dtype=numpy.float32)
        options[o > state[0]] = 0.00001
        return numpy.argmax(options)

    _, perfs, _ = run_experiment(
        env_eval, qf, False, _trials_per_sequence, _sevs)
    mean_perf = float(numpy.mean(perfs))

    # (4) Guardar métricas auxiliares
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf',  float(numpy.std(perfs)))
    trial.set_user_attr('min_perf',  float(numpy.min(perfs)))
    trial.set_user_attr('max_perf',  float(numpy.max(perfs)))

    # (5) Preservar la mejor Q-table en memoria
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['Q'] = Q.copy()
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf

    return mean_perf
```

**Correspondencia teoría ↔ código:**

| Concepto teórico | Implementación |
|------------------|----------------|
| $\boldsymbol{\theta}$ muestreado por TPE | `trial.suggest_*()` — Optuna aplica el modelo sustituto para decidir qué valores probar. |
| Evaluación $f(\boldsymbol{\theta})$ | Entrenar Q-table + evaluar con `run_experiment()` sobre 64 secuencias fijas. |
| Resultado devuelto a Optuna | `return mean_perf` — Optuna lo usa para actualizar su modelo sustituto y elegir el próximo $\boldsymbol{\theta}$. |

#### Detalle de `trial.suggest_*`

- `suggest_float('learning_rate', 0.10, 0.4, log=True)`:
  Muestrea $\alpha$ en escala logarítmica dentro de un rango
  ($0.10$–$0.4$) determinado por corridas exploratorias previas.

- `suggest_int('num_episodes', 800000, 1000000, step=100000)`:
  Discretiza en múltiplos de 100 000 para reducir la dimensionalidad.
  El rango alto ($800\text{k}$–$1\text{M}$) asegura convergencia
  suficiente de la Q-table.

**Nota sobre la semilla:** Todos los trials de entrenamiento usan `seed=SEED`
(importado de `CONFIG.py`, valor por defecto 42).  Esto garantiza que las
diferencias de rendimiento entre trials se deban exclusivamente a los
hiperparámetros y no a variabilidad aleatoria del entrenamiento.

#### Función `qf` (política greedy con masking)

Dentro de `objective`, se define una política greedy con enmascaramiento de acciones
infactibles, consistente con `rl_agent_meta_cognitive` en `pandemic.py`
(y, por extensión, con la versión en `pygameMediator.py` usada al ejecutar el
experimento con `python3 -m PES_Bayesian`):

```python
def qf(_env, state, _seqid):
    s0 = min(int(state[0]), Q.shape[0] - 1)
    s1 = min(int(state[1]), Q.shape[1] - 1)
    s2 = min(int(state[2]), Q.shape[2] - 1)
    options = Q[s0, s1, s2].copy()
    o = numpy.arange(len(options), dtype=numpy.float32)
    options[o > state[0]] = 0.00001
    return numpy.argmax(options)
```

Esta función se pasa a `run_experiment()` como la *action function*.  Evalúa la
Q-table **sin epsilon**: siempre elige la acción con mayor Q-valor entre las
acciones factibles (las que no exceden los recursos disponibles `state[0]`).

**Masking de acciones infactibles:** Las acciones cuyo índice supera los recursos
disponibles se reducen a `0.00001`, garantizando que `argmax` seleccione solo
acciones factibles.  Esto replica el comportamiento de `rl_agent_meta_cognitive`
en `pygameMediator.py`, asegurando que la métrica obtenida durante la optimización
sea **consistente** con el rendimiento observado al ejecutar `python3 -m PES_Bayesian`.

Los `min(...)` aseguran que los índices de estado no excedan las dimensiones de la
Q-table `(31, 11, 10, 11)` = (recursos, trial, severidad, acciones).

#### Preservación de la mejor Q-table

Al final de cada trial, si el `mean_perf` supera el mejor valor previo, se guarda
una copia de la Q-table en `_best_artifacts`.  Esto evita el problema de re-entrenar
desde cero al final: Q-Learning es estocástico (inicialización aleatoria,
exploración $\varepsilon$-greedy, secuencias de entrenamiento aleatorias), por lo que
un re-entrenamiento con los mismos hiperparámetros puede producir una Q-table diferente
y potencialmente peor.

```python
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['Q'] = Q.copy()
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf
```

### 2.3 Persistencia con SQLite

```python
db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
storage = f'sqlite:///{db_path}'

study = optuna.create_study(
    direction='maximize',
    study_name=f'qlearning_opt_{opt_date}',
    sampler=optuna.samplers.TPESampler(seed=42),
    storage=storage,
    load_if_exists=True,
)
```

Cada trial completado se persiste en un archivo SQLite.  Si el proceso se interrumpe
(suspensión de la máquina, error, etc.), al re-ejecutar el mismo comando Optuna detecta
los trials ya completados y continúa desde donde se detuvo:

```python
completed = len([t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE])
remaining = max(0, n_trials - completed)
```

**Relación con la teoría:** El modelo sustituto del TPE se reconstruye a partir del
historial almacenado; no hay pérdida de información respecto a correr todo de corrido.

### 2.4 Protección contra underflow numérico

```python
_prev_err = numpy.seterr(under='ignore')
try:
    study.optimize(objective, n_trials=remaining,
                   callbacks=[_progress_callback])
finally:
    numpy.seterr(**_prev_err)
```

El módulo `PES_Bayesian` configura `numpy.seterr(all='raise', under='ignore')` en
su `__init__.py` para detectar errores numéricos durante la simulación.  Aunque el
*underflow* ya se ignora a nivel de paquete, el muestreador TPE calcula internamente
`numpy.exp(x)` donde $x$ puede ser muy negativo ($x < -700$), produciendo un
*underflow* a $0.0$ que es matemáticamente inofensivo (simplemente indica
probabilidades ínfimas).

El `try/finally` funciona como protección adicional (*belt-and-suspenders*): captura
el estado actual de `numpy.seterr` antes de la optimización y lo restaura al
terminar, asegurando robustez incluso si la configuración de errores cambiara en
versiones futuras del paquete.

### 2.5 Guardado de Q-table y reportes

Una vez completada la optimización, `main()` realiza dos pasos finales:

1. **Usar la Q-table preservada** — Si `_best_artifacts` contiene la Q-table del
   mejor trial (corrida completa sin interrupción), se usa directamente sin
   re-entrenar.  Solo si se reanudó un estudio previo y la Q-table no está en
   memoria se aplica re-entrenamiento como fallback (usando la misma `SEED` de
   CONFIG para obtener resultados idénticos):

```python
if _best_artifacts['Q'] is not None and _best_artifacts['value'] >= best.value:
    best_Q = _best_artifacts['Q']
    best_rewards = numpy.array(_best_artifacts['rewards'])
else:
    # Fallback: retrain (solo si se resumió y la Q-table original no está en memoria)
    best_rewards, best_Q, _ = QLearning(env_final, ..., seed=SEED)
```

2. **Generar reportes** mediante `_save_report()`:

| Archivo generado | Contenido |
|------------------|-----------|
| `optimization_results_<fecha>.txt` | Tabla de todos los trials ordenados por rendimiento, mejores hiperparámetros, estadísticas. |
| `optimization_history_<fecha>.png` | Gráfico de convergencia: rendimiento de cada trial y curva de mejor acumulado (*running best*). |
| `hyperparameter_importances_<fecha>.png` | Gráfico de barras con la importancia relativa de cada hiperparámetro (calculada por Optuna con *fANOVA*). |
| `q_best_<fecha>.npy` | Q-table entrenada con los mejores hiperparámetros. |
| `rewards_best_<fecha>.npy` | Historia de recompensas del mejor entrenamiento. |

El gráfico de importancia responde directamente a la pregunta: *¿cuál hiperparámetro
afecta más al rendimiento?*  Esto orienta futuros ajustes manuales o refinamientos
del espacio de búsqueda.

### 2.6 Notificaciones push

El módulo incluye soporte opcional de **notificaciones push** a través de
`utils.notify`.  Si el módulo `utils` está disponible en `sys.path`, se envían
notificaciones en dos momentos:

1. **Progreso cada 10 trials** — El callback `_progress_callback` invoca
   `notify()` cada vez que `done % 10 == 0`, informando trials completados,
   mejor valor hasta el momento, tiempo transcurrido y rendimiento del último
   trial.

2. **Error durante la optimización** — El bloque `if __name__ == '__main__'`
   envuelve `main()` en un `try/except`: si ocurre cualquier excepción, se
   envía una notificación con prioridad `urgent` y el traceback completo antes
   de re-lanzar la excepción.

```python
# Notificación de progreso (cada 10 trials)
if done > 0 and done % 10 == 0:
    notify(
        f"[{_PKG_NAME}] {done}/{n_trials} trials",
        f"Se completaron {done} de {n_trials} trials.\n"
        f"Mejor valor hasta ahora: {best_val:.6f}\n"
        f"Último trial: value={trial.value:.4f}\n"
        f"Tiempo transcurrido: {elapsed:.0f}s ({elapsed / 60:.1f} min)",
        tags="chart_with_upwards_trend"
    )
```

Si `utils.notify` no está disponible (por ejemplo, al ejecutar fuera del
workspace mPES), el import falla silenciosamente y `notify` se reemplaza por
una función no-op (`lambda *a, **kw: None`).

---

## 3. Diagrama de flujo

```
main()
  │
  ├─ parsear argumentos: n_trials, --resume YYYY-MM-DD
  ├─ _load_evaluation_data()
  │
  ├─ crear/cargar estudio Optuna (SQLite)
  │       │
  │       │  ┌──────────────────────────────────────────┐
  │       └──►  study.optimize(objective, n_trials)     │
  │           │                                         │
  │           │  ┌─ trial n ────────────────────────┐   │
  │           │  │ TPE elige θ = (α, γ, ε₀, ε_min, N) │
  │           │  │ QLearning(env, θ) → Q-table      │   │
  │           │  │ run_experiment(Q, 64 seqs) → perf│   │
  │           │  │ return mean(perf)                │   │
  │           │  └─────────────────────────────────┘   │
  │           │  repetir hasta completar n_trials       │
  │           └──────────────────────────────────────────┘
  │
  ├─ imprimir mejores hiperparámetros
  ├─ usar Q-table preservada (o retrain si --resume)
  └─ _save_report() → .txt, .png, .npy
```

---

## 4. Uso

### 4.1 Ejecución básica

```bash
cd /home/mecatronica/Documentos/maximiliano/mPES
source linux_mpes_env/bin/activate

# 50 trials (valor por defecto)
python3 -m PES_Bayesian.ext.optimize_rl

# 100 trials
python3 -m PES_Bayesian.ext.optimize_rl 100
```

### 4.2 Ejecución en segundo plano (recomendada)

```bash
nohup python3 -m PES_Bayesian.ext.optimize_rl 100 \
  > PES_Bayesian/inputs/bayesian_opt.log 2>&1 &
```

Para evitar suspensión de la máquina ver
[bayesian_optimization_guide.md](bayesian_optimization_guide.md) § 2.

### 4.3 Reanudar una corrida interrumpida

**Mismo día** — re-ejecutar el mismo comando; Optuna detecta los trials previos en el
SQLite y continúa:

```bash
python3 -m PES_Bayesian.ext.optimize_rl 100
```

**Día diferente** — usar `--resume` con la fecha original del SQLite:

```bash
python3 -m PES_Bayesian.ext.optimize_rl 100 --resume 2026-02-12
```

### 4.4 Salida esperada

```
══════════════════════════════════════════════════════════════════════
  BAYESIAN OPTIMISATION — Q-LEARNING HYPERPARAMETERS
══════════════════════════════════════════════════════════════════════

ℹ Output directory: .../2026-02-12_BAYESIAN_OPT
ℹ Target number of trials: 100

── Loading Evaluation Data ──
  ● Sequence lengths shape: (64,)
  ● Sequences loaded: 64

── Running Bayesian Optimisation ──
ℹ Search space:
  ● learning_rate    ∈ [0.10, 0.4]   (log scale)
  ● discount_factor  ∈ [0.80, 0.99]
  ● epsilon_initial  ∈ [0.4, 1.0]
  ● epsilon_min      ∈ [0.05, 0.1]
  ● num_episodes     ∈ [800000, 1000000]  (step=100000)

  Trial   1/100  |  value=0.7563  |  best=0.7563  |  elapsed=19s
  Trial   2/100  |  value=0.6941  |  best=0.7563  |  elapsed=76s
  ...
  Trial 100/100  |  value=0.8201  |  best=0.8482  |  elapsed=2934s

✓ Optimisation finished in 2934.2s (48.9 min)

── Best Hyperparameters Found ──
  ● learning_rate             = 0.3597
  ● discount_factor           = 0.8651
  ● epsilon_initial           = 0.6791
  ● epsilon_min               = 0.0848
  ● num_episodes              = 900000
ℹ Mean normalised performance: 0.848200
```

### 4.5 Archivos de salida

Todos se guardan en `PES_Bayesian/inputs/<FECHA>_BAYESIAN_OPT/`:

| Archivo | Descripción |
|---------|-------------|
| `optuna_study_<fecha>.db` | Base de datos SQLite con todo el historial de Optuna (permite reanudar). |
| `q_best_<fecha>.npy` | Q-table `(31,11,10,11)` del mejor trial de la optimización. Lista para copiar a `inputs/q.npy` y usar en el experimento. |
| `rewards_best_<fecha>.npy` | Historia de recompensas promedio del mejor entrenamiento (cada 10 000 episodios). |
| `optimization_results_<fecha>.txt` | Reporte textual completo: mejores parámetros, estadísticas, tabla de todos los trials. |
| `optimization_history_<fecha>.png` | Gráfico de convergencia (rendimiento por trial + curva de mejor acumulado). |
| `hyperparameter_importances_<fecha>.png` | Importancia relativa de cada hiperparámetro (fANOVA). |

---

## 5. Relación con Q-Learning

La ecuación de actualización de Q-Learning usada en `pandemic.py` es:

$$Q(s, a) \;\leftarrow\; Q(s, a) + \alpha\left[r + \gamma\max_{a'}Q(s', a') - Q(s, a)\right]$$

donde:

- $s = (\text{recursos},\; \text{trial},\; \text{severidad})$ — estado discretizado.
- $a \in \{0, 1, \ldots, 10\}$ — recursos asignados a la ciudad actual.
- $r$ — recompensa inmediata del entorno.
- $\alpha$ — tasa de aprendizaje (`learning_rate`).
- $\gamma$ — factor de descuento (`discount_factor`).

La política de exploración es $\varepsilon$-greedy con decaimiento lineal:

$$\varepsilon_t = \max\!\left(\varepsilon_{\min},\;\; \varepsilon_0 - t\cdot\frac{\varepsilon_0 - \varepsilon_{\min}}{N}\right)$$

La optimización Bayesiana actúa **un nivel por encima** de Q-Learning: no modifica
el algoritmo, sino que busca los valores de $(\alpha, \gamma, \varepsilon_0,
\varepsilon_{\min}, N)$ que maximicen el rendimiento de la política resultante.
Es decir, es una **meta-optimización** (optimización de la optimización).

Todos los trials de entrenamiento usan la misma semilla (`SEED` de `CONFIG.py`)
para que las diferencias de rendimiento reflejen exclusivamente los hiperparámetros.

```
┌────────────────────────────────────────────────────┐
│  Nivel externo: Optimización Bayesiana (Optuna)    │
│  Decide θ = (α, γ, ε₀, ε_min, N)                  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Nivel interno: Q-Learning (seed=SEED)       │  │
│  │  Entrena Q-table con θ durante N episodios   │  │
│  │  Q(s,a) ← Q(s,a) + α[r + γ·max Q - Q]      │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  Evaluar Q-table → f(θ) = rendimiento medio        │
│  Devolver f(θ) a Optuna                            │
└────────────────────────────────────────────────────┘
```

---

## 6. Flujo de trabajo completo

La optimización Bayesiana es el **primer paso** de un flujo de tres etapas:

```
1. Optimización Bayesiana         python3 -m PES_Bayesian.ext.optimize_rl [N]
   └─ Buscar mejores (α, γ, ε₀, ε_min, N)
   └─ Guardar Q-table y reporte en inputs/<fecha>_BAYESIAN_OPT/

2. Entrenamiento definitivo       python3 -m PES_Bayesian.ext.train_rl [episodes]
   └─ Usa hiperparámetros fijos en train_rl.py (copiados del mejor trial)
   └─ Guarda Q-table y gráficos en inputs/<fecha>_RL_TRAIN/

3. Experimento                    python3 -m PES_Bayesian
   └─ Lee inputs/q.npy y inputs/rewards.npy
   └─ Ejecuta el agente RL sobre 8 bloques × 8 secuencias
```

### 6.1 Transferencia de hiperparámetros

Después de ejecutar la optimización, los mejores hiperparámetros deben copiarse
manualmente a `ext/train_rl.py` (variables `learning_rate`, `discount_factor`,
`epsilon_initial`, `epsilon_min` y `num_episodes`).

Actualmente `train_rl.py` usa los valores del trial #40:

| Parámetro | Valor |
|-----------|-------|
| `learning_rate` | 0.35965545888114453 |
| `discount_factor` | 0.8650520580454709 |
| `epsilon_initial` | 0.6791201210873763 |
| `epsilon_min` | 0.08483331103075126 |
| `num_episodes` | 900 000 |

### 6.2 Transferencia de Q-table al experimento

Para que `python3 -m PES_Bayesian` use la Q-table entrenada, ésta debe copiarse
a la raíz de `inputs/`:

```bash
cp PES_Bayesian/inputs/<fecha>_RL_TRAIN/q_<fecha>.npy  PES_Bayesian/inputs/q.npy
cp PES_Bayesian/inputs/<fecha>_RL_TRAIN/rewards_<fecha>.npy  PES_Bayesian/inputs/rewards.npy
```

`__main__.py` valida la existencia de `q.npy` y `rewards.npy` antes de iniciar
el experimento; si no los encuentra, muestra un error indicando que se debe
ejecutar `train_rl.py` primero.
