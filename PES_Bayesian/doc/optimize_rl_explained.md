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
| $\alpha$ | `learning_rate` | $[0.01,\; 0.5]$ | logarítmica |
| $\gamma$ | `discount_factor` | $[0.80,\; 0.99]$ | lineal |
| $\varepsilon_0$ | `epsilon_initial` | $[0.30,\; 1.00]$ | lineal |
| $\varepsilon_{\min}$ | `epsilon_min` | $[0.00,\; 0.10]$ | lineal |
| $N$ | `num_episodes` | $[5\,000,\; 40\,000]$ | paso = 5 000 |

Probar todas las combinaciones (grid search) requiere un número exponencial de
evaluaciones.  Cada evaluación implica **entrenar una Q-table completa** y después
evaluar sobre 64 secuencias fijas, lo que tarda ~10–60 s por combinación dependiendo
del `num_episodes` muestreado.

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

donde $\text{perf}$ es la métrica de severidad final normalizada implementada
en `pandemic.py`.  Un valor cercano a $1.0$ indica que el agente redujo la severidad
al mínimo posible con los recursos disponibles.

---

## 2. Estructura del código

El archivo `optimize_rl.py` tiene cuatro secciones principales:

```
optimize_rl.py
├── _load_evaluation_data()     # Datos de evaluación (carga una vez)
├── objective(trial)            # Función objetivo que Optuna llama
├── _save_report(study, ...)    # Reportes y gráficos
└── main()                      # Orquestación: CLI, estudio, re-entrenamiento
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
    learning_rate   = trial.suggest_float('learning_rate',   0.01, 0.5, log=True)
    discount_factor = trial.suggest_float('discount_factor', 0.80, 0.99)
    epsilon_initial = trial.suggest_float('epsilon_initial', 0.3,  1.0)
    epsilon_min     = trial.suggest_float('epsilon_min',     0.0,  0.1)
    num_episodes    = trial.suggest_int('num_episodes',    5000, 40000, step=5000)

    # (2) Entrenar Q-table
    env = Pandemic()
    env.number_cities_prob = _number_cities_prob
    env.severity_prob      = _severity_prob
    env.verbose = False
    rewards, Q, _ = QLearning(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes
    )

    # (3) Evaluar sobre 64 secuencias fijas
    env_eval = Pandemic()
    env_eval.verbose = False

    def qf(env, state, seqid):
        s0 = min(int(state[0]), Q.shape[0] - 1)
        s1 = min(int(state[1]), Q.shape[1] - 1)
        s2 = min(int(state[2]), Q.shape[2] - 1)
        return numpy.argmax(Q[s0, s1, s2])

    _, perfs, _ = run_experiment(
        env_eval, qf, False, _trials_per_sequence, _sevs)
    mean_perf = float(numpy.mean(perfs))

    # (4) Guardar métricas auxiliares
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf',  float(numpy.std(perfs)))
    trial.set_user_attr('min_perf',  float(numpy.min(perfs)))
    trial.set_user_attr('max_perf',  float(numpy.max(perfs)))

    return mean_perf
```

**Correspondencia teoría ↔ código:**

| Concepto teórico | Implementación |
|------------------|----------------|
| $\boldsymbol{\theta}$ muestreado por TPE | `trial.suggest_*()` — Optuna aplica el modelo sustituto para decidir qué valores probar. |
| Evaluación $f(\boldsymbol{\theta})$ | Entrenar Q-table + evaluar con `run_experiment()` sobre 64 secuencias fijas. |
| Resultado devuelto a Optuna | `return mean_perf` — Optuna lo usa para actualizar su modelo sustituto y elegir el próximo $\boldsymbol{\theta}$. |

#### Detalle de `trial.suggest_*`

- `suggest_float('learning_rate', 0.01, 0.5, log=True)`:
  Muestrea $\alpha$ en escala logarítmica porque valores bajos
  ($0.01$–$0.05$) y altos ($0.1$–$0.5$) tienen impactos cualitativamente
  diferentes.  La escala log da igual densidad de muestreo a cada orden de
  magnitud.

- `suggest_int('num_episodes', 5000, 40000, step=5000)`:
  Discretiza en múltiplos de 5 000 para reducir la dimensionalidad sin
  perder resolución práctica.

#### Función `qf` (política greedy)

Dentro de `objective`, se define una política puramente greedy (sin exploración):

```python
def qf(env, state, seqid):
    s0 = min(int(state[0]), Q.shape[0] - 1)
    s1 = min(int(state[1]), Q.shape[1] - 1)
    s2 = min(int(state[2]), Q.shape[2] - 1)
    return numpy.argmax(Q[s0, s1, s2])
```

Esta función se pasa a `run_experiment()` como la *action function*.  Evalúa la
Q-table **sin epsilon**: siempre elige la acción con mayor Q-valor.  Así se mide
únicamente la calidad de la política aprendida, sin ruido de exploración.

Los `min(...)` aseguran que los índices de estado no excedan las dimensiones de la
Q-table `(31, 11, 10, 11)` = (recursos, trial, severidad, acciones).

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

El módulo `PES_Bayesian` configura `numpy.seterr(all='raise')` para detectar errores
numéricos durante la simulación.  Sin embargo, el muestreador TPE calcula internamente
`numpy.exp(x)` donde $x$ puede ser muy negativo ($x < -700$), produciendo un *underflow*
a $0.0$ que es matemáticamente inofensivo (simplemente indica probabilidades ínfimas).

El `try/finally` desactiva la excepción de underflow solo durante la optimización
y la restaura al terminar.

### 2.5 Re-entrenamiento y reportes

Una vez completada la optimización, `main()` realiza dos pasos finales:

1. **Re-entrenar** la Q-table con los mejores hiperparámetros encontrados:

```python
bp = best.params
best_rewards, best_Q, _ = QLearning(
    env_final,
    bp['learning_rate'], bp['discount_factor'],
    bp['epsilon_initial'], bp['epsilon_min'],
    bp['num_episodes']
)
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
  ├─ re-entrenar Q-table final con mejores θ
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
  ● learning_rate    ∈ [0.01, 0.5]   (log scale)
  ● discount_factor  ∈ [0.80, 0.99]
  ● epsilon_initial  ∈ [0.30, 1.00]
  ● epsilon_min      ∈ [0.00, 0.10]
  ● num_episodes     ∈ [5000, 40000]  (step=5000)

  Trial   1/100  |  value=0.7563  |  best=0.7563  |  elapsed=19s
  Trial   2/100  |  value=0.6941  |  best=0.7563  |  elapsed=76s
  ...
  Trial 100/100  |  value=0.8201  |  best=0.8482  |  elapsed=2934s

✓ Optimisation finished in 2934.2s (48.9 min)

── Best Hyperparameters Found ──
  ● learning_rate             = 0.07234
  ● discount_factor           = 0.9412
  ● epsilon_initial           = 0.872
  ● epsilon_min               = 0.0321
  ● num_episodes              = 25000
ℹ Mean normalised performance: 0.848200
```

### 4.5 Archivos de salida

Todos se guardan en `PES_Bayesian/inputs/<FECHA>_BAYESIAN_OPT/`:

| Archivo | Descripción |
|---------|-------------|
| `optuna_study_<fecha>.db` | Base de datos SQLite con todo el historial de Optuna (permite reanudar). |
| `q_best_<fecha>.npy` | Q-table `(31,11,10,11)` entrenada con los mejores hiperparámetros. Lista para usar en el experimento. |
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

```
┌────────────────────────────────────────────────┐
│  Nivel externo: Optimización Bayesiana (Optuna)│
│  Decide θ = (α, γ, ε₀, ε_min, N)              │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │  Nivel interno: Q-Learning               │  │
│  │  Entrena Q-table con θ durante N episodios│  │
│  │  Q(s,a) ← Q(s,a) + α[r + γ·max Q - Q]  │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  Evaluar Q-table → f(θ) = rendimiento medio    │
│  Devolver f(θ) a Optuna                        │
└────────────────────────────────────────────────┘
```
