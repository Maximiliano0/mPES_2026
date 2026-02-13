# Sugerencias futuras: Mejoras al modelo de Reinforcement Learning

## Contexto

Este documento presenta un análisis de mejoras viables al modelo de Q-Learning
tabular del escenario pandémico, como ruta de evolución tras completar la
optimización Bayesiana de hiperparámetros.

### Estado actual del modelo

| Aspecto | Valor |
|---------|-------|
| Algoritmo | Q-Learning tabular |
| Espacio de estados | Discreto: `(31 × 11 × 10)` = 3,410 estados |
| Espacio de acciones | Discreto: 11 acciones (0–10 recursos) |
| Episodios por entrenamiento | 500k–2M (optimizable por Bayesiana) |
| Steps por episodio | 3–10 |
| Exploración | ε-greedy con decaimiento lineal |
| Optimización de hiperparámetros | Bayesiana (Optuna, TPE) sobre 5 parámetros |
| Rendimiento actual | ~0.84 normalizado sobre 64 secuencias |

---

## Mejoras propuestas

Se proponen tres mejoras combinables entre sí, ordenadas por facilidad de
implementación. Las tres son compatibles con la optimización Bayesiana existente.

---

### 1. Decaimiento exponencial de ε con warm-up

#### Problema

El decaimiento lineal actual distribuye la exploración uniformemente en el tiempo:

$$\varepsilon_t^{\text{lineal}} = \max\!\left(\varepsilon_{\min},\; \varepsilon_0 - t \cdot \frac{\varepsilon_0 - \varepsilon_{\min}}{N}\right)$$

Esto es subóptimo porque:
- Al inicio, la Q-table es aleatoria y necesita **mucha exploración** para descubrir
  qué acciones son buenas.
- Al final, la tabla ya convergió y la exploración solo introduce ruido que degrada
  la política.
- El decaimiento lineal explora demasiado al final y proporcionalmente poco al inicio.

#### Solución

Reemplazar por decaimiento exponencial con fase de warm-up:

$$\varepsilon_t = \begin{cases} \varepsilon_0 & \text{si } t < W \quad \text{(warm-up: exploración pura)} \\ \max\!\left(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t-W}\right) & \text{si } t \geq W \end{cases}$$

donde:
- $\lambda \in [0.9990, 0.9999]$ controla la velocidad de decaimiento
- $W$ es la duración del warm-up (ej: 5% de $N$)

Comparación visual:

```
ε
1.0 ┤■■■■■\
    │      \\  exponencial
0.5 ┤       \\___
    │  lineal \   \___
    │          \      \___
0.0 ┤───────────\─────────\──────
    0          N/2          N
```

#### Impacto esperado

- Convergencia más rápida (la tabla se estabiliza antes)
- Mejor rendimiento final con el mismo número de episodios
- El warm-up garantiza que todos los estados se visiten al menos una vez

#### Implementación

- **Esfuerzo:** Mínimo — cambiar una línea en el bucle de entrenamiento
- **Hiperparámetros nuevos:** 2 ($\lambda$, $W$) — se integran en la Bayesiana
- **Riesgo:** Nulo — si $\lambda$ y $W$ no ayudan, la Bayesiana los descartará

---

### 2. Double Q-Learning

#### Problema

Q-Learning estándar sobreestima los Q-valores porque usa el mismo operador `max`
para **seleccionar** la mejor acción y para **evaluar** su valor:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \underbrace{\max_{a'} Q(s',a')}_{\text{sesgo optimista}} - Q(s,a)\right]$$

Cuando las estimaciones tienen ruido (siempre al inicio del entrenamiento), `max`
selecciona sistemáticamente valores sobreestimados. Esto produce:
- Políticas que asignan recursos de forma excesivamente agresiva
- Convergencia inestable
- Q-valores inflados que no reflejan el rendimiento real

#### Solución

Mantener **dos Q-tables** independientes $Q_A$ y $Q_B$. En cada step, con
probabilidad 0.5 se actualiza una u otra:

$$Q_A(s,a) \leftarrow Q_A(s,a) + \alpha\left[r + \gamma\, Q_B\!\left(s',\, \arg\max_{a'} Q_A(s',a')\right) - Q_A(s,a)\right]$$

Una tabla **selecciona** la acción (argmax), la otra la **evalúa** (valor). Como
los errores de estimación de ambas tablas son independientes, el sesgo optimista
se cancela en promedio.

Para la evaluación final (función `qf`), se promedian ambas tablas:

$$Q_{\text{eval}}(s,a) = \frac{Q_A(s,a) + Q_B(s,a)}{2}$$

#### Impacto esperado

- Política más estable y conservadora en asignación de recursos
- Mejor rendimiento con menos episodios de entrenamiento
- Convergencia más suave

#### Implementación

- **Esfuerzo:** Bajo — duplicar la Q-table, alternar actualización con `random() < 0.5`
- **Memoria adicional:** ×2 (37 KB → 74 KB, trivial)
- **Hiperparámetros nuevos:** 0 — usa los mismos $\alpha$ y $\gamma$
- **Riesgo:** Bajo — en el peor caso rinde igual que Q-Learning simple
- **Compatible con Bayesiana:** sin cambios en `optimize_rl.py`

---

### 3. Reward Shaping con penalización por severidad residual acumulada

#### Problema

La recompensa actual evalúa cada acción de forma local (step a step). El agente
no recibe señal sobre el **impacto compuesto** de dejar ciudades con severidad alta:

- Una ciudad con severidad 8 a la que no se asignan recursos crece por el
  `SEVERITY_MULTIPLIER` (1.4×) en cada step siguiente:
  - Step 0: 8.0
  - Step 1: 11.2
  - Step 2: 15.7
  - Step 3: 22.0

El agente puede aprender a ignorar ciudades "difíciles" porque la penalización
inmediata por no atenderlas es la misma que por no atender ciudades "fáciles".

#### Solución

Agregar un término de penalización que capture el costo futuro estimado de la
severidad residual:

$$r'_t = r_t - \beta \sum_{i=0}^{t} S_i^{(\text{residual})}$$

donde:
- $\beta \in [0.01, 0.5]$ es un coeficiente de penalización (hiperparámetro)
- $S_i^{(\text{residual})} = \max(0,\; \text{SEVERITY\_MULTIPLIER} \cdot S_i - \text{RESPONSE\_MULTIPLIER} \cdot a_i)$

Esto enseña al agente que **ignorar ciudades con alta severidad tiene un costo
acumulativo creciente**, incentivando asignaciones preventivas tempranas.

#### Impacto esperado

- El agente aprende a priorizar ciudades con severidad alta antes de que escalen
- Mejor rendimiento en secuencias largas (más ciudades acumuladas = más impacto
  del efecto compuesto)
- Política más "estratégica" (planificación a futuro) vs la actual que es más
  "reactiva" (solo mira el step actual)

#### Implementación

- **Esfuerzo:** Medio — modificar la función `step()` del entorno para calcular
  la penalización acumulada
- **Hiperparámetros nuevos:** 1 ($\beta$) — se integra en la Bayesiana
- **Riesgo:** Medio — un $\beta$ demasiado alto distorsiona la señal de recompensa
  original; la Bayesiana debe calibrarlo
- **Requiere:** Re-entrenar y re-optimizar todos los hiperparámetros

---

## Combinación de las tres mejoras

Las tres propuestas se refuerzan mutuamente y pueden implementarse juntas en una
nueva función `QLearning_v2()`:

```
┌─────────────────────────────────────────────────────────┐
│  QLearning_v2(env, α, γ, ε₀, ε_min, N, λ, W, β)       │
│                                                         │
│  ┌─ Double Q-Learning ─────────────────────────────┐   │
│  │  Q_A, Q_B: dos tablas (31,11,10,11)             │   │
│  │  Selección con Q_A, evaluación con Q_B           │   │
│  │  (o viceversa, alternando con prob 0.5)          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ Decaimiento exponencial + warm-up ─────────────┐   │
│  │  if t < W: ε = ε₀  (exploración pura)           │   │
│  │  else:     ε = max(ε_min, ε₀ · λ^(t-W))         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ Reward Shaping ────────────────────────────────┐   │
│  │  r' = r - β · Σ severidades_residuales          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Hiperparámetros: α, γ, ε₀, ε_min, N, λ, W, β         │
│  (los 5 originales + 3 nuevos)                          │
└─────────────────────────────────────────────────────────┘
```

### Sinergia entre mejoras

| Combinación | Efecto |
|---|---|
| Double Q + Reward Shaping | El reward shaping cambia la magnitud de las recompensas. Sin Double Q, las nuevas recompensas pueden amplificar la sobreestimación de Q-valores. Double Q estabiliza el aprendizaje con la nueva señal. |
| Double Q + Decaimiento exponencial | Double Q necesita más exploración inicial para llenar ambas tablas con estimaciones independientes. El warm-up garantiza esa exploración, y el exponencial la reduce rápido después. |
| Reward Shaping + Decaimiento exponencial | La penalización por severidad residual es más informativa al inicio (cuando el agente ignora ciudades). El warm-up le permite descubrir ese patrón antes de comprometerse con una política. |

### Hiperparámetros del modelo combinado

| Símbolo | Parámetro | Origen | Rango sugerido | Escala |
|---|---|---|---|---|
| $\alpha$ | `learning_rate` | Original | $[0.01, 0.5]$ | logarítmica |
| $\gamma$ | `discount_factor` | Original | $[0.80, 0.99]$ | lineal |
| $\varepsilon_0$ | `epsilon_initial` | Original | $[0.3, 1.0]$ | lineal |
| $\varepsilon_{\min}$ | `epsilon_min` | Original | $[0.0, 0.1]$ | lineal |
| $N$ | `num_episodes` | Original | $[5k, 40k]$ | paso 5k |
| $\lambda$ | `decay_rate` | **Nuevo** | $[0.9990, 0.9999]$ | logarítmica |
| $W$ | `warmup_ratio` | **Nuevo** | $[0.0, 0.1]$ | lineal (fracción de N) |
| $\beta$ | `penalty_coeff` | **Nuevo** | $[0.01, 0.5]$ | logarítmica |

Pasar de 5 a 8 hiperparámetros aumenta los trials de Bayesiana necesarios
de ~100 a ~150-200, pero sigue siendo factible (~1-2 horas de cómputo).

---

## Alternativas evaluadas y descartadas

### Policy Gradient (REINFORCE)

**Descartado** porque:
- Alta varianza del gradiente con episodios cortos (3-10 steps)
- Convergencia más lenta que Q-Learning para espacios discretos pequeños
- No aporta ventaja cuando las acciones son discretas y pocas (11)

### Actor-Critic (A2C tabular)

**Descartado** porque:
- Agrega complejidad (2 tablas, 2 learning rates) sin ganancia significativa
  para 3,410 estados
- La política estocástica elimina ε-greedy pero no mejora rendimiento en
  este tamaño de problema

### Deep Q-Network (DQN)

**Descartado** porque:
- El espacio de estados (3,410) es tan pequeño que una tabla lo cubre
  completamente — la red neuronal no tiene dónde generalizar
- Requiere PyTorch/TensorFlow, GPU, replay buffer, target network
- Mayor inestabilidad de entrenamiento sin beneficio

### Black Mamba / Decision Transformer

**Descartado** porque:
- Diseñados para secuencias largas (100-1000+ tokens) con modelos de
  millones de parámetros — desproporcionado para episodios de 3-10 steps
- Requieren GPU con CUDA obligatorio
- No existen como algoritmos de RL nativos — habría que adaptar
  arquitecturas de lenguaje
- Complejidad de implementación extrema sin beneficio demostrable

---

## Plan de implementación recomendado

```
Fase 1 (actual)
└── ✅ Optimización Bayesiana sobre Q-Learning original (5 params)
     └── Resultado: mejores α, γ, ε₀, ε_min, N

Fase 2 (siguiente)
├── Implementar QLearning_v2() con las 3 mejoras combinadas
│   ├── 2a. Decaimiento exponencial + warm-up (~15 min)
│   ├── 2b. Double Q-Learning (~30 min)
│   └── 2c. Reward Shaping (~1 hora)
│
└── Re-optimización Bayesiana sobre QLearning_v2 (8 params, ~200 trials)
     └── Resultado: mejores α, γ, ε₀, ε_min, N, λ, W, β

Fase 3 (comparación)
└── Comparar rendimiento QLearning vs QLearning_v2
    ├── Mismo benchmark (64 secuencias)
    ├── Gráficos de convergencia
    └── Análisis de importancia de hiperparámetros nuevos
```

### Tiempo estimado

| Tarea | Tiempo |
|---|---|
| Implementar `QLearning_v2()` | 2 horas |
| Adaptar `optimize_rl.py` para v2 | 30 min |
| Correr Bayesiana 200 trials | 1-2 horas (background) |
| Comparar resultados | 30 min |
| **Total** | **~5 horas** |

---

## Métricas de éxito

La implementación se considera exitosa si `QLearning_v2` cumple al menos
uno de estos criterios respecto al Q-Learning original optimizado:

1. **Rendimiento medio ≥ 0.86** (mejora de ≥2% sobre ~0.84 actual)
2. **Convergencia en ≤50% de episodios** para alcanzar el mismo rendimiento
3. **Menor varianza** entre las 64 secuencias (std_perf más bajo)