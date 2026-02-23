# Sugerencias futuras: Mejoras al modelo de Reinforcement Learning

## Contexto

Este documento presenta un análisis de mejoras viables al modelo de Q-Learning
tabular del escenario pandémico, como ruta de evolución tras completar la
optimización Bayesiana de hiperparámetros.

### Estado actual del modelo

| Aspecto | Valor |
|---------|-------|
| Algoritmo | Double Q-Learning tabular |
| Espacio de estados | Discreto: `(31 × 11 × 10)` = 3,410 estados |
| Espacio de acciones | Discreto: 11 acciones (0–10 recursos) |
| Episodios por entrenamiento | 800k–1.2M (optimizable por Bayesiana) |
| Steps por episodio | 3–10 |
| Exploración | ε-greedy con decaimiento exponencial + warm-up |
| Reward Shaping | PBRS (Ng et al., 1999), coeficiente β optimizable |
| Optimización de hiperparámetros | Bayesiana (Optuna, TPE) sobre 8 parámetros |
| Semilla de entrenamiento | CONFIG.SEED (default 42) — fija para comparabilidad |
| Rendimiento actual | ~0.84 normalizado sobre 64 secuencias |

---

## Mejoras implementadas

Las tres mejoras propuestas originalmente ya están implementadas e integradas
en `QLearning()` dentro de `pandemic.py`. Ver `mejoras_qlearning.md` para
detalles de teoría e implementación.

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
- **Estado:** ✅ Implementado en `pandemic.py` (`QLearning()`) y `train_rl.py`

#### Criterios de calibración de $\lambda$ y $W$

##### Calibración de $\lambda$ (decay_rate)

El valor de $\lambda$ se deriva analíticamente a partir de la pregunta:
**¿en qué episodio queremos que ε alcance ε_min?**

Dado que ε debe llegar a ε_min en el episodio $T$ (post warm-up), despejamos:

$$\varepsilon_{\min} = \varepsilon_0 \cdot \lambda^{T - W}$$

$$\lambda = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{\frac{1}{T - W}}$$

**Regla práctica:** Se busca que ε_min se alcance entre el 50% y 75% del
entrenamiento, dejando al menos 25% de episodios en explotación pura.

##### Calibración de $W$ (warmup_ratio)

El warm-up garantiza cobertura mínima del espacio de estados antes de explotar.
El criterio es:

$$W \geq \frac{|S|}{E[\text{estados por episodio}]}$$

Con $|S| = 31 \times 11 \times 10 = 3{,}410$ estados y ~6.5 estados/episodio:

$$W_{\min} \approx \frac{3{,}410}{6.5} \approx 525 \text{ episodios}$$

**Regla práctica:** $W \in [1\%, 10\%]$ de $N$. El valor 5% proporciona ~3-4×
cobertura del espacio de estados.

##### Cálculo para $N = 1{,}000{,}000$ episodios

Con los hiperparámetros del modelo:
- $\varepsilon_0 = 0.4915$, $\varepsilon_{\min} = 0.0710$
- $N = 1{,}000{,}000$, $W = 5\% = 50{,}000$ episodios

Objetivo: ε_min al 66% de $N$ ($T = 660{,}000$), dejando 34% para explotación:

$$\lambda = \left(\frac{0.0710}{0.4915}\right)^{\frac{1}{660{,}000 - 50{,}000}} = \left(0.1444\right)^{\frac{1}{610{,}000}} \approx 0.9999968$$

| Episodio | Fracción de N | ε exponencial | ε lineal (ref.) |
|---|---|---|---|
| 0–50,000 | 0–5% | 0.4915 (warm-up) | 0.4705 |
| 100,000 | 10% | 0.4191 | 0.4495 |
| 250,000 | 25% | 0.2608 | 0.3865 |
| 500,000 | 50% | 0.1264 | 0.2813 |
| 660,000 | 66% | 0.0710 (ε_min) | 0.2136 |
| 1,000,000 | 100% | 0.0710 | 0.0710 |

**Resultado:** Con $\lambda = 0.9999968$, el agente tiene ~340,000 episodios
(34% de N) de explotación pura al final, vs 0 episodios con decaimiento lineal.

##### Fórmula general para recalcular $\lambda$ con otro $N$

Si se cambia el número de episodios, recalcular con:

$$\lambda = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{\frac{1}{f \cdot N - W}}$$

donde $f \in [0.50, 0.75]$ es la fracción de $N$ en la que se desea alcanzar
$\varepsilon_{\min}$ (recomendado: $f = 0.66$).

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
- **Estado:** ✅ Implementado en `pandemic.py` (`QLearning()`) y `train_rl.py`

#### Consideraciones teóricas de la implementación

##### Inicialización independiente de las tablas

Ambas tablas $Q_A$ y $Q_B$ se inicializan con valores aleatorios uniformes
en $[-1, 1]$, de forma **independiente**. Esto es esencial para que el
mecanismo de desacoplamiento funcione: si ambas tablas empezaran con los
mismos valores, las actualizaciones iniciales serían idénticas y se perdería
la diversidad que elimina el sesgo optimista.

##### Selección de acciones con la tabla promedio

Durante el entrenamiento, la acción ε-greedy se selecciona con
$\arg\max_a (Q_A(s,a) + Q_B(s,a))$ (equivalente al promedio, sin dividir
por 2 ya que argmax es invariante a escalado positivo). Esto genera una
política de comportamiento más estable que usar una sola tabla, porque los
errores individuales se suavizan.

##### Estado terminal

En el estado terminal (`done=True`), se asigna directamente
$Q_X(s,a) = r$ (sin bootstrap), ya que no hay estado futuro. Esto es
correcto porque $V(s_{\text{terminal}}) = 0$ por definición.

##### Compatibilidad con Reward Shaping

El reward shaping (mejora 3) modifica el valor de `reward` **antes** de
las actualizaciones de Q_A/Q_B. Ambas tablas reciben la misma recompensa
modificada en cada step, lo cual preserva la simetría del algoritmo.
La penalización $\beta$ no distorsiona el desacoplamiento porque afecta
igual a ambas tablas.

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

Aplicar **Potential-Based Reward Shaping** (PBRS, Ng et al., 1999) con una
función potencial basada en la severidad total:

$$\Phi(s) = -\sum_{i=0}^{t} S_i$$

El shaping reward se define como la diferencia de potenciales:

$$F(s, s') = \beta \left(\gamma\, \Phi(s') - \Phi(s)\right) = \beta \left(\sum S_i^{\text{(antes)}} - \gamma \sum S_i^{\text{(después)}}\right)$$

donde:
- $\beta \in [0.01, 0.5]$ es un coeficiente de escalado (hiperparámetro)
- $\gamma$ es el factor de descuento del MDP
- $\Phi(s_{\text{terminal}}) = 0$

Esto da un **bonus** cuando la acción reduce la severidad total, y una
**penalización** cuando la deja crecer. Por el teorema de invarianza de
Ng et al., esta forma **garantiza que la política óptima no cambia**
independientemente del valor de $\beta$.

#### Impacto esperado

- El agente aprende a priorizar ciudades con severidad alta antes de que escalen
- Mejor rendimiento en secuencias largas (más ciudades acumuladas = más impacto
  del efecto compuesto)
- Política más "estratégica" (planificación a futuro) vs la actual que es más
  "reactiva" (solo mira el step actual)

#### Implementación

- **Esfuerzo:** Bajo — agregar PBRS en `QLearning()` (no en `step()`)
- **Hiperparámetros nuevos:** 1 ($\beta$) — se integra en la Bayesiana
- **Riesgo:** Bajo — PBRS garantiza invarianza de política óptima por
  construcción teórica (Ng et al., 1999)
- **Requiere:** Re-entrenar y re-optimizar todos los hiperparámetros
- **Estado:** ✅ Implementado en `pandemic.py` (`QLearning()`) y `train_rl.py`

#### Consideraciones teóricas de la implementación

##### Decisión de diseño: shaping en `QLearning()` vs en `step()`

Se evaluaron dos opciones para inyectar la penalización:

| Criterio | Opción A: modificar `step()` | Opción B: modificar `QLearning()` |
|---|---|---|
| Archivos afectados | `pandemic.py` (entorno) | `pandemic.py` (solo función de entrenamiento) |
| `pygameMediator.py` | Afectado (usa `step()` en runtime) | No afectado |
| `run_experiment()` | Afectado (evalúa con `step()`) | No afectado |
| Evaluación del agente | Recompensas distorsionadas | Recompensas originales |
| Retrocompatible | No — cambia la semántica del entorno | Sí — `penalty_coeff=0.0` = sin cambio |

**Se eligió la Opción B** porque:
1. El shaping es una técnica de **entrenamiento**, no una propiedad del entorno.
   La recompensa verdadera del entorno ($r_t = -\sum S_i$) debe permanecer
   intacta para que la evaluación mida rendimiento real.
2. `pygameMediator.py` y `run_experiment()` usan `step()` sin modificaciones,
   preservando la compatibilidad completa con el pipeline de ejecución y
   evaluación.
3. El parámetro `penalty_coeff=0.0` por defecto hace que la función sea
   100% retrocompatible — llamadas existentes sin este argumento producen
   resultados idénticos.

##### Doble conteo controlado

A diferencia de un shaping aditivo ingenuo ($r' = r - \beta \sum S_i$),
que colapsa a un simple escalado $r' = -(1+\beta) \sum S_i$ y puede
cambiar la política óptima, el PBRS usa una **diferencia de potenciales**:

$$F(s, s') = \beta\left(\sum S_i^{\text{antes}} - \gamma \sum S_i^{\text{después}}\right)$$

Esta forma no simplemente amplífica la recompensa. El término $\sum S_i^{\text{antes}}$
(estado previo) introduce información **diferencial**: el agente recibe feedback
sobre *cómo cambió* la severidad total, no solo sobre *cuánta* hay.

**Garantía teórica (Ng et al., 1999):** Para cualquier MDP $M$ con recompensa
$r$ y cualquier función potencial $\Phi: S \to \mathbb{R}$, la política óptima
bajo $r + F$ (con $F = \gamma \Phi(s') - \Phi(s)$) es idéntica a la política
óptima bajo $r$. Esto significa que $\beta$ **no puede empeorar** la política
óptima — solo puede acelerar o mantener igual la convergencia.

##### Interacción con el efecto compuesto

Con `SEVERITY_MULTIPLIER = 1.4`, una ciudad ignorada crece exponencialmente.
El PBRS captura esto a través de la diferencia de potenciales:

| Step | $\sum S_i$ antes | $\sum S_i$ después | $F$ ($\beta=0.1$, $\gamma=0.9$) |
|---|---|---|---|
| 0 | 8.0 | 11.2 + nuevo | $0.1 \cdot (8.0 - 0.9 \cdot 18.2) = -0.84$ |
| 1 | 18.2 | 25.5 + nuevo | $0.1 \cdot (18.2 - 0.9 \cdot 32.5) = -1.11$ |
| 2 | 32.5 | 45.5 + nuevo | $0.1 \cdot (32.5 - 0.9 \cdot 52.5) = -1.50$ |

La penalización crece con cada step porque la **diferencia de potencial**
se amplifica por el crecimiento exponencial. Esto crea un gradiente de
señal que incentiva la intervención temprana sin distorsionar la política
óptima.

##### Rango recomendado para $\beta$

- $\beta = 0$: Sin shaping (comportamiento original)
- $\beta \in [0.01, 0.1]$: Rango conservador — acelera convergencia levemente
- $\beta \in [0.1, 0.3]$: Rango moderado — aceleración significativa
- $\beta > 0.5$: Rango alto — la señal de shaping domina numéricamente;
  puede causar inestabilidad numérica en las actualizaciones de Q aunque
  la política óptima se preserva teóricamente

**Valor inicial elegido:** $\beta = 0.1$ (conservador). La optimización
Bayesiana debe explorar el rango $[0.01, 0.5]$ en escala logarítmica.

##### Referencia teórica

Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under
reward transformations: Theory and application to reward shaping."
*Proceedings of the 16th International Conference on Machine Learning
(ICML 1999)*, pp. 278–287.

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
│  ┌─ Reward Shaping (PBRS) ─────────────────────┐   │
│  │  Φ(s) = −Σ sᵢ                              │   │
│  │  F = β · (γ·Φ(s') − Φ(s))               │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Hiperparámetros: α, γ, ε₀, ε_min, N, λ, W, β         │
│  (los 5 originales + 3 nuevos)                          │
└─────────────────────────────────────────────────────────┘
```

### Sinergia entre mejoras

| Combinación | Efecto |
|---|---|
| Double Q + Reward Shaping | El PBRS modifica la magnitud de las recompensas por cada transición. Double Q estabiliza el aprendizaje con la nueva señal, evitando sobreestimación amplificada. Ambas tablas reciben la misma recompensa shaped, preservando la simetría del algoritmo. |
| Double Q + Decaimiento exponencial | Double Q necesita más exploración inicial para llenar ambas tablas con estimaciones independientes. El warm-up garantiza esa exploración, y el exponencial la reduce rápido después. |
| Reward Shaping + Decaimiento exponencial | El PBRS da bonus/penalización según cómo cambia la severidad total. El warm-up permite que el agente explore transiciones diversas antes de comprometerse con una política, lo que genera señales de shaping más informativas. |

### Hiperparámetros del modelo combinado

| Símbolo | Parámetro | Origen | Rango sugerido | Escala |
|---|---|---|---|---|
| $\alpha$ | `learning_rate` | Original | $[0.2, 0.4]$ | logarítmica |
| $\gamma$ | `discount_factor` | Original | $[0.80, 0.99]$ | lineal |
| $\varepsilon_0$ | `epsilon_initial` | Original | $[0.4, 1.0]$ | lineal |
| $\varepsilon_{\min}$ | `epsilon_min` | Original | $[0.05, 0.1]$ | lineal |
| $N$ | `num_episodes` | Original | $[800k, 1.2M]$ | paso 10k |
| $w$ | `warmup_ratio` | **Nuevo** | $[0.01, 0.1]$ | lineal (fracción de N) |
| $f$ | `target_ratio` | **Nuevo** | $[0.50, 0.80]$ | lineal |
| $\beta$ | `penalty_coeff` | **Nuevo** | $[0.001, 0.5]$ | logarítmica |

Todos los trials de entrenamiento usan la misma semilla (`SEED` de `CONFIG.py`,
default 42) para que las diferencias de rendimiento reflejen exclusivamente
los hiperparámetros.

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

## Plan de implementación

```
Fase 1
└── ✅ Optimización Bayesiana sobre Q-Learning original (5 params)
     └── Resultado: mejores α, γ, ε₀, ε_min, N

Fase 2
├── ✅ Implementar las 3 mejoras en QLearning()
│   ├── ✅ 2a. Decaimiento exponencial + warm-up
│   ├── ✅ 2b. Double Q-Learning
│   └── ✅ 2c. Reward Shaping (PBRS)
│
└── ✅ Adaptar optimize_rl.py para los 3 nuevos params
     (+ semilla fija SEED en CONFIG para comparabilidad)

Fase 3 (siguiente)
└── Re-optimización Bayesiana sobre QLearning mejorado (8 params)
    ├── Mismo benchmark (64 secuencias)
    ├── Gráficos de convergencia
    └── Análisis de importancia de hiperparámetros nuevos
```

### Tiempo estimado

| Tarea | Tiempo |
|---|---|
| ✅ Implementar las 3 mejoras en `QLearning()` | Completado |
| ✅ Adaptar `optimize_rl.py` para los 3 nuevos params | Completado |
| ✅ Centralizar semilla en `CONFIG.SEED` | Completado |
| Correr Bayesiana ~200 trials | 1-2 horas (background) |
| Comparar resultados | 30 min |
| **Total restante** | **~1.5–2.5 horas** |

---

## Métricas de éxito

La implementación se considera exitosa si `QLearning_v2` cumple al menos
uno de estos criterios respecto al Q-Learning original optimizado:

1. **Rendimiento medio ≥ 0.86** (mejora de ≥2% sobre ~0.84 actual)
2. **Convergencia en ≤50% de episodios** para alcanzar el mismo rendimiento
3. **Menor varianza** entre las 64 secuencias (std_perf más bajo)

---

## Evaluación de Transformers como alternativa a Reinforcement Learning

### Contexto

Se evaluó la viabilidad de reemplazar el Q-Learning tabular por una
arquitectura basada en Transformers (Decision Transformer, Attention Model).
El análisis se realizó en dos escenarios: la escala actual del problema y
una posible futura ampliación con más ciudades y mapas.

### Escenario actual (3,410–3,751 estados, 3–10 steps)

**Veredicto: No recomendado.**

| Factor | Q-Learning tabular | Transformer |
|---|---|---|
| Parámetros | ~41K (Q-table) | ~500K–5M |
| Memoria | ~160 KB | ~20–200 MB |
| GPU requerida | No | Sí (CUDA) |
| Tiempo de entrenamiento | ~2 min (40K ep.) | ~1–4 horas |
| Dependencias | NumPy + Gym | PyTorch, HuggingFace |
| Interpretabilidad | Alta (tabla consultable) | Baja (caja negra) |

#### Razones de descarte a escala actual

1. **Secuencias demasiado cortas.** Los Transformers requieren secuencias
   largas (100–1000+ tokens) para que la atención multi-cabezal aprenda
   dependencias útiles. Con episodios de 3–10 steps, la matriz de atención
   es a lo sumo $10 \times 10$ — trivialmente pequeña. No hay dependencias
   de largo alcance que justifiquen self-attention.

2. **Espacio de estados completamente enumerable.** La Q-table cubre
   exhaustivamente los 3,751 estados posibles con 11 acciones. Un
   Transformer necesitaría millones de parámetros para aproximar una
   función que la tabla resuelve de forma exacta.

3. **Datos de entrenamiento insuficientes.** Un Decision Transformer
   requiere datasets offline de trayectorias (decenas de miles). PES tiene
   64 secuencias de evaluación. Incluso generando datos sintéticos, las
   trayectorias de 3–10 steps producen un dataset diminuto.

4. **Overhead computacional desproporcionado.** GPU obligatoria, mayor
   inestabilidad de entrenamiento, y complejidad de implementación extrema
   sin beneficio demostrable.

### Escenario escalado (50–500+ ciudades, múltiples mapas)

**Veredicto: Recomendado si se amplía significativamente el problema.**

Si se incrementa la dimensión del problema (más ciudades por secuencia,
más mapas con diferentes distribuciones geográficas), el Transformer
se vuelve la arquitectura más apropiada gracias a tres ventajas clave:

#### 1. Generalización de tamaño

Un Transformer entrenado con instancias de $N$ ciudades puede resolver
instancias de $M > N$ ciudades **sin reentrenar**. Esto es imposible con
Q-Learning tabular (la tabla tiene dimensiones fijas) y muy difícil con
RL basado en redes feedforward.

#### 2. Generalización entre mapas

Un solo modelo resuelve mapas con distribuciones diferentes (uniforme,
clusters, anillo, grilla) sin re-optimizar hiperparámetros. El enfoque
actual requiere reentrenamiento completo para cada nuevo mapa.

#### 3. Curriculum Learning

Se puede entrenar incrementalmente:
- Fase 1: 20–30 ciudades (rápido, establece representaciones)
- Fase 2: 50–80 ciudades (fine-tuning)
- Fase 3: Evaluación zero-shot en 100–500 ciudades

#### Arquitectura recomendada para el escenario escalado

La arquitectura más probada es el **Attention Model** (Kool et al., 2019):

```
┌─────────────────────────────────────────────────────────────────┐
│  PES Transformer Solver                                         │
│                                                                 │
│  ┌─ Encoder (Auto-atención) ──────────────────────────────┐    │
│  │  Entrada: [x, y, severidad, costo] por ciudad           │    │
│  │  Multi-Head Self-Attention × 3–6 capas                  │    │
│  │  Codifica relaciones globales entre todas las ciudades   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─ Decoder (Autoregresivo) ──────────────────────────────┐    │
│  │  Selecciona la siguiente ciudad a visitar               │    │
│  │  Pointer Network: atención sobre encodings              │    │
│  │  Máscara de ciudades ya visitadas                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Entrenamiento: REINFORCE + baseline exponencial                │
│  Inferencia: greedy (argmax) o beam search                      │
└─────────────────────────────────────────────────────────────────┘
```

Para problemas con 500+ ciudades, se puede reducir la complejidad de
$O(n^2)$ a $O(nk)$ usando **atención por vecinos geográficos** (solo
atender a los $k$ vecinos más cercanos en lugar de todas las ciudades).

### Tabla comparativa por escala

| Escala | Evolutivo | RL (Q-Learning) | Transformer |
|---|---|---|---|
| < 20 ciudades | ✅ Mejor | ⚠ Overkill | ⚠ Overkill |
| 20–50 | ⚠ Lento | ✅ Funciona | ✅ Funciona |
| 50–100 | ❌ Muy lento | ⚠ Difícil | ✅ Ideal |
| 100–500 | ❌ Intratable | ❌ No converge | ✅ Eficiente |
| 500+ | ❌ | ❌ | ✅ + Atención local |
| Multi-mapa | ❌ Reentrenar | ❌ Reentrenar | ✅ Generaliza |

### Requisitos para implementar Transformers

| Requisito | Detalle |
|---|---|
| Framework | PyTorch ≥ 2.0 |
| Hardware | GPU con CUDA (mínimo 4 GB VRAM) |
| Datos de entrenamiento | Generados sintéticamente por el entorno |
| Tiempo de implementación | ~1–2 semanas |
| Tiempo de entrenamiento | ~2–8 horas (GPU) |
| Parámetros del modelo | ~500K–2M |

### Conclusión

- **A la escala actual del problema (3–10 steps, 3,751 estados):** el
  Transformer **no aporta valor**. Las mejoras incrementales al Q-Learning
  tabular (Double Q-Learning, reward shaping, decaimiento exponencial)
  son la ruta correcta.

- **Si se escala a 50+ ciudades y múltiples mapas:** el Transformer
  se convierte en la **mejor opción** gracias a su capacidad de
  generalización entre tamaños e instancias. En ese escenario, se
  recomienda implementar un Attention Model con REINFORCE y curriculum
  learning.

- **Umbral de decisión:** si los episodios superan ~30 steps y el
  espacio de estados deja de ser enumerable (> 100K estados), el
  Transformer justifica su complejidad adicional.