# Mapeo Teoría Transformer RL ↔ Implementación en pes_trf

## 1. Introducción

Este documento conecta la **teoría de Transformers causales con
actor-critic** con su **implementación concreta** en el paquete `pes_trf`.
Para cada concepto teórico se indica la variable, función o línea de
código correspondiente.

`pes_trf` reemplaza las redes fully-connected de `pes_dqn` y `pes_ac`
por un **Transformer causal** con cabezas de actor-critic.  El modelo
procesa la **trayectoria completa** de estados dentro de un episodio
(hasta 10 timesteps) y produce una distribución de acción y un valor
escalar en cada posición.

Conserva el mismo entorno Gym (`Pandemic`), el mismo espacio de estados y
acciones, y la misma integración con la UI de Pygame.

---

## 2. Componentes del MDP (heredados de `pes_base`)

El MDP subyacente es idéntico al del paquete base:

| Componente | Símbolo | Implementación |
|------------|---------|----------------|
| Estados | $S$ | `(resources_left, trial_no, severity)` — 3 dimensiones discretas |
| Acciones | $A$ | $\{0, 1, 2, \dots, 10\}$ — recursos a asignar |
| Transiciones | $P(s' \mid s, a)$ | Determinísticas: `env.step(action)` en `pandemic.py` |
| Recompensas | $R(s, a)$ | $-\sum_{i} \text{severities}_i$ |
| Factor de descuento | $\gamma$ | `gamma` (por defecto 0.98 en `train_transformer.py`) |

**Cardinalidad del espacio de estados** (con `MAX_SEVERITY = 9`):

$$|S| = 31 \times 11 \times 10 = 3{,}410 \text{ estados}$$

---

## 3. ¿Por qué un Transformer?

### 3.1 Filosofía del Enfoque

Los enfoques anteriores (`pes_dqn`, `pes_ac`) procesan cada estado de
forma **independiente** — una red feedforward recibe $(r, t, v)$ y
produce Q-values o probabilidades de acción.  El historial del episodio
se descarta.

Un Transformer causal procesa la **secuencia completa de estados** del
episodio, permitiendo que cada decisión tenga en cuenta todo el contexto
anterior.  Esto es especialmente relevante en el Pandemic Scenario donde
las decisiones pasadas afectan los recursos disponibles y las severidades
futuras.

### 3.2 Comparación

| Aspecto | DQN (`pes_dqn`) | A2C (`pes_ac`) | Transformer (`pes_trf`) |
|---------|-----------------|----------------|-------------------------|
| Entrada | Estado individual $(r, t, v)$ | Estado individual $(r, t, v)$ | Trayectoria $\{s_0, s_1, \dots, s_t\}$ |
| Arquitectura | MLP 3→64→64→11 | Actor MLP + Critic MLP | Transformer encoder + heads |
| Contexto temporal | Ninguno | Ninguno | Atención causal sobre toda la secuencia |
| Parámetros | ~5 131 | ~9 612 | ~77 000 |
| Algoritmo | DQN + replay buffer | A2C (on-policy, per-episode) | REINFORCE + value baseline + GAE |
| Exploración | ε-greedy sobre Q | Muestreo de π + ε-overlay | Muestreo de π + entropy bonus |

### 3.3 Ventaja Temporal del Transformer

La atención causal permite al modelo aprender patrones como:

- "Si ya gasté muchos recursos, debo ser conservador ahora."
- "La severidad ha crecido en los últimos trials → necesito intervenir."
- Correlaciones entre la posición en la secuencia y la estrategia óptima.

---

## 4. Arquitectura del Modelo

### 4.1 Visión General

Implementada en `ext/transformer_model.py` → `PandemicTransformer`:

```text
Input:  (batch, seq_len, 3)  — [resources_left, trial_no, severity]
  ↓ Normalise to [0, 1]³  (/ [30, 10, 9])
  ↓ Linear projection → (batch, seq_len, d_model=64)
  ↓ + Positional encoding aprendido (1, 10, 64)
  ↓ Input dropout (0.1)
  ↓
  L=2 × TransformerBlock
  │  ├─ LayerNorm → CausalSelfAttention(4 heads) → Dropout → Residual
  │  └─ LayerNorm → FFN(Dense(128,GELU) → Dense(64)) → Dropout → Residual
  ↓
  LayerNormalization final
  ├→ Policy head:  Dense(64, ReLU) → Dense(11)       [logits]
  └→ Value head:   Dense(64, ReLU) → Dense(1)        [V(s)]
```

### 4.2 Hiperparámetros por Defecto

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `d_model` | 64 | Dimensión de embedding |
| `n_heads` | 4 | Cabezas de atención (depth = 16 por cabeza) |
| `n_layers` | 2 | Bloques Transformer |
| `d_ff` | 128 | Dimensión oculta del FFN |
| `max_seq_len` | 10 | Longitud máxima de episodio |
| `n_actions` | 11 | Espacio de acciones discreto |
| `dropout_rate` | 0.1 | Dropout en todas las sub-capas |

### 4.3 Conteo de Parámetros (~77 000)

| Componente | Estimación |
|------------|------------|
| State projection (3→64) | 256 |
| Positional embeddings (1×10×64) | 640 |
| 2× TransformerBlock (Q,K,V + FFN + LN) | ~70 000 |
| Policy head (64→64→11) | ~4 800 |
| Value head (64→64→1) | ~4 200 |
| Final LayerNorm | 128 |
| **Total** | **~77 000** |

### 4.4 Normalización del Estado

Antes de alimentar el modelo, el estado se escala internamente a $[0, 1]^3$:

$$\hat{s} = \left(\frac{r}{30},\; \frac{t}{10},\; \frac{v}{9}\right)$$

Implementada dentro de `PandemicTransformer.call()`:

```python
STATE_MAX = tf.constant([30.0, 10.0, 9.0], dtype=tf.float32)
x = tf.cast(states, tf.float32) / self.STATE_MAX
```

---

## 5. Atención Causal (Causal Self-Attention)

### 5.1 Motivación Teórica

En un Transformer estándar (encoder), cada posición puede atender a
**todas** las demás.  Para decisiones secuenciales, necesitamos que la
posición $t$ solo pueda atender a posiciones $\{0, 1, \dots, t\}$ —
es decir, atención **causal** (como en GPT).

### 5.2 Scaled Dot-Product Attention

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

donde $M$ es una máscara triangular inferior que asigna $-\infty$ a
posiciones futuras.

### 5.3 Implementación

`ext/transformer_model.py` → `CausalSelfAttention`:

```python
# Máscara causal — triangular inferior
causal = tf.linalg.band_part(
    tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
)
attn_logits += (1.0 - causal) * (-1e9)   # -∞ para posiciones futuras
```

**Multi-head**: Los `d_model=64` se dividen en `n_heads=4` cabezas de
`depth=16` cada una.  Cada cabeza aprende patrones de atención diferentes.

### 5.4 Máscara de Padding

Cuando los episodios son más cortos que `max_seq_len=10`, se rellenan
con ceros.  Una máscara de padding adicional evita que el modelo atienda
a posiciones vacías:

```python
if padding_mask is not None:
    pm = tf.cast(tf.reshape(padding_mask, (batch, 1, 1, -1)), tf.float32)
    attn_logits += (1.0 - pm) * (-1e9)
```

---

## 6. Bloques Transformer (Pre-Norm)

### 6.1 Estilo Pre-Norm (GPT-2)

A diferencia del Transformer original (Vaswani et al., 2017) que aplica
LayerNorm **después** de cada sub-capa (post-norm), `pes_trf` usa el
estilo **pre-norm** (aplicar LayerNorm **antes** de cada sub-capa), como
en GPT-2.  Pre-norm facilita el entrenamiento y mejora la estabilidad de
gradientes.

```text
Entrada x
  │
  ├→ LayerNorm → CausalSelfAttention → Dropout → (+x) = h₁
  │
  └→ LayerNorm → FFN(GELU) → Dropout → (+h₁) = salida
```

### 6.2 Feed-Forward Network

Cada bloque contiene un FFN de dos capas con activación GELU:

$$\text{FFN}(x) = \text{Dense}_{d\_model}(\text{GELU}(\text{Dense}_{d\_ff}(x)))$$

GELU (Gaussian Error Linear Unit) es una activación suave que
permite pequeños gradientes negativos, mejorando el entrenamiento
frente a ReLU.

---

## 7. Funciones de Pérdida

### 7.1 Pérdida Combinada

El modelo se entrena con una pérdida combinada de tres componentes:

$$\mathcal{L} = \mathcal{L}_\text{policy} + c_v \cdot \mathcal{L}_\text{value} - c_e \cdot H(\pi)$$

donde:

- $c_v = 0.5$ (`value_coeff`): peso de la pérdida del Critic.
- $c_e = 0.01$ (`entropy_coeff`): peso del bono de entropía.

### 7.2 Pérdida de Política (REINFORCE con Advantage)

$$\mathcal{L}_\text{policy} = -\frac{1}{|\text{valid}|} \sum_{t} \log \pi_\theta(a_t \mid s_{\le t}) \cdot \hat{A}_t \cdot m_t$$

donde $\hat{A}_t$ es la ventaja estimada por GAE y $m_t$ es la máscara
de padding.

### 7.3 Pérdida del Value Head (MSE)

$$\mathcal{L}_\text{value} = \frac{1}{|\text{valid}|} \sum_{t} \left( V_\theta(s_{\le t}) - R_t \right)^2 \cdot m_t$$

donde $R_t$ es el retorno descontado calculado por GAE.

### 7.4 Bono de Entropía

$$H(\pi) = \frac{1}{|\text{valid}|} \sum_{t} \left[ -\sum_a \pi_\theta(a \mid s_{\le t}) \log \pi_\theta(a \mid s_{\le t}) \right] \cdot m_t$$

### 7.5 Implementación

`ext/train_transformer.py` → `train_step()` (decorada con `@tf.function`):

```python
@tf.function
def train_step(model, optimizer, states, actions, returns, advantages,
               masks, value_coeff, entropy_coeff, max_grad_norm):
    with tf.GradientTape() as tape:
        policy_logits, values = model(states, padding_mask=masks, training=True)

        log_probs_all = tf.nn.log_softmax(policy_logits, axis=-1)
        actions_oh = tf.one_hot(actions, depth=model.n_actions)
        log_probs = tf.reduce_sum(log_probs_all * actions_oh, axis=-1)

        probs_all = tf.nn.softmax(policy_logits, axis=-1)
        entropy = -tf.reduce_sum(probs_all * log_probs_all, axis=-1)

        policy_loss = -tf.reduce_sum(log_probs * advantages * masks) / tf.reduce_sum(masks)
        value_loss = tf.reduce_sum(tf.square(values - returns) * masks) / tf.reduce_sum(masks)
        entropy_mean = tf.reduce_sum(entropy * masks) / tf.reduce_sum(masks)

        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_mean

    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, policy_loss, value_loss, entropy_mean
```

**Notas de implementación**:

- A diferencia de `pes_ac` y `pes_dqn`, esta función **sí** lleva
  `@tf.function` a nivel de módulo, porque el Transformer no se
  re-instancia entre trials de Optuna en el pipeline de entrenamiento.
- Se aplica **gradient clipping** (`tf.clip_by_global_norm`, default 0.5)
  para evitar explosiones de gradientes, especialmente importantes con
  la atención causal.
- La máscara `masks` asegura que las posiciones de padding no contribuyan
  a ninguna de las tres componentes de la pérdida.

---

## 8. Generalised Advantage Estimation (GAE)

### 8.1 Motivación

REINFORCE puro sufre de alta varianza.  GAE (Schulman et al., 2016)
proporciona un estimador de la ventaja con un trade-off continuo entre
sesgo y varianza controlado por $\lambda$:

$$\hat{A}_t^\text{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

donde $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ es el error TD.

- $\lambda = 0$: Advantage de un paso (TD(0)) — bajo varianza, alto sesgo.
- $\lambda = 1$: Monte-Carlo returns — alta varianza, cero sesgo.

### 8.2 Implementación

`ext/train_transformer.py` → `compute_gae()`:

```python
def compute_gae(rewards, values, masks, gamma, lam=0.95):
    B, T = rewards.shape
    advantages = numpy.zeros_like(rewards)
    returns = numpy.zeros_like(rewards)

    for b in range(B):
        gae = 0.0
        for t in reversed(range(T)):
            if masks[b, t] == 0.0:
                continue
            next_val = values[b, t + 1] if (t + 1 < T and masks[b, t + 1] > 0) else 0.0
            delta = rewards[b, t] + gamma * next_val - values[b, t]
            gae = delta + gamma * lam * gae
            advantages[b, t] = gae
            returns[b, t] = gae + values[b, t]

    return advantages, returns
```

**Configuración** por defecto: `gamma=0.98`, `lam=0.95`.

### 8.3 Normalización de Ventajas

Antes de la actualización, las ventajas se normalizan (media 0, std 1)
sobre los timesteps válidos, lo que estabiliza el entrenamiento:

```python
adv_valid = advantages[masks > 0]
advantages = (advantages - adv_valid.mean()) / (adv_valid.std() + 1e-8)
```

---

## 9. Recolección de Trayectorias

### 9.1 Batch Collection

`ext/train_transformer.py` → `collect_batch()`:

En cada iteración de entrenamiento, se recolectan `batch_size=64` episodios
completos usando la política actual (muestreo, no greedy).  Cada episodio
se rellena con ceros hasta `max_seq_len=10` y se marca con una máscara
de padding.

```
Para cada episodio k = 1 … batch_size:
    env.random_sequence()
    state = env.reset()
    agent.reset()

    Mientras no done:
        action, probs, value, log_prob = agent.act(state)
        action = min(action, resources_left)    ← enmascarar infactibles
        state', reward, done = env.step(action)

    Rellenar hasta max_seq_len con ceros
    masks[k] = [1.0]*T + [0.0]*(max_seq_len - T)
```

### 9.2 TrajectoryAgent

`ext/transformer_model.py` → `TransformerAgent`:

Un wrapper que mantiene un buffer de trayectoria dentro del episodio.
En cada `act(state)`:

1. Añade `state` al buffer `_trajectory`.
2. Alimenta la trayectoria completa al modelo.
3. El modelo calcula logits/values en **cada** posición, pero solo se usa
   la del **último** timestep (mediante `get_action_and_value`).

```python
def act(self, state, resources_left=None):
    self._trajectory.append([int(s) for s in state])
    states = tf.constant([self._trajectory], dtype=tf.float32)
    action, probs, value, log_prob = self.model.get_action_and_value(
        states, greedy=self.greedy, resources_left=resources_left,
    )
    return action, probs, value, log_prob
```

---

## 10. Bucle de Entrenamiento Completo

`ext/train_transformer.py` → `main()`:

```
Para cada batch b = 1 … n_batches (1500):
    1. Recolectar batch_size=64 episodios con la política actual
       → states, actions, rewards, masks, log_probs, values

    2. Calcular ventajas y retornos con GAE (γ=0.98, λ=0.95)
    3. Normalizar ventajas (media 0, std 1)
    4. Actualizar modelo con train_step()
       → total_loss = policy_loss + 0.5·value_loss - 0.01·entropy

    Cada 50 batches:
        ─ Imprimir loss, entropía, recompensa media

    Cada 100 batches (eval_every):
        ─ Evaluar con greedy en 64 secuencias fijas
        ─ Si mean_perf > best_perf: guardar pesos

Restaurar mejores pesos
Evaluación final en 64 secuencias
Guardar modelo, rewards, plots, report
```

### 10.1 On-Policy con Batches

| Aspecto | A2C (`pes_ac`) | Transformer (`pes_trf`) |
|---------|----------------|--------------------------|
| Unidad de update | 1 episodio | Batch de 64 episodios |
| Ventaja | TD(0): $r + \gamma V(s') - V(s)$ | GAE($\lambda=0.95$) |
| Total episodios | 100 000 | 96 000 (1500 × 64) |
| Gradient clipping | No | `clip_by_global_norm(0.5)` |
| Normalización de ventajas | No | Sí |

### 10.2 Parámetros de Entrenamiento

| Parámetro | Valor por defecto | Fuente |
|-----------|-------------------|--------|
| `n_batches` | 1 500 | CLI arg o default |
| `batch_size` | 64 | `train_transformer.py` |
| `learning_rate` | 3 × 10⁻⁴ | `train_transformer.py` |
| `gamma` | 0.98 | `train_transformer.py` |
| `lam` (GAE) | 0.95 | `compute_gae()` |
| `value_coeff` | 0.5 | `train_transformer.py` |
| `entropy_coeff` | 0.01 | `train_transformer.py` |
| `max_grad_norm` | 0.5 | `train_transformer.py` |
| `eval_every` | 100 batches | `train_transformer.py` |
| `SEED` | 42 | `config/CONFIG.py` |

---

## 11. Encoding Posicional

### 11.1 Posicional Aprendido (vs. Sinusoidal)

A diferencia del Transformer original (Vaswani et al., 2017) que usa
encodings sinusoidales fijos, `pes_trf` utiliza **embeddings posicionales
aprendidos** (como GPT-2):

```python
self.pos_embedding = self.add_weight(
    name='pos_embedding',
    shape=(1, max_seq_len, d_model),    # (1, 10, 64)
    initializer='glorot_uniform',
    trainable=True,
)
```

**Justificación**: Con secuencias cortas (máximo 10 timesteps), no hay
beneficio en la extrapolación de los encodings sinusoidales.  Los
embeddings aprendidos son más flexibles y se optimizan directamente
para el dominio.

---

## 12. Enmascaramiento de Acciones

### 12.1 Acciones Infactibles

Al seleccionar una acción, si el agente tiene `resources_left < 10`,
las acciones que exceden los recursos disponibles se enmascaran con
logits de $-10^9$ antes del softmax:

```python
if resources_left is not None:
    rl = int(resources_left)
    if rl < self.n_actions - 1:
        mask_vals = tf.concat([
            tf.zeros(rl + 1, dtype=tf.float32),
            tf.fill([self.n_actions - rl - 1], -1e9),
        ], axis=0)
        logits = logits + mask_vals
```

Esto asegura que el softmax asigne probabilidad ~0 a acciones imposibles.

---

## 13. Meta-Cognición y Confianza

### 13.1 Confianza Basada en Entropía

Al igual que en `pes_ac`, la confianza se calcula a partir de la entropía
de la distribución de probabilidad del actor:

$$H(\pi_\theta(\cdot \mid s_{\le t})) = -\sum_{a} \pi_\theta(a \mid s_{\le t}) \log \pi_\theta(a \mid s_{\le t})$$

$$\text{confidence} = \frac{H - H_\text{max}}{H_\text{min} - H_\text{max}}$$

### 13.2 Implementación

La distribución softmax del policy head se pasa a
`rl_agent_meta_cognitive()` en `pandemic.py`, que calcula la entropía
y normaliza entre el mínimo (distribución peaked) y el máximo
(distribución uniforme):

```python
_, conf, _, _ = rl_agent_meta_cognitive(probs, state[0], 10000)
```

### 13.3 Ventaja sobre DQN

Como en `pes_ac`, la salida del policy head ya es una distribución de
probabilidad válida (softmax), por lo que la entropía tiene un significado
teórico preciso — a diferencia de `pes_dqn` donde los Q-values deben
normalizarse heurísticamente.

---

## 14. Inferencia en Tiempo de Experimento

### 14.1 Carga del Modelo

`src/pygameMediator.py` → `provide_rl_agent_response()`:

```python
model = PandemicTransformer.from_pretrained(INPUTS_PATH)
```

El modelo se carga una vez (singleton) desde `transformer_config.json` +
`transformer_weights.weights.h5`.

### 14.2 Gestión de Trayectoria

El mediador mantiene un buffer `_trajectory_buffer` que se resetea al
inicio de cada secuencia:

```python
_trajectory_buffer.append([resources_idx, city_idx, sever_idx])
states = tf.constant([_trajectory_buffer], dtype=tf.float32)
_, probs, _, _ = model.get_action_and_value(
    states, greedy=True, resources_left=resources_idx
)
```

A diferencia de los modelos feedforward, el Transformer **necesita** la
trayectoria completa del episodio para producir una decisión informada.

### 14.3 Selección de Acción

Se usa **argmax** (greedy) sobre los logits enmascarados en inferencia:

```python
action = argmax(logits)   # tras enmascarar acciones infactibles
```

---

## 15. Persistencia del Modelo

### 15.1 Formato de Guardado

A diferencia de `pes_dqn` y `pes_ac` que usan `.keras`, el Transformer
usa un formato custom con dos archivos:

| Archivo | Contenido |
|---------|-----------|
| `transformer_config.json` | Hiperparámetros de arquitectura (d_model, n_heads, etc.) |
| `transformer_weights.weights.h5` | Pesos del modelo (Keras HDF5) |

### 15.2 Serialización

```python
# Guardar
model.save_pretrained(directory)

# Cargar
model = PandemicTransformer.from_pretrained(directory)
```

El método `from_pretrained` reconstruye la arquitectura desde el JSON,
ejecuta un forward pass dummy para inicializar las variables, y luego
carga los pesos.

---

## 16. Optimización Bayesiana de Hiperparámetros Q-Learning

**Nota:** El módulo `ext/optimize_tr.py` optimiza hiperparámetros de
**Q-Learning tabular** (heredado de `pes_dql`), **no** del Transformer.
Se conserva como baseline de comparación.

### 16.1 Espacio de Búsqueda (Q-Learning)

| Parámetro | Rango | Tipo |
|-----------|-------|------|
| `learning_rate` | $[0.2,\; 0.4]$ | log-uniforme |
| `discount_factor` | $[0.80,\; 0.99]$ | uniforme |
| `epsilon_initial` | $[0.4,\; 1.0]$ | uniforme |
| `epsilon_min` | $[0.05,\; 0.1]$ | uniforme |
| `num_episodes` | $[800\text{K},\; 1.2\text{M}]$ | entero (paso 10K) |
| `warmup_ratio` | $[0.01,\; 0.10]$ | uniforme |
| `target_ratio` | $[0.50,\; 0.80]$ | uniforme |
| `penalty_coeff` | $[0.001,\; 0.5]$ | log-uniforme |

---

## 17. Comparación con Algoritmos Anteriores

| Componente | `pes_base` (Q-tabular) | `pes_dqn` (DQN) | `pes_ac` (A2C) | `pes_trf` (Transformer) |
|------------|-------------------|-----------------|----------------|--------------------------|
| Modelo | `numpy.ndarray` | MLP 5 131 params | Actor 5 131 + Critic 4 481 | Transformer ~77 000 params |
| Contexto | Sin contexto | Sin contexto | Sin contexto | Trayectoria completa |
| Update | TD tabular | Huber + replay | Policy gradient + MSE | REINFORCE + MSE + GAE |
| Datos | Un paso → un update | Replay buffer | Batch de 1 episodio | Batch de 64 episodios |
| Política | Implícita ($\arg\max Q$) | Implícita ($\arg\max Q$) | Explícita ($\pi_\theta$) | Explícita ($\pi_\theta$) |
| Confianza | Entropía de Q (heurística) | Entropía de Q (heurística) | Entropía de $\pi$ (teórica) | Entropía de $\pi$ (teórica) |
| Atención | — | — | — | Causal multi-head |
| Gradient clip | — | — | — | Global norm (0.5) |
| Episodios típicos | 900 000 | 100 000 | 100 000 | 96 000 (1 500 × 64) |

---

## 18. Estructura de Archivos

```
pes_trf/
├── __init__.py                 # Exporta constantes; configura CPU, OMP_NUM_THREADS
├── __main__.py                 # Valida transformer_config.json antes de ejecutar
├── config/CONFIG.py            # Parámetros del experimento (no arquitectura)
├── ext/
│   ├── transformer_model.py    # CausalSelfAttention, TransformerBlock,
│   │                           #   PandemicTransformer, TransformerAgent
│   ├── train_transformer.py    # Pipeline: collect_batch, compute_gae, train_step
│   ├── pandemic.py             # Entorno Gym + QLearning (baseline) + run_experiment
│   ├── optimize_tr.py          # Bayesian optimisation (Q-Learning, no Transformer)
│   └── tools.py                # Entropía, gráficas
├── src/
│   ├── pygameMediator.py       # Carga modelo, gestión de trayectoria en experimento
│   ├── exp_utils.py            # Severidades, secuencias
│   ├── log_utils.py            # Logging dual
│   ├── result_formatter.py     # Gráficas matplotlib
│   └── terminal_utils.py       # UI de consola Rich
└── doc/
    ├── explained_transformer.md        # ← este documento
    └── how_to_train_and_test.md        # Guía práctica de entrenamiento y pruebas
```

---

## 19. Formulario Resumen de Ecuaciones

| Concepto | Ecuación |
|----------|----------|
| Atención causal | $\text{Attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k} + M)\, V$ |
| Pre-norm residual | $h = x + \text{SubLayer}(\text{LN}(x))$ |
| Pérdida de política | $\mathcal{L}_\pi = -\frac{1}{N}\sum_t \log \pi(a_t \mid s_{\le t}) \hat{A}_t \cdot m_t$ |
| Pérdida de valor | $\mathcal{L}_V = \frac{1}{N}\sum_t (V(s_{\le t}) - R_t)^2 \cdot m_t$ |
| Entropía | $H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ |
| Pérdida total | $\mathcal{L} = \mathcal{L}_\pi + c_v \mathcal{L}_V - c_e H(\pi)$ |
| GAE | $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$ |
| TD error | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| Decaimiento posicional | Aprendido: $\text{pos\_emb} \in \mathbb{R}^{1 \times 10 \times 64}$ |
| Normalización | $\hat{s} = (r/30, t/10, v/9)$ |

---

## 20. Referencias

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
2. Chen, L. et al. (2021). *Decision Transformer: Reinforcement Learning
   via Sequence Modeling*. NeurIPS 2021.
3. Schulman, J. et al. (2016). *High-Dimensional Continuous Control Using
   Generalized Advantage Estimation*. ICLR 2016.
4. Williams, R. J. (1992). *Simple statistical gradient-following
   algorithms for connectionist reinforcement learning*. Machine Learning,
   8(3-4), 229–256.
5. Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic
   Optimization*. ICLR 2015.
6. Akiba, T. et al. (2019). *Optuna: A next-generation hyperparameter
   optimization framework*. KDD '19.
7. Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units
   (GELUs)*. arXiv:1606.08415.
