# Mapeo Teoría DQN ↔ Implementación en pes_dqn

## 1. Introducción

Este documento conecta la **teoría de Deep Q-Networks (DQN)** con su
**implementación concreta** en el paquete `pes_dqn`.  Para cada concepto
teórico se indica la variable, función o línea de código correspondiente.

`pes_dqn` reemplaza la tabla Q tabular de `pes` / `pes_base_line` por una
**red neuronal** que aproxima la función $Q(s, a)$.  Conserva el mismo
entorno Gym (`Pandemic`), el mismo espacio de estados y acciones, y la
misma integración con la UI de Pygame.

---

## 2. Componentes del MDP (heredados de `pes`)

El MDP subyacente es idéntico al del paquete base:

| Componente | Símbolo | Implementación |
|------------|---------|----------------|
| Estados | $S$ | `(resources_left, trial_no, severity)` — 3 dimensiones discretas |
| Acciones | $A$ | `{0, 1, 2, …, 10}` — recursos a asignar |
| Transiciones | $P(s' \mid s, a)$ | Determinísticas: `env.step(action)` en `pandemic.py` |
| Recompensas | $R(s, a)$ | $-\sum_{i} \text{severities}_i$ |
| Factor de descuento | $\gamma$ | `discount` (por defecto 0.865 en `train_drl.py`) |

**Cardinalidad del espacio de estados** (con `MAX_SEVERITY = 9`):

$$|S| = 31 \times 11 \times 10 = 3{,}410 \text{ estados}$$

---

## 3. ¿Por qué DQN en lugar de Q-tabular?

| Aspecto | Q-tabular (`pes`) | DQN (`pes_dqn`) |
|---------|-------------------|-----------------|
| Representación de $Q$ | Tabla $|S| \times |A|$ (37 510 celdas) | Red neuronal (~5 131 parámetros) |
| Generalización | Ninguna — cada celda se actualiza de forma independiente | La red interpola entre estados similares |
| Escalabilidad | Crece exponencialmente con dimensiones | Crece linealmente con parámetros de red |
| Muestra de datos | On-policy (un paso, un update) | Off-policy con experience replay |
| Estabilidad | Convergencia garantizada (tabular) | Se estabiliza mediante red objetivo (target network) |

---

## 4. Arquitectura de la Red Neuronal

### 4.1 Definición

Implementada en `ext/dqn_model.py` → `build_q_network()`:

```
Input  (3)  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  Dense(11, linear)
               hidden_units[0]     hidden_units[1]     action_dim
```

```python
model = tf.keras.Sequential(name="DQN")
model.add(tf.keras.layers.Input(shape=(state_dim,)))
for idx, units in enumerate(hidden_units):
    model.add(tf.keras.layers.Dense(
        units, activation="relu", name=f"hidden_{idx}"))
model.add(tf.keras.layers.Dense(
    action_dim, activation="linear", name="q_values"))
```

**Parámetros** (con `DQN_HIDDEN_UNITS = [64, 64]`):

| Capa | Forma | Parámetros |
|------|-------|------------|
| `hidden_0` | 3 → 64 | $3 \times 64 + 64 = 256$ |
| `hidden_1` | 64 → 64 | $64 \times 64 + 64 = 4{,}160$ |
| `q_values` | 64 → 11 | $64 \times 11 + 11 = 715$ |
| **Total** | | **5 131** |

### 4.2 Normalización del Estado

Antes de alimentar la red, el estado entero se escala a $[0, 1]^3$:

$$\hat{s} = \left(\frac{r}{30},\; \frac{t}{10},\; \frac{v}{9}\right)$$

Implementada en `ext/dqn_model.py` → `normalize_state()`:

```python
numpy.array([
    state[0] / max(max_resources, 1),   # resources / 30
    state[1] / max(max_trials, 1),      # trial / 10
    state[2] / max(max_severity, 1),    # severity / 9
], dtype=numpy.float32)
```

---

## 5. Experience Replay

### 5.1 Motivación Teórica

En Q-Learning tabular, cada transición se usa una vez y se descarta.  Esto
genera alta correlación temporal entre muestras consecutivas, perjudicando
el entrenamiento de redes neuronales.

Experience Replay almacena transiciones $(s, a, r, s', \text{done})$ en un
buffer circular y las muestrea **uniformemente al azar**, rompiendo la
correlación y permitiendo reutilizar datos.

### 5.2 Implementación

`ext/dqn_model.py` → clase `ReplayBuffer`:

```python
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[...] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (numpy.array(states, ...), ...)
```

**Configuración** (`config/CONFIG.py`):

| Constante | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `DQN_REPLAY_BUFFER_SIZE` | 50 000 | Capacidad máxima del buffer |
| `DQN_BATCH_SIZE` | 32 | Tamaño de mini-batch |

---

## 6. Red Objetivo (Target Network)

### 6.1 Problema de la Inestabilidad

Si se usa la misma red para calcular los targets de TD y para actualizar
los pesos, la función objetivo se mueve con cada gradiente — feedback
positivo que puede divergir.

### 6.2 Solución: Hard Copy Periódico

Se mantiene una **copia congelada** (*target network*) cuyos pesos se
reemplazan completamente con los de la red online cada
`DQN_TARGET_SYNC_FREQ` pasos de gradiente:

```python
def sync_target_network(online_model, target_model):
    target_model.set_weights(online_model.get_weights())
```

El target de TD se calcula con la red objetivo:

$$y_i = r_i + \gamma \cdot \max_{a'} Q_{\theta^-}(s'_i, a') \cdot (1 - d_i)$$

donde $\theta^-$ son los pesos congelados.

**Configuración**:

| Constante | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `DQN_TARGET_SYNC_FREQ` | 1 000 | Gradient steps entre sincronizaciones |

---

## 7. Función de Pérdida (Huber Loss)

### 7.1 Definición

La **Huber loss** (smooth L1) combina MSE para errores pequeños con MAE
para errores grandes, ofreciendo robustez ante outliers:

$$L_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \le \delta \\
\delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{en otro caso}
\end{cases}$$

### 7.2 Implementación

`ext/dqn_model.py` → `train_step()` (decorada con `@tf.function`):

```python
@tf.function
def train_step(online_model, target_model, optimizer,
               states, actions, rewards, next_states, dones, discount):
    # TD targets
    next_q = target_model(next_states, training=False)
    max_next_q = tf.reduce_max(next_q, axis=1)
    targets = rewards + discount * max_next_q * (1.0 - dones)

    with tf.GradientTape() as tape:
        q_all = online_model(states, training=True)
        action_mask = tf.one_hot(actions, depth=tf.shape(q_all)[1])
        q_selected = tf.reduce_sum(q_all * action_mask, axis=1)
        loss = tf.keras.losses.huber(targets, q_selected)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, online_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, online_model.trainable_variables))
    return loss
```

**Notas de implementación**:
- `@tf.function` compila el grafo de TensorFlow para ejecución eficiente.
- `tf.one_hot` + `tf.reduce_sum` selecciona los Q-values correspondientes
  a las acciones tomadas, sin indexación que rompa la diferenciabilidad.

---

## 8. Exploración ε-Greedy con Decaimiento Lineal

### 8.1 Política

$$a_t =
\begin{cases}
\arg\max_a Q_\theta(s_t, a) & \text{con probabilidad } 1 - \varepsilon \\
\text{acción aleatoria} & \text{con probabilidad } \varepsilon
\end{cases}$$

### 8.2 Decaimiento

$$\varepsilon_{i+1} = \max(\varepsilon_{\min},\; \varepsilon_i - \Delta)$$

donde $\Delta = (\varepsilon_0 - \varepsilon_{\min}) / N_{\text{episodes}}$.

Implementado en `ext/pandemic.py` → `DQNTraining()`:

```python
reduction = (epsilon - min_eps) / episodes

# Al final de cada episodio:
if epsilon > min_eps:
    epsilon -= reduction
```

---

## 9. Bucle de Entrenamiento Completo

`ext/pandemic.py` → `DQNTraining()`:

```
Para cada episodio i = 1 … N:
    env.random_sequence()          ← secuencia aleatoria
    state = env.reset()

    Mientras no done:
        1. Normalizar estado:     s_norm = normalize_state(state, 30, 10, 9)
        2. Selección ε-greedy:    action = argmax Q_θ(s_norm) ó random
        3. Meta-cognición:        confidence = rl_agent_meta_cognitive(q_vals, ...)
        4. Paso del entorno:      s', r, done = env.step(action)
        5. Almacenar transición:  buffer.push(s_norm, action, r, s'_norm, done)
        6. Si global_step % train_freq == 0 y |buffer| ≥ batch_size:
             ─ Samplear mini-batch del buffer
             ─ train_step(online, target, optimizer, batch)
             ─ Si train_steps % target_sync_freq == 0:
                   sync_target_network(online, target)
        7. Decaer ε

    Cada 10 000 episodios:
        ─ Imprimir recompensa promedio

    Retornar (ave_reward_list, online_model, conf_list)
```

**Parámetros configurables** (`config/CONFIG.py`):

| Constante | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `DQN_HIDDEN_UNITS` | `[64, 64]` | Capas ocultas de la red |
| `DQN_BATCH_SIZE` | 32 | Mini-batch de replay |
| `DQN_REPLAY_BUFFER_SIZE` | 50 000 | Capacidad del buffer |
| `DQN_TARGET_SYNC_FREQ` | 1 000 | Pasos entre sincronización de target |
| `DQN_LEARNING_RATE` | 1e-3 | Learning rate de Adam |
| `DQN_TRAIN_FREQ` | 4 | Pasos entre gradient updates |
| `DQN_MODEL_FILE` | `dqn_model.keras` | Archivo del modelo guardado |

---

## 10. Optimización Bayesiana de Hiperparámetros

`ext/optimize_drl.py` utiliza **Optuna** (TPE sampler) para buscar
hiperparámetros óptimos del DQN.

### 10.1 Espacio de Búsqueda

| Parámetro | Rango | Tipo |
|-----------|-------|------|
| `learning_rate` | $[10^{-4},\; 10^{-2}]$ | log-uniforme |
| `discount_factor` | $[0.80,\; 0.99]$ | uniforme |
| `epsilon_initial` | $[0.4,\; 1.0]$ | uniforme |
| `epsilon_min` | $[0.01,\; 0.10]$ | uniforme |
| `num_episodes` | $[50\,000,\; 200\,000]$ | entero (paso 25k) |
| `hidden_dim` | $[32,\; 128]$ | entero (paso 32) |
| `n_hidden_layers` | $\{1, 2, 3\}$ | categórico |
| `batch_size` | $\{16, 32, 64\}$ | categórico |
| `replay_buffer_size` | $[20\,000,\; 100\,000]$ | entero (paso 10k) |
| `target_sync_freq` | $[500,\; 2\,000]$ | entero (paso 500) |

### 10.2 Función Objetivo

Se entrena un agente DQN con los hiperparámetros sugeridos y se evalúa
en las **64 secuencias fijas** (las mismas que usa `__main__.py`).  El score
reportado a Optuna es el **rendimiento normalizado medio** (a maximizar),
calculado con `calculate_normalised_final_severity_performance_metric()`.
Se aplica enmascaramiento de acciones infactibles (`actions > resources_left`)
para que la métrica coincida con el comportamiento del agente en el experimento.

### 10.3 Artefactos

Los mejores pesos del modelo se almacenan durante la búsqueda y al
finalizar se reconstruye la red y se guarda como `dqn_model.keras`.

---

## 11. Inferencia en Tiempo de Experimento

### 11.1 Carga del Modelo

`src/pygameMediator.py` → `provide_rl_agent_response()`:

```python
model = tf.keras.models.load_model(model_path)
```

### 11.2 Selección de Acción

```python
s_norm = normalize_state([resources_left, city_trial_no, severity], 30, 10, 9)
q_values = model(s_norm[numpy.newaxis, :], training=False)[0].numpy()
action = int(numpy.argmax(q_values))
```

El vector de Q-values (11 elementos) se pasa a `rl_agent_meta_cognitive()`
para calcular la confianza por entropía, exactamente igual que en el
paquete tabular.

---

## 12. Comparación con Algoritmo Original (pes)

| Componente | `pes` (Q-tabular) | `pes_dqn` (DQN) |
|------------|-------------------|-----------------|
| Modelo | `numpy.ndarray` (q.npy) | `tf.keras.Model` (.keras) |
| Update | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max Q(s') - Q(s,a)]$ | Gradient descent con Huber loss |
| Datos | Un paso → un update | Replay buffer → mini-batch |
| Estabilidad | Convergencia tabular directa | Target network + Huber loss |
| Episodios típicos | 900 000 | 100 000 |
| Tiempo (CPU) | ~30 s | ~30 min – 1 h |

---

## 13. Estructura de Archivos

```
pes_dqn/
├── __init__.py              # Exporta constantes DQN al nivel de paquete
├── __main__.py              # Valida .keras antes de ejecutar
├── config/CONFIG.py         # 7 constantes DQN_*
├── ext/
│   ├── dqn_model.py         # ReplayBuffer, build_q_network, normalize_state,
│   │                        #   train_step, sync_target_network
│   ├── pandemic.py          # Entorno Gym + DQNTraining()
│   ├── train_drl.py         # Pipeline de entrenamiento autónomo
│   ├── optimize_drl.py      # Búsqueda Bayesiana con Optuna
│   └── tools.py             # Entropía, gráficas (sin cambios)
├── src/
│   ├── pygameMediator.py    # Carga .keras, forward pass en experimento
│   ├── exp_utils.py         # Severidades, secuencias
│   ├── log_utils.py         # Logging dual
│   ├── result_formatter.py  # Gráficas matplotlib
│   └── terminal_utils.py    # UI de consola Rich
└── doc/
    ├── explained_dqn.md             # ← este documento
    └── how_to_train_and_test.md     # Guía práctica de entrenamiento y pruebas
```

---

## 14. Referencias

1. Mnih, V. et al. (2015). *Human-level control through deep reinforcement
   learning*. Nature, 518(7540), 529–533.
2. Lin, L.-J. (1992). *Self-improving reactive agents based on
   reinforcement learning, planning and teaching*. Machine Learning,
   8(3-4), 293–321.
3. Akiba, T. et al. (2019). *Optuna: A next-generation hyperparameter
   optimization framework*. KDD '19.
