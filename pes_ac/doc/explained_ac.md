# Mapeo Teoría Actor-Critic (A2C) ↔ Implementación en pes_ac

## 1. Introducción

Este documento conecta la **teoría de Advantage Actor-Critic (A2C)** con su
**implementación concreta** en el paquete `pes_ac`.  Para cada
concepto teórico se indica la variable, función o línea de código
correspondiente.

`pes_ac` reemplaza la red DQN de `pes_dqn` (y la tabla Q tabular
de `pes` / `pes_bline`) por **dos redes neuronales separadas**:

- un **Actor** $\pi_\theta(a \mid s)$ que produce directamente una
  distribución de probabilidad sobre las acciones, y
- un **Critic** $V_\phi(s)$ que estima el valor del estado actual.

Conserva el mismo entorno Gym (`Pandemic`), el mismo espacio de estados y
acciones, y la misma integración con la UI de Pygame.

---

## 2. Componentes del MDP (heredados de `pes`)

El MDP subyacente es idéntico al del paquete base:

| Componente | Símbolo | Implementación |
|------------|---------|----------------|
| Estados | $S$ | `(resources_left, trial_no, severity)` — 3 dimensiones discretas |
| Acciones | $A$ | $\{0, 1, 2, \dots, 10\}$ — recursos a asignar |
| Transiciones | $P(s' \mid s, a)$ | Determinísticas: `env.step(action)` en `pandemic.py` |
| Recompensas | $R(s, a)$ | $-\sum_{i} \text{severities}_i$ |
| Factor de descuento | $\gamma$ | `AC_DISCOUNT` (por defecto 0.99 en `CONFIG.py`) |

**Cardinalidad del espacio de estados** (con `MAX_SEVERITY = 9`):

$$|S| = 31 \times 11 \times 10 = 3{,}410 \text{ estados}$$

---

## 3. ¿Por qué Actor-Critic en lugar de DQN?

### 3.1 Filosofía del Enfoque

DQN aprende una **función de valor-acción** $Q(s, a)$ y deriva la política
de forma implícita como $\pi(s) = \arg\max_a Q(s, a)$.  Actor-Critic
aprende **explícitamente** una política $\pi_\theta(a \mid s)$ junto con
una función de valor de estado $V_\phi(s)$, combinando las ventajas de
métodos basados en política y basados en valor.

### 3.2 Comparación

| Aspecto | DQN (`pes_dqn`) | A2C (`pes_ac`) |
|---------|-----------------|--------------------------|
| Representación de la política | Implícita: $\arg\max_a Q(s, a)$ | Explícita: $\pi_\theta(a \mid s)$ |
| Salida de la red | Q-values para todas las acciones | Actor: probabilidades; Critic: valor escalar |
| Tipo de aprendizaje | Off-policy (replay buffer) | On-policy (sin replay buffer) |
| Exploración | ε-greedy sobre Q-values | Muestreo de $\pi_\theta$ + ε-greedy overlay |
| Estabilidad | Target network | Ventaja centrada (Advantage) |
| Confianza meta-cognitiva | Heurística (entropía de Q-values) | Teóricamente fundamentada (entropía de $\pi$) |
| Nº de redes | 2 (online + target) | 2 (Actor + Critic) |

### 3.3 Ventaja Teórica de la Confianza en A2C

En `pes_dqn`, la "confianza" del agente se calcula como la entropía de
los Q-values normalizados — una heurística, ya que los Q-values **no son
una distribución de probabilidad**.

En `pes_ac`, la salida del Actor $\pi_\theta(a \mid s)$ **es** una
distribución de probabilidad (softmax), por lo que la entropía tiene un
significado teórico preciso:

$$H(\pi_\theta(\cdot \mid s)) = -\sum_{a} \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$$

- $H$ bajo → el agente está **seguro** (la distribución se concentra en
  pocas acciones).
- $H$ alto → el agente está **inseguro** (la distribución es casi uniforme).

---

## 4. Arquitectura de las Redes Neuronales

### 4.1 Actor (Red de Política)

Implementada en `ext/ac_model.py` → `build_actor()`:

```
Input  (3)  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  Dense(11, softmax)
               hidden_units[0]     hidden_units[1]     action_dim
```

```python
model = tf.keras.Sequential(name="Actor")
model.add(tf.keras.layers.Input(shape=(state_dim,)))
for idx, units in enumerate(hidden_units):
    model.add(tf.keras.layers.Dense(
        units, activation="relu", name=f"actor_hidden_{idx}"))
model.add(tf.keras.layers.Dense(
    action_dim, activation="softmax", name="policy"))
```

La capa de salida **softmax** garantiza que $\sum_a \pi_\theta(a \mid s) = 1$
y $\pi_\theta(a \mid s) \ge 0$, produciendo una distribución de
probabilidad válida.

**Parámetros** (con `AC_ACTOR_HIDDEN_UNITS = [64, 64]`):

| Capa | Forma | Parámetros |
|------|-------|------------|
| `actor_hidden_0` | 3 → 64 | $3 \times 64 + 64 = 256$ |
| `actor_hidden_1` | 64 → 64 | $64 \times 64 + 64 = 4{,}160$ |
| `policy` | 64 → 11 | $64 \times 11 + 11 = 715$ |
| **Total Actor** | | **5 131** |

### 4.2 Critic (Red de Valor de Estado)

Implementada en `ext/ac_model.py` → `build_critic()`:

```
Input  (3)  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  Dense(1, linear)
               hidden_units[0]     hidden_units[1]     scalar value
```

```python
model = tf.keras.Sequential(name="Critic")
model.add(tf.keras.layers.Input(shape=(state_dim,)))
for idx, units in enumerate(hidden_units):
    model.add(tf.keras.layers.Dense(
        units, activation="relu", name=f"critic_hidden_{idx}"))
model.add(tf.keras.layers.Dense(1, activation="linear", name="value"))
```

La capa de salida **lineal** permite que $V_\phi(s)$ tome cualquier valor
real (las recompensas del entorno son negativas).

**Parámetros** (con `AC_CRITIC_HIDDEN_UNITS = [64, 64]`):

| Capa | Forma | Parámetros |
|------|-------|------------|
| `critic_hidden_0` | 3 → 64 | $3 \times 64 + 64 = 256$ |
| `critic_hidden_1` | 64 → 64 | $64 \times 64 + 64 = 4{,}160$ |
| `value` | 64 → 1 | $64 \times 1 + 1 = 65$ |
| **Total Critic** | | **4 481** |

### 4.3 Total de Parámetros

$$|Θ_\text{total}| = |Θ_\text{Actor}| + |Θ_\text{Critic}| = 5{,}131 + 4{,}481 = 9{,}612$$

Comparado con DQN (5 131 parámetros), A2C usa ~1.87× más parámetros, pero
la mitad (Critic) solo se necesita durante entrenamiento; en inferencia
solo se usa el Actor.

### 4.4 Normalización del Estado

Antes de alimentar las redes, el estado entero se escala a $[0, 1]^3$:

$$\hat{s} = \left(\frac{r}{30},\; \frac{t}{10},\; \frac{v}{9}\right)$$

Implementada en `ext/ac_model.py` → `normalize_state()`:

```python
numpy.array([
    state[0] / max(max_resources, 1),   # resources / 30
    state[1] / max(max_trials, 1),      # trial / 10
    state[2] / max(max_severity, 1),    # severity / 9
], dtype=numpy.float32)
```

---

## 5. Fundamento Teórico: Policy Gradient y Advantage

### 5.1 El Teorema del Gradiente de Política

El objetivo del Actor es maximizar el retorno esperado:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

El **Teorema del Gradiente de Política** (Sutton et al., 1999) establece
que:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t \right]$$

donde $\Psi_t$ es una señal de refuerzo genérica.  En REINFORCE,
$\Psi_t = G_t$ (retorno desde $t$), lo que produce alta varianza.

### 5.2 La Función de Ventaja (Advantage)

Actor-Critic reduce la varianza reemplazando $G_t$ por la **ventaja**
(*advantage*):

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

La ventaja mide cuánto **mejor (o peor)** fue la acción $a_t$ respecto al
valor promedio del estado $s_t$.  Usando la aproximación TD(0):

$$\hat{A}(s_t, a_t) = r_t + \gamma V_\phi(s_{t+1}) \cdot (1 - d_t) - V_\phi(s_t)$$

donde $d_t \in \{0, 1\}$ indica si el episodio terminó.

Esta señal tiene **media cero** en expectativa, lo que reduce drásticamente
la varianza del gradiente respecto a REINFORCE.

### 5.3 Intuición

- Si $A > 0$: la acción fue **mejor** que el promedio → aumentar
  $\pi_\theta(a_t \mid s_t)$.
- Si $A < 0$: la acción fue **peor** que el promedio → disminuir
  $\pi_\theta(a_t \mid s_t)$.
- Si $A \approx 0$: la acción fue **típica** → cambio mínimo.

---

## 6. Funciones de Pérdida

### 6.1 Pérdida del Critic (MSE sobre TD Target)

El Critic se entrena para minimizar el error entre su predicción y el
*target* de diferencia temporal:

$$y_t = r_t + \gamma \cdot V_\phi(s_{t+1}) \cdot (1 - d_t)$$

$$\mathcal{L}_\text{Critic} = \frac{1}{N} \sum_{t=1}^{N} \left( V_\phi(s_t) - y_t \right)^2$$

**Nota:** $y_t$ se calcula con `tf.stop_gradient()` para que los gradientes
del Critic no fluyan hacia los targets, asegurando estabilidad.

### 6.2 Pérdida del Actor (Policy Gradient + Entropía)

El Actor se actualiza siguiendo el gradiente de política con ventaja,
**más** un bono de entropía que incentiva la exploración:

$$\mathcal{L}_\text{Actor} = -\frac{1}{N} \sum_{t=1}^{N} \left[ \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right] - c_\text{ent} \cdot H(\pi_\theta)$$

donde:

- $\log \pi_\theta(a_t \mid s_t)$: log-probabilidad de la acción tomada.
- $\hat{A}_t = y_t - V_\phi(s_t)$: ventaja estimada (detached del Actor).
- $c_\text{ent}$: coeficiente de entropía (`AC_ENTROPY_COEFF`, default 0.01).
- $H(\pi_\theta) = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$:
  entropía de la política.

El signo negativo delante del primer término convierte el **ascenso** de
gradiente de política en una **minimización** compatible con
`optimizer.apply_gradients()`.

### 6.3 Implementación

`ext/ac_model.py` → `train_step_actor_critic()`:

```python
def train_step_actor_critic(actor, critic, actor_optimizer, critic_optimizer,
                            states, actions, rewards, next_states, dones,
                            discount, entropy_coeff):
    # --- Critic update ---
    next_values = tf.squeeze(critic(next_states, training=False), axis=1)
    td_targets = rewards + discount * next_values * (1.0 - dones)
    td_targets = tf.stop_gradient(td_targets)

    with tf.GradientTape() as critic_tape:
        values = tf.squeeze(critic(states, training=True), axis=1)
        critic_loss = tf.reduce_mean(tf.square(values - td_targets))

    critic_grads = critic_tape.gradient(
        critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grads, critic.trainable_variables))

    # --- Actor update ---
    advantages = tf.stop_gradient(td_targets - values)

    with tf.GradientTape() as actor_tape:
        probs = actor(states, training=True)
        action_mask = tf.one_hot(actions, depth=tf.shape(probs)[1])
        chosen_probs = tf.reduce_sum(probs * action_mask, axis=1)
        log_probs = tf.math.log(chosen_probs + 1e-8)
        entropy = -tf.reduce_sum(
            probs * tf.math.log(probs + 1e-8), axis=1)
        actor_loss = -tf.reduce_mean(
            log_probs * advantages + entropy_coeff * entropy)

    actor_grads = actor_tape.gradient(
        actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grads, actor.trainable_variables))

    mean_entropy = tf.reduce_mean(entropy)
    return actor_loss, critic_loss, mean_entropy
```

**Notas de implementación**:

- `train_step_actor_critic` **no** lleva `@tf.function` a nivel de módulo
  porque la optimización bayesiana (Optuna) ejecuta múltiples trials con
  diferentes instancias de modelo y optimizador.  `@tf.function` prohíbe
  crear nuevas `tf.Variable` dentro de un grafo ya trazado, lo que
  provocaría un error al iniciar el segundo trial.
- En su lugar, `A2CTraining()` en `pandemic.py` envuelve la función con
  `compiled_train_step = tf.function(train_step_actor_critic)` de forma
  **local** a cada trial, asegurando que cada grafo sea independiente.
- Los optimizadores se pre-construyen con `optimizer.build()` antes del
  loop de entrenamiento para que sus `tf.Variable` internas se creen
  fuera del `@tf.function`.
- `discount` y `entropy_coeff` se pasan como `tf.constant` (no como
  `float` de Python) para evitar re-tracing cuando sus valores cambian
  entre trials de Optuna.
- `tf.stop_gradient(td_targets)` evita que los gradientes del Critic
  fluyan hacia los targets (estabilidad).
- `tf.stop_gradient(td_targets - values)` evita que el gradiente del Actor
  fluya hacia el Critic.
- `+ 1e-8` previene $\log(0)$ en estados donde la política es casi
  determinística.

---

## 7. Exploración: ε-Greedy Overlay + Muestreo de Política

### 7.1 Política de Exploración

A diferencia de DQN (que usa ε-greedy puro sobre Q-values), A2C tiene
exploración **intrínseca** dado que muestrea de $\pi_\theta$.  Sin embargo,
para asegurar exploración suficiente al inicio del entrenamiento, se añade
un **overlay ε-greedy**:

$$a_t =
\begin{cases}
a \sim \pi_\theta(\cdot \mid s_t) & \text{con probabilidad } 1 - \varepsilon \\
\text{acción aleatoria factible} & \text{con probabilidad } \varepsilon
\end{cases}$$

### 7.2 Decaimiento Lineal

$$\varepsilon_{i+1} = \max(\varepsilon_{\min},\; \varepsilon_i - \Delta)$$

donde $\Delta = (\varepsilon_0 - \varepsilon_{\min}) / N_{\text{episodes}}$.

Implementado en `ext/pandemic.py` → `A2CTraining()`:

```python
reduction = (epsilon - min_eps) / episodes

# Al final de cada episodio:
if epsilon > min_eps:
    epsilon -= reduction
```

### 7.3 Bono de Entropía

Adicionalmente, el término de entropía en $\mathcal{L}_\text{Actor}$ actúa
como un **regularizador de exploración** a nivel de gradiente: si
$c_\text{ent} > 0$, el optimizador penaliza distribuciones demasiado
concentradas, incentivando la diversificación de acciones aun cuando
ε ya sea bajo.

---

## 8. Bucle de Entrenamiento Completo

`ext/pandemic.py` → `A2CTraining()`:

```
Para cada episodio i = 1 … N:
    env.random_sequence()          ← secuencia aleatoria
    state = env.reset()

    batch_states, batch_actions, batch_rewards,
    batch_next_states, batch_dones = [], [], [], [], []

    Mientras no done:
        1. Normalizar estado:     s_norm = normalize_state(state, 30, 10, 9)
        2. Selección con ε-overlay:
              - Con prob ε  → acción aleatoria factible
              - Con prob 1-ε → a ~ π_θ(· | s_norm)
        3. (Opcional) Meta-cognición: confidence = ac_agent_meta_cognitive(π(·|s), ...)
        4. Paso del entorno:      s', r, done = env.step(action)
        5. Almacenar transición en batch de episodio:
              batch_states.append(s_norm)
              batch_actions.append(action)
              batch_rewards.append(r)
              batch_next_states.append(s'_norm)
              batch_dones.append(done)

    ── Fin del episodio ──
    6. Convertir batch a tensores:
         states_t, actions_t, rewards_t, next_t, dones_t = cast_to_tensors(batch)
    7. Actualizar Actor y Critic:
         actor_loss, critic_loss, entropy = train_step_actor_critic(
             actor, critic, actor_opt, critic_opt,
             states_t, actions_t, rewards_t, next_t, dones_t,
             discount, entropy_coeff)
    8. Decaer ε

    Cada 10 000 episodios:
        ─ Imprimir recompensa promedio

    Retornar (ave_reward_list, actor, critic, conf_list)
```

### 8.1 Diferencia Clave: On-Policy vs. Off-Policy

| Aspecto | DQN (off-policy) | A2C (on-policy) |
|---------|-------------------|-----------------|
| Buffer | Replay buffer de 50 000 transiciones | Sin buffer; batch por episodio |
| Reutilización de datos | Cada transición se muestrea múltiples veces | Cada transición se usa exactamente una vez |
| Actualización | Cada 4 env steps (mini-batch del buffer) | Al final de cada episodio (batch completo) |
| Correlación temporal | Eliminada por muestreo aleatorio | Presente dentro del episodio |
| Eficiencia de datos | Alta (reutilización) | Baja (un solo uso) |
| Estabilidad del gradiente | Target network | Advantage centrada + entropía |

### 8.2 Parámetro `verbose`

`A2CTraining()` acepta un parámetro opcional `verbose` (por defecto `True`).
Cuando está desactivado (`verbose=False`), se suprimen los mensajes
periódicos de progreso que normalmente se imprimen cada 10 000 episodios.
Esto es especialmente útil durante la optimización bayesiana, donde
decenas de trials consecutivos generarían ruido excesivo en la terminal.

De forma análoga, `run_experiment()` también acepta `verbose` (por defecto
`True`).  Cuando `verbose=False`, se omiten las impresiones del estado
inicial y de los valores de severidad por secuencia.

### 8.3 Parámetro `compute_confidence`

Al igual que `DQNTraining()`, `A2CTraining()` acepta `compute_confidence`
(por defecto `False`).  Cuando está desactivado, se omite la llamada a
`ac_agent_meta_cognitive()` durante entrenamiento, ahorrando cómputo.

En A2C la meta-cognición **no** requiere un forward pass adicional: la
distribución $\pi_\theta(a \mid s)$ ya se calculó para la selección de
acción.  Sin embargo, la función `ac_agent_meta_cognitive()` realiza
operaciones adicionales (enmascaramiento de acciones, cálculo de entropía,
normalización) que se pueden omitir durante entrenamiento intensivo.

---

## 9. Optimización Bayesiana de Hiperparámetros

`ext/optimize_ac.py` utiliza **Optuna** (TPE sampler) para buscar
hiperparámetros óptimos del A2C.

### 9.1 Espacio de Búsqueda

| Parámetro | Rango | Tipo |
|-----------|-------|------|
| `actor_lr` | $[10^{-4},\; 10^{-2}]$ | log-uniforme |
| `critic_lr` | $[10^{-4},\; 10^{-2}]$ | log-uniforme |
| `discount_factor` | $[0.85,\; 0.99]$ | uniforme |
| `entropy_coeff` | $[10^{-3},\; 10^{-1}]$ | log-uniforme |
| `epsilon_initial` | $[0.50,\; 0.90]$ | uniforme |
| `epsilon_min` | $[0.02,\; 0.10]$ | uniforme |
| `num_episodes` | $[50\,000,\; 300\,000]$ | entero (paso 25k) |
| `actor_hidden_dim` | $[32,\; 64]$ | entero (paso 32) |
| `critic_hidden_dim` | $[32,\; 64]$ | entero (paso 32) |
| `n_hidden_layers` | $\{1, 2\}$ | entero |

**Diferencias frente al espacio de DQN:**

- A2C tiene **dos learning rates** (Actor y Critic) en lugar de uno solo.
- Se añade `entropy_coeff` — ausente en DQN.
- No existen `batch_size`, `replay_buffer_size`, ni `target_sync_freq`
  (conceptos exclusivos de DQN off-policy).

### 9.2 Función Objetivo

Se entrena un agente A2C con los hiperparámetros sugeridos y se evalúa
en las **64 secuencias fijas** (las mismas que usa `__main__.py`).  El score
reportado a Optuna es el **rendimiento normalizado medio** (a maximizar),
calculado con `calculate_normalised_final_severity_performance_metric()`.

Durante la evaluación, tanto `A2CTraining()` como `run_experiment()` se
invocan con `verbose=False` para suprimir las impresiones de consola por
episodio/secuencia, evitando ruido en la salida de terminal durante
decenas de trials consecutivos.

### 9.3 Poda Temprana (MedianPruner)

El estudio de Optuna incorpora un **MedianPruner** configurado con
`n_startup_trials=5` y `n_warmup_steps=2`.  Este pruner descarta trials
cuyo rendimiento intermedio es inferior a la mediana de trials anteriores,
acelerando la búsqueda al evitar completar trials prometedoramente malos:

```python
study = optuna.create_study(
    ...
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    ...
)
```

### 9.4 Warm-Start

La búsqueda comienza con un **trial semilla** usando los valores por defecto
de `CONFIG.py`, asegurando que al menos un trial alcance un rendimiento
razonable:

```python
warm_start = {
    "actor_lr": 3e-4,
    "critic_lr": 1e-3,
    "discount_factor": 0.99,
    "entropy_coeff": 0.01,
    "epsilon_initial": 0.679,
    "epsilon_min": 0.085,
    "num_episodes": 100000,
    "actor_hidden_dim": 64,
    "critic_hidden_dim": 64,
    "n_hidden_layers": 2
}
```

---

## 10. Inferencia en Tiempo de Experimento

### 10.1 Carga del Modelo Actor

`src/pygameMediator.py` → `provide_ac_agent_response()`:

```python
model = tf.keras.models.load_model(model_path)
```

Solo se carga el **Actor** — el Critic no es necesario en inferencia.

### 10.2 Selección de Acción

```python
s_norm = normalize_state([resources_left, city_trial_no, severity], 30, 10, 9)
policy_probs = model(s_norm[numpy.newaxis, :], training=False)[0].numpy()
action = int(numpy.argmax(policy_probs))
```

A diferencia de DQN (que toma $\arg\max$ de Q-values), aquí se toma
$\arg\max$ de $\pi_\theta(a \mid s)$ — la acción más probable bajo la
política aprendida.  Se aplica enmascaramiento de acciones infactibles
(`action > resources_left`) estableciendo su probabilidad a 0 antes del
argmax.

### 10.3 Confianza Meta-Cognitiva

El vector $\pi_\theta(a \mid s)$ (11 probabilidades) se pasa a
`ac_agent_meta_cognitive()` que calcula entropía y la normaliza:

$$H(\pi_\theta(\cdot \mid s)) = -\sum_{a=0}^{10} \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$$

$$\text{confidence} = \frac{H - H_\text{max}}{H_\text{min} - H_\text{max}}$$

donde $H_\text{min}$ y $H_\text{max}$ se calculan sobre las acciones
factibles únicamente.

---

## 11. Comparación con Algoritmos Anteriores

| Componente | `pes` (Q-tabular) | `pes_dqn` (DQN) | `pes_ac` (A2C) |
|------------|-------------------|-----------------|--------------------------|
| Modelo | `numpy.ndarray` (q.npy) | Red 5 131 params (.keras) | Actor 5 131 + Critic 4 481 params |
| Update | $Q(s,a) \leftarrow Q + \alpha[r + \gamma \max Q - Q]$ | Huber loss + replay | Policy gradient + MSE + entropía |
| Datos | Un paso → un update | Replay buffer → mini-batch | Batch de episodio → un update |
| Política | Implícita ($\arg\max Q$) | Implícita ($\arg\max Q$) | Explícita ($\pi_\theta$) |
| Confianza | Entropía de Q (heurística) | Entropía de Q (heurística) | Entropía de $\pi$ (teórica) |
| Exploración | ε-greedy | ε-greedy | Muestreo de $\pi$ + ε-overlay + entropía bonus |
| Episodios típicos | 900 000 | 100 000 | 100 000+ |

---

## 12. Optimizaciones para CPU

### 12.1 `tf.function` por Trial (JIT Compilado)

`train_step_actor_critic` se envuelve con `tf.function` **localmente**
dentro de cada llamada a `A2CTraining`, creando un grafo JIT-compilado
fresco por trial de Optuna.  Esto elimina el overhead de eager mode
que sería particularmente costoso dado que A2C realiza una actualización
por episodio (vs. cada 4 steps en DQN), y a la vez evita conflictos
de `tf.Variable` entre trials.

Además, los hiperparámetros escalares (`discount`, `entropy_coeff`) se
convierten a `tf.constant` antes del loop para que `tf.function` no
retrace el grafo en cada trial con valores distintos.

### 12.2 Eliminación del Forward Pass de Confianza

El parámetro `compute_confidence=False` (por defecto) omite el cálculo
de meta-cognición durante entrenamiento.  En A2C esto no ahorra un
forward pass adicional (a diferencia de DQN), pero sí evita el cómputo
de entropía, enmascaramiento, y normalización por step.

### 12.3 Configuración de Hilos TensorFlow

Al importar `ext/ac_model.py` se configuran los pools de hilos de TF:

```python
tf.config.threading.set_intra_op_parallelism_threads(0)   # auto-detect
tf.config.threading.set_inter_op_parallelism_threads(2)
```

Además, `OMP_NUM_THREADS` se configura en `__init__.py` al número de
cores disponibles antes de importar TensorFlow.

### 12.4 Solo el Actor en Inferencia

Durante el experimento (`__main__.py`), solo se carga y ejecuta el Actor.
El Critic se descarta tras el entrenamiento, reduciendo la memoria en
inferencia a ~50 % del total.

---

## 13. Estructura de Archivos

```
pes_ac/
├── __init__.py              # Exporta constantes AC_*; configura OMP_NUM_THREADS
├── __main__.py              # Valida ac_actor.keras antes de ejecutar
├── config/CONFIG.py         # 8 constantes AC_*
├── ext/
│   ├── ac_model.py          # build_actor, build_critic, normalize_state,
│   │                        #   train_step_actor_critic (sin @tf.function;
│   │                        #   se envuelve por trial en pandemic.py);
│   │                        #   configura tf.config.threading al importar
│   ├── pandemic.py          # Entorno Gym + A2CTraining(compute_confidence=False)
│   ├── train_ac.py          # Pipeline de entrenamiento autónomo
│   ├── optimize_ac.py       # Búsqueda Bayesiana con Optuna
│   └── tools.py             # Entropía, gráficas (sin cambios)
├── src/
│   ├── pygameMediator.py    # Carga ac_actor.keras, forward pass en experimento
│   ├── exp_utils.py         # Severidades, secuencias
│   ├── log_utils.py         # Logging dual
│   ├── result_formatter.py  # Gráficas matplotlib
│   └── terminal_utils.py    # UI de consola Rich
└── doc/
    ├── explained_ac.md              # ← este documento
    └── how_to_train_and_test.md     # Guía práctica de entrenamiento y pruebas
```

---

## 14. Formulario Resumen de Ecuaciones

| Concepto | Ecuación |
|----------|----------|
| Objetivo del Actor | $J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \gamma^t r_t\right]$ |
| Gradiente de Política | $\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat{A}\right]$ |
| Ventaja (Advantage) | $\hat{A}_t = r_t + \gamma V_\phi(s_{t+1})(1 - d_t) - V_\phi(s_t)$ |
| TD Target | $y_t = r_t + \gamma V_\phi(s_{t+1})(1 - d_t)$ |
| Pérdida del Critic | $\mathcal{L}_\text{C} = \frac{1}{N}\sum_t (V_\phi(s_t) - y_t)^2$ |
| Pérdida del Actor | $\mathcal{L}_\text{A} = -\frac{1}{N}\sum_t [\log \pi_\theta(a_t \mid s_t) \hat{A}_t] - c_\text{ent} H(\pi_\theta)$ |
| Entropía | $H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ |
| Decaimiento ε | $\varepsilon_{i+1} = \max(\varepsilon_\min, \varepsilon_i - \Delta)$ |
| Normalización | $\hat{s} = (r/30, t/10, v/9)$ |

---

## 15. Referencias

1. Sutton, R. S. et al. (1999). *Policy gradient methods for reinforcement
   learning with function approximation*. NeurIPS.
2. Mnih, V. et al. (2016). *Asynchronous methods for deep reinforcement
   learning*. ICML.  (Introduce A3C; A2C es la variante síncrona.)
3. Williams, R. J. (1992). *Simple statistical gradient-following
   algorithms for connectionist reinforcement learning*. Machine Learning,
   8(3-4), 229–256.  (REINFORCE.)
4. Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic
   Optimization*. ICLR 2015.
5. Akiba, T. et al. (2019). *Optuna: A next-generation hyperparameter
   optimization framework*. KDD '19.
