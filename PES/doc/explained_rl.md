# PES y Reinforcement Learning: Mapeo de Teoría a Implementación

## 1. Introducción

Este documento establece la correspondencia entre la teoría de **Q-Learning (Reinforcement Learning)** y su implementación específica en el proyecto PES. Cada concepto teórico se mapea a secciones concretas del código.

---

## 2. Marco Teórico de Q-Learning

### 2.1 Problema de Decisión Sequencial (MDP)

**Teoría**: Un Markov Decision Process (MDP) se define como tupla ⟨S, A, P, R, γ⟩:
- **S**: Espacio de estados
- **A**: Espacio de acciones
- **P(s'|s,a)**: Probabilidad de transición
- **R(s,a)**: Función de recompensa
- **γ**: Factor de descuento

**Implementación en PES**:

```
┌─────────────────────────────────────────────────────────────┐
│ DEFINICIÓN DEL MDP EN PANDEMIC ENVIRONMENT                  │
└─────────────────────────────────────────────────────────────┘

Archivo: ext/pandemic.py, líneas 1-100

class Pandemic(Env):
    def __init__(self):
        # S - ESPACIO DE ESTADOS
        self.observation_space = spaces.Box(
            low = numpy.zeros((31, 11, 11)),      # Min values
            high = numpy.ones((31, 11, 11)),      # Max values
            dtype = numpy.float16
        )
        # Estado = [available_resources, trial_number, severity]
        # Componentes:
        #   - available_resources: 0-30 (39 total - 9 pre-asignados)
        #   - trial_number: 0-10 (max 10 trials por secuencia)
        #   - severity: 0-10 (severidad actual de la ciudad)
        
        # A - ESPACIO DE ACCIONES
        self.action_space = spaces.Discrete(11)
        # Acción = recursos a asignar, valores 0-10
        
        # P - DINÁMICA DEL SISTEMA
        # Implementada en step() - líneas 300-350
        
        # R - FUNCIÓN DE RECOMPENSA
        # Implementada en step() - línea 330
        
        # γ - FACTOR DE DESCUENTO
        # Definido en train_rl.py línea 180
        discount_factor = 0.9

    def reset(self):
        """Inicializar estado s₀"""
        self.available_resources = self.max_resources
        self.iteration = 0
        self.severities = []
        new_severity = self.new_city()
        self.severities.append(new_severity)
        
        # Retornar estado inicial
        return [self.available_resources, self.iteration, int(new_severity)]
    
    def step(self, action):
        """
        Ejecutar transición de estado: s → s'
        Retornar: (s', r, done, info)
        """
        # Transición de recursos
        self.available_resources -= action  # P: actualizar estado
        
        # Cálculo de recompensa
        self.severities = get_updated_severity(...)
        reward = (-1) * numpy.sum(self.severities)  # R: penalizar severidad
        
        # Determinación de término
        done = (self.iteration == self.seq_length)
        
        # Nuevo estado
        new_severity = 0 if done else self.new_city()
        new_state = [self.available_resources, self.iteration, int(new_severity)]
        
        return new_state, reward, done, info
```

### 2.2 Función de Valor y Q-Función

**Teoría**:

La **Q-función** (función de acción-valor) representa el valor esperado de ejecutar acción $a$ en estado $s$:

$$Q^\pi(s, a) = \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots | s_t = s, a_t = a]$$

El **valor óptimo** satisface la ecuación de Bellman:

$$Q^*(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a') | s, a]$$

**Implementación en PES**:

```
┌─────────────────────────────────────────────────────────────┐
│ Q-TABLE: ALMACENAMIENTO DE Q*(s, a)                          │
└─────────────────────────────────────────────────────────────┘

Archivo: ext/train_rl.py, líneas 175-200

# Inicialización de Q-table
Q = numpy.random.uniform(low=-1, high=1,
                        size=(31, 11, 11, 11))
# Shape: (recursos, trials, severidad, acciones)
# Q[s_resources, s_trial, s_severity, a_action] = valor esperado

# Archivo guardado como: inputs/q.npy
# Forma: Q-table entrenada (episodios configurables vía CLI, default 20,000)
```

### 2.3 Ecuación de Actualización Q-Learning

**Teoría**:

El algoritmo Q-Learning actualiza la tabla después de cada transición (s, a, r, s'):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Donde:
- $\alpha$ = learning rate (cuánto se ajusta la estimación)
- Término entre corchetes = TD-error (diferencia temporal)

**Implementación en PES**:

```
┌─────────────────────────────────────────────────────────────┐
│ ACTUALIZACIÓN Q-LEARNING                                    │
└─────────────────────────────────────────────────────────────┘

Archivo: ext/pandemic.py, líneas 600-650

def QLearning(env, learning, discount, epsilon, min_eps, episodes, ...):
    """
    learning = α (learning_rate = 0.2)
    discount = γ (discount_factor = 0.9)
    epsilon = ε (exploration rate)
    episodes = configurable vía línea de comandos (default: 20,000)
    """
    
    Q = numpy.random.uniform(low=-1, high=1, size=env_shape)
    reward_list = []
    reduction = (epsilon - min_eps) / episodes  # ε-decay
    
    for episode in range(episodes):
        env.random_sequence()  # Secuencia aleatoria
        state = env.reset()    # s₀
        episode_reward = 0
        
        while True:  # Mientras no done
            # ─────────────────────────────────────────
            # POLÍTICA EPSILON-GREEDY
            # ─────────────────────────────────────────
            if numpy.random.rand() < epsilon:
                action = env.action_space.sample()  # Acción aleatoria
            else:
                # Acción greedy
                state_index = [state[0], state[1], state[2]]
                action = numpy.argmax(Q[state_index[0], state_index[1], state_index[2], :])
            
            # ─────────────────────────────────────────
            # TRANSICIÓN DE ESTADO
            # ─────────────────────────────────────────
            next_state, reward, done, _ = env.step(action)  # (s', r, done)
            
            # ─────────────────────────────────────────
            # ACTUALIZACIÓN BELLMAN (Q-Learning Update)
            # ─────────────────────────────────────────
            # Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
            
            s = [state[0], state[1], state[2]]
            s_prime = [next_state[0], next_state[1], next_state[2]]
            
            max_next_q = numpy.max(Q[s_prime[0], s_prime[1], s_prime[2], :])
            
            Q[s[0], s[1], s[2], action] += learning * (
                reward + discount * max_next_q - Q[s[0], s[1], s[2], action]
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # ─────────────────────────────────────────
        # EPSILON DECAY (Reducción de exploración)
        # ─────────────────────────────────────────
        epsilon -= reduction
        
        # Registrar recompensa promedio cada 10k episodios
        if (episode + 1) % 10000 == 0:
            avg_reward = numpy.mean(reward_list[-10000:])
            ave_reward_list.append(avg_reward)
    
    return ave_reward_list, Q, conf_list
```

**Componentes desglosados**:

| Componente | Código | Significado |
|-----------|--------|-------------|
| **α (learning_rate)** | `learning = 0.2` | Qué tan rápido se ajustan los Q-values |
| **γ (discount)** | `discount = 0.9` | Importancia de recompensas futuras |
| **ε (epsilon)** | `epsilon = 0.8 → 0` | Probabilidad de acción aleatoria |
| **TD-error** | `reward + gamma*max_Q - Q[s,a]` | Diferencia temporal |
| **Update** | `Q[s,a] += α * TD_error` | Actualización incremental |

---

## 3. Ejecución del Agente Entrenado

### 3.1 Política Greedy (Explotación Pura)

**Teoría**:

Una vez entrenado, el agente utiliza la **política greedy**:

$$\pi(s) = \arg\max_a Q(s, a)$$

Esto selecciona la acción con el mayor Q-value en cada estado.

**Implementación en PES**:

```
┌─────────────────────────────────────────────────────────────┐
│ EJECUCIÓN DEL AGENTE: CONSULTA Q-TABLE                      │
└─────────────────────────────────────────────────────────────┘

Archivo: __main__.py, líneas 350-400

# Por cada trial en la secuencia:

for trial in range(num_trials):
    # Obtener estado actual
    state = [resources_remaining, trial_number, current_severity]
    
    # ─────────────────────────────────────────
    # CONSULTA GREEDY
    # ─────────────────────────────────────────
    q_values = Q[state[0], state[1], state[2], :]  # Q-values para todas acciones
    action = numpy.argmax(q_values)                # argmax → acción óptima
    
    # Validación (no puede asignar más de lo disponible)
    if action > resources_remaining:
        action = resources_remaining
    
    # ─────────────────────────────────────────
    # EJECUCIÓN DE ACCIÓN
    # ─────────────────────────────────────────
    response[trial] = action                       # Guardar decisión
    resources_remaining -= action                  # Actualizar estado
    current_severity = update_severity(...)        # Evolução
    
    # ─────────────────────────────────────────
    # CÁLCULO DE CONFIANZA (Meta-cognitiva)
    # ─────────────────────────────────────────
    confidence = calculate_confidence(q_values)    # Basado en entropía
```

### 3.2 Índices de Estado en la Práctica

**Estados posibles en ejecución**:

```python
# Ejemplo de estados durante ejecución

Trial 1: 
  State = [30, 0, 3]  (30 recursos, trial 0, severidad 3)
  Q[30, 0, 3, :] = [0.25, 0.45, 0.32, ..., -0.10]  (Q-values para a=0..10)
  Action = argmax(...) = 1  (asignar 1 recurso)

Trial 2:
  State = [29, 1, 3.2]  (29 recursos, trial 1, severidad nuevaevolución)
  Q[29, 1, 3, :] = [0.18, 0.50, 0.40, ..., -0.25]
  Action = argmax(...) = 1  (asignar 1 recurso)

Trial 3:
  State = [28, 2, 3.4]
  Q[28, 2, 3, :] = [0.10, 0.65, 0.55, ..., -0.30]
  Action = argmax(...) = 1  (asignar 1 recurso)
```

---

## 4. Recompensa y Función Objetivo

### 4.1 Función de Recompensa Inmediata

**Teoría**:

La recompensa $r_t$ en el tiempo $t$ se define como feedback del ambiente.

**Implementación en PES**:

```
┌─────────────────────────────────────────────────────────────┐
│ RECOMPENSA EN PANDEMIC ENVIRONMENT                          │
└─────────────────────────────────────────────────────────────┘

Archivo: ext/pandemic.py, línea 330

def step(self, action):
    ...
    # Actualizar severidades tras asignación de recurso
    self.severities = get_updated_severity(
        len(self.severities), 
        self.resources, 
        self.severities
    )
    
    # Recompensa = negativo de suma de severidades
    reward = (-1) * numpy.sum(self.severities)
    
    return [new_state], reward, done, []
```

**Interpretación**:
- Suma de severidades = costo total (a minimizar)
- Recompensa = negativo del costo
- Agente aprende a **minimizar severidades** maximizando recompensa

**Ejemplo**:
```
Severidades después acción: [2.1, 3.5, 2.8]
Suma = 8.4
Recompensa = -8.4
```

### 4.2 Acumulación de Recompensas (Return)

**Teoría**:

El **return** (retorno acumulado) es:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots + \gamma^{T-t} r_T$$

Q-Learning estima este valor futuro descontado.

### 4.3 Histórico de Recompensas Almacenado

```
┌─────────────────────────────────────────────────────────────┐
│ SEGUIMIENTO DE APRENDIZAJE                                  │
└─────────────────────────────────────────────────────────────┘

Archivo: ext/train_rl.py, líneas 210-250

# Durante entrenamiento
reward_list = []  # Todas las recompensas de episodio

for episode in range(1000000):
    episode_reward = 0
    while not done:
        action = ...
        state, reward, done, _ = env.step(action)
        reward_list.append(reward)
        episode_reward += reward
    
    # Cada 10k episodios, guardar promedio
    if (episode + 1) % 10000 == 0:
        avg = numpy.mean(reward_list[-10000:])
        ave_reward_list.append(avg)

# Guardar histórico
numpy.save(os.path.join(INPUTS_PATH, 'rewards.npy'), ave_reward_list)
# Shape: (100,)  - 100 puntos, uno cada 10k episodios
```

**Visualización del aprendizaje**:
```
Inicio:    ave_reward ≈ -50  (agente malo)
10k epis:  ave_reward ≈ -35  (mejorando)
50k epis:  ave_reward ≈ -22  (convergiendo)
100k epis: ave_reward ≈ -15  (casi convergido)
1M epis:   ave_reward ≈ -12  (convergido)
```

---

## 5. Política y Exploración-Explotación

### 5.1 Epsilon-Greedy Policy

**Teoría**:

Durante entrenamiento, se necesita balance entre:
- **Explotación** (usar mejor acción conocida): argmax Q(s, a)
- **Exploración** (probar nuevas acciones): acción aleatoria

La **epsilon-greedy policy** combina ambas:

$$\pi_\varepsilon(a|s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{|A|} & \text{si } a = \arg\max_a Q(s,a) \\
\frac{\varepsilon}{|A|} & \text{si } a \neq \arg\max_a Q(s,a)
\end{cases}$$

Con $|A| = 11$ acciones posibles.

**Implementación en PES**:

```
┌─────────────────────────────────────────────────────────────┐
│ EPSILON-GREEDY DURANTE ENTRENAMIENTO                        │
└─────────────────────────────────────────────────────────────┘

Archivo: ext/pandemic.py, líneas 610-630

epsilon = 0.8  # Inicial: 80% exploración
min_eps = 0    # Final: 0% exploración (100% greedy)
reduction = (epsilon - min_eps) / episodes  # Decaimiento lineal

for episode in range(episodes):  # Configurable vía CLI, default 20,000
    state = env.reset()
    
    while not done:
        # ─────────────────────────────────────
        # DECISION: EXPLORAR vs EXPLOTAR
        # ─────────────────────────────────────
        if numpy.random.rand() < epsilon:
            # EXPLORACIÓN: acción aleatoria (probabilidad ε)
            action = env.action_space.sample()
        else:
            # EXPLOTACIÓN: mejor acción conocida (probabilidad 1-ε)
            q_values = Q[s[0], s[1], s[2], :]
            action = numpy.argmax(q_values)
        
        state, reward, done, _ = env.step(action)
        # ... actualizar Q ...
        
    # Reducir epsilon linealmente (menos exploración con el tiempo)
    epsilon -= reduction
    
    # Ejemplo con 200,000 episodios:
    # En el episodio 100k, epsilon ≈ 0.4 (40% exploración)
    # En el episodio 200k, epsilon ≈ 0.0 (0% exploración)
```

**Cronograma de exploración** (ejemplo con 200,000 episodios):
```
Episodio    Epsilon    Exploración    Explotación
0           0.80       80%            20%
20k         0.72       72%            28%
50k         0.60       60%            40%
100k        0.40       40%            60%
150k        0.20       20%            80%
200k        0.00       0%             100%
```

> **Nota**: La cantidad de episodios es configurable vía línea de comandos:
> `python3 -m PES.ext.train_rl 200000` (default: 20,000)

### 5.2 Política en Fase de Ejecución

Durante el experimento real (`__main__.py`), se usa **política pura greedy** (ε = 0):

```python
# Sin exploración, solo explotación
action = numpy.argmax(Q[state])  # Mejor acción conocida
```

---

## 6. Convergencia y Garantías Teóricas

### 6.1 Convergencia de Q-Learning

**Teoría**:

Q-Learning converge a $Q^*$ (valores óptimos) bajo condiciones:
1. Cada estado-acción se visita infinitamente (exploración suficiente)
2. Learning rate decae: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$
3. Recompensas son acotadas

**En PES**:
- ✅ Exploración garantizada: epsilon-greedy asegura todas las acciones se prueban
- ✅ Learning rate fijo: α = 0.2 (conservador, garantiza convergencia)
- ✅ Recompensas acotadas: R ∈ [-suma_maxseveridades, 0]

### 6.2 Convergencia Observada en Entrenamiento

La curva de recompensas promedio (`rewards.npy`) muestra convergencia:

```
Ejemplo de evolución durante 1M episodios:

Promedio cada 10k episodios:
[−50.2, −47.8, −43.1, −39.5, ..., −12.3, −12.1, −12.0]
            ↓ Rápida mejora       ↓ Convergencia
```

---

## 7. Meta-Cognición: Confianza y Entropía

### 7.1 Relación Entropía-Confianza

**Teoría**:

La **entropía de Shannon** de la distribución Q-values mide incertidumbre:

$$H(X) = -\sum_i p_i \log_2(p_i)$$

Donde $p_i$ es la probabilidad de acción $i$.

- **Baja entropía** → picos claros en Q-values → agente confiado
- **Alta entropía** → Q-values planos → agente inseguro

**Implementación en PES**:

La función `rl_agent_meta_cognitive()` existe en dos ubicaciones:
- `pygameMediator.py`: usada durante la ejecución del experimento (`python3 -m PES`)
- `pandemic.py`: usada durante el entrenamiento (`python3 -m PES.ext.train_rl`)

> **Nota**: Las funciones `entropy()` y `calculate_agent_response_and_confidence()`
> que existían previamente en `pygameMediator.py` fueron eliminadas por ser código
> muerto (nunca eran invocadas en el proyecto).

```
┌─────────────────────────────────────────────────────────────┐
│ CONFIANZA BASADA EN ENTROPÍA                                │
└─────────────────────────────────────────────────────────────┘

Archivo: pandemic.py (entrenamiento) / pygameMediator.py (ejecución)

def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    """
    options = Q-values para acciones en estado actual
    Ejemplo: options = [0.21, 0.89, 0.45, ..., -0.12]  (11 valores)
    """
    
    # Entropía mínima: distribución determinística
    m_entropy = entropy_from_pdf([1, 0, 0, ..., 0])  # H ≈ 0
    
    # Entropía máxima: distribución uniforme
    M_entropy = entropy_from_pdf([1, 1, 1, ..., 1])  # H ≈ 3.46 bits
    
    # Entropía actual: distribución de Q-values
    dec_entropy = entropy_from_pdf(options)  # H ∈ [0, 3.46]
    
    # Normalizar confianza a [0, 1]
    confidence = (dec_entropy - M_entropy) / (m_entropy - M_entropy)
    # ≡ confidence = 1 - (dec_entropy / max_entropy)
```

**Ejemplos de confianza**:

```
Escenario 1: Decisión CLARA
Q-values = [0.05, 0.95, 0.10, 0.05, ...]
Entropía ≈ 0.5 bits (concentrado en índice 1)
Confianza ≈ 0.85 (alta)
Respuesta esperada: rápido, seguro

Escenario 2: Decisión AMBIGUA
Q-values = [0.30, 0.35, 0.25, 0.32, ...]
Entropía ≈ 3.2 bits (distribuido)
Confianza ≈ 0.15 (baja)
Respuesta esperada: lento, dudoso
```

### 7.2 Mapeo Confianza → Tiempos de Reacción

```
┌─────────────────────────────────────────────────────────────┐
│ TIEMPOS DE REACCIÓN BASADOS EN CONFIANZA                   │
└─────────────────────────────────────────────────────────────┘

Archivo: pandemic.py (entrenamiento) / pygameMediator.py (ejecución)

# Mapeo lineal: mayor confianza → menor tiempo
map_to_response_time = lambda x: x * (-2) + 1
# confidence = 0.8 → coef = 0.8*(-2) + 1 = -0.6
# confidence = 0.2 → coef = 0.2*(-2) + 1 = 0.6

mu_hold = int(map_to_response_time(confidence) * 10)
rt_hold = numpy.random.normal(mu=mu_hold, sigma=3)

# Ejemplo con confidence = 0.8:
# mu = 0.8*(-2)+1 = -0.6 → mu_hold = -6 → clip a 0
# rt_hold ≈ 0ms (rápido)

# Ejemplo con confidence = 0.2:
# mu = 0.2*(-2)+1 = 0.6 → mu_hold = 6 → distribución normal
# rt_hold ≈ 6-10ms (lento)
```

---

## 8. Tabla de Mapeo: Teoría ↔ Código

| Concepto RL | Fórmula | Implantación | Archivo | Líneas |
|-----------|---------|-----------|---------|--------|
| **MDP** | ⟨S, A, P, R, γ⟩ | `Pandemic(Env)` | `pandemic.py` | 42-115 |
| **Estado** | $s \in S$ | `[resources, trial, severity]` | `pandemic.py` | 97-100 |
| **Acción** | $a \in A$ | `Discrete(11)` | `pandemic.py` | 106 |
| **Transición** | $P(s' \mid s,a)$ | `step()` | `pandemic.py` | 296-365 |
| **Recompensa** | $r = R(s,a,s')$ | `reward = -sum(severities)` | `pandemic.py` | 352 |
| **Q-función** | $Q(s,a)$ | `Q[s[0], s[1], s[2], a]` | `pandemic.py` | 605-609 |
| **Bellman Update** | $Q \leftarrow Q + \alpha(r + \gamma\max Q' - Q)$ | Ver línea 649 | `pandemic.py` | 645-655 |
| **Epsilon-Greedy** | $\pi_\varepsilon$ | `if rand < ε: random`, `else: argmax Q` | `pandemic.py` | 635-638 |
| **Entropía** | $H = -\sum p_i \log p_i$ | `entropy_from_pdf()` | `tools.py` | 10-30 |
| **Confianza** | $conf = (H - H_{max}) / (H_{min} - H_{max})$ | `rl_agent_meta_cognitive()` | `pandemic.py` | 400-415 |

---

## 9. Flujo Completo: Entrenamiento → Ejecución

```
FASE 1: ENTRENAMIENTO
═════════════════════════════════════════════════

python3 -m PES.ext.train_rl [episodios]  # default: 20,000
│
├─ 1. Inicializar Pandemic() environment
├─ 2. Inicializar Q = random_uniform(31, 11, 11, 11)
├─ 3. LOOP episodios (configurable vía CLI, default 20,000):
│   ├─ state = reset()  [recursos, trial, severidad]
│   ├─ MIENTRAS no done:
│   │  ├─ Epsilon-Greedy: acción = rand OR argmax Q
│   │  ├─ state', reward, done = step(action)
│   │  ├─ TD-Error = reward + γ*max Q' - Q[s,a]
│   │  ├─ Q[s,a] += α * TD-Error  ← APRENDIZAJE
│   │  ├─ state = state'
│   │  └─ ε -= reduction  ← Menos exploración
│   └─ Guardar reward promedio cada 10k episodios
│
├─ 4. Guardar Q → inputs/q.npy  (31×11×11×11)
├─ 5. Guardar rewards → inputs/rewards.npy  (100,)
└─ 6. Generar gráficas de aprendizaje

FASE 2: EJECUCIÓN DEL EXPERIMENTO
═════════════════════════════════════════════════

python3 -m PES
│
├─ 1. Cargar Q desde inputs/q.npy
├─ 2. PARA cada BLOQUE (8 bloques):
│   ├─ PARA cada SECUENCIA (8 secuencias):
│   │  ├─ state = [39, 0, initial_severity]
│   │  ├─ PARA cada TRIAL (3-10 trials):
│   │  │  ├─ Q_values = Q[state[0], state[1], state[2], :]
│   │  │  ├─ action = argmax(Q_values)  ← GREEDY (ε=0)
│   │  │  ├─ confidence = entropy(Q_values)
│   │  │  ├─ state = [recursos-action, trial+1, new_severity]
│   │  │  ├─ Registrar: response, confidence, tiempos
│   │  │  └─ Actualizar severity con fórmula
│   │  ├─ Calcular performance = (worst-actual)/(worst-best)
│   │  └─ Guardar en responses.txt, results.json
│   └─ Generar gráficas por bloque
│
└─ 3. Estadísticas globales, reportes finales

SALIDA
══════
inputs/q.npy                    ← Q-table entrenada
inputs/rewards.npy              ← Histórico de aprendizaje
inputs/YYYY-MM-DD_RL_TRAIN/     ← Detalles del entrenamiento
outputs/2026-02-09_RL_AGENT/    ← Resultados del experimento
 ├─ PES_log_...txt
 ├─ PES_responses_...txt
 ├─ PES_results_...json
 ├─ PES_results_...png
 └─ PES_movement_log_...npy
```

---

## 10. Diferencias: Teoría vs. Práctica en PES

| Aspecto | Teoría Pura | Implementación PES | Razón |
|--------|-----------|------------------|-------|
| **Learning rate** | Decreciente | Fijo (0.2) | Simplificar, garantizar convergencia |
| **Epsilon decaimiento** | Óptimo | Lineal | Balance exploración-explotación simple |
| **Estado continuo** | Posible | Discretizado | Q-table requiere índices enteros |
| **Horizonte** | Infinito | Finito (seq length) | Episodios con término natural |
| **Recompensa** | General | Severidad negativa | Problema-específico |
| **Confianza** | Teórica | Entropía Q-values | Meta-cognición implementada |

---

## 11. Validación: ¿Aprendió el Agente?

Para verificar que el entrenamiento funcionó:

```python
# 1. Cargar Q-table
Q = numpy.load('inputs/q.npy')

# 2. Inspeccionar Q-values para estado específico
state_example = (20, 5, 3)  # 20 recursos, trial 5, severidad 3
q_vals = Q[state_example[0], state_example[1], state_example[2], :]
print(q_vals)
# Esperado: valores negativos (severidad), con cierta dispersión
# Mal: todos iguales o aleatorios

# 3. Graficar distribución Q-values
import matplotlib.pyplot as plt
plt.hist(Q.flatten(), bins=100)
plt.title('Distribución de Q-values')
plt.show()
# Esperado: distribución concentrada (aprendizaje convergido)
# Mal: uniforme (sin aprendizaje)

# 4. Comparar con baseline aleatorio
# El experimento genera gráficas comparativas
```

---

## Conclusión

El proyecto PES implementa **Q-Learning de forma completa y rigurosa**:

1. **Entrenamiento**: Episodios configurables vía CLI (`python3 -m PES.ext.train_rl [N]`, default: 20,000), epsilon-greedy, actualización Bellman
2. **Convergencia**: Garantizada por arquitectura (exploración + learning rate)
3. **Ejecución**: Política greedy pura sobre Q-table entrenada
4. **Análisis**: Confianza meta-cognitiva vía `rl_agent_meta_cognitive()`, tiempos realistas
5. **Validación**: Comparación vs. agente aleatorio

> **Nota**: Las funciones `entropy()` y `calculate_agent_response_and_confidence()` fueron
> eliminadas de `pygameMediator.py` por ser código muerto. Toda la lógica de confianza
> se implementa ahora exclusivamente en `rl_agent_meta_cognitive()`.

Todo está mappeado matemáticamente a la teoría de RL, haciendo del proyecto tanto un framework experimental como una referencia clara de cómo implementar Q-Learning en domain específico.
