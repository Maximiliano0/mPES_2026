# Reinforcement Learning Agent (RL-Agent)

## Introducción

Este documento detalla la implementación y operación del **Agente de Reinforcement Learning** utilizado en el Pandemic Experiment Scenario (PES). Se profundiza en cómo se entrena, cómo funciona durante la ejecución, y su relación con la teoría de RL.

---

## 1. Arquitectura General del Agente

### 1.1 Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│            REINFORCEMENT LEARNING AGENT (PES)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                                          │
│  │  Q-Learning  │◄──────────┐                              │
│  │  (Training)  │           │                              │
│  └──────────────┘           │                              │
│        │                    │                              │
│        ▼                    │ (Cargar Q-table)             │
│  ┌──────────────┐           │                              │
│  │   Q-Table    │───────────┘                              │
│  │ (Decisiones) │                                          │
│  └──────────────┘                                          │
│        │                    │                              │
│        ▼                    │                              │
│  ┌──────────────────┐       │                              │
│  │  Metacognitive   │       │                              │
│  │  Confidence      │       │                              │
│  │  (Agent.py)      │       │                              │
│  └──────────────────┘       │                              │
│        │                    │                              │
│        ▼                    │                              │
│  ┌──────────────────┐       │                              │
│  │ Response with    │       │                              │
│  │ Noise & Decay    │       │                              │
│  │ (Humanization)   │       │                              │
│  └──────────────────┘       │                              │
│        │                    │                              │
│        ▼                    │                              │
│  ┌──────────────────┐       │                              │
│  │ Final Resource   │       │                              │
│  │ Allocation (0-10)│       │                              │
│  └──────────────────┘       │                              │
│                             │                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Definición Formal del Problema

El agente se modela como un **Markov Decision Process (MDP)**:

- **Estado $(s)$**: La observación actual del entorno
- **Acción $(a)$**: Recurso a asignar (0-10)
- **Transición $(s \to s')$**: Cambio en severidad tras asignación
- **Recompensa $(r)$**: Negativa de la suma de severidades ($-\sum \text{severity}$)
- **Política $(\pi)$**: Mapeo del estado a acción óptima

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

---

## 2. Espacio de Estados y Acciones

### 2.1 Espacio de Estados

Definido en `ext/pandemic.py:43-60`:

```python
# Dimensiones del espacio de observación
self.available_resources_states = 41  # 0-40 recursos disponibles
self.trial_no_states = 13             # 0-12 trials
self.severity_states = 11             # 0-10 severidad

observation_shape = (41, 13, 11)      # Espacio total: 5,863 estados
```

**Interpretación del Estado**:

$$s = (\text{recursos\_disponibles}, \text{número\_trial}, \text{severidad\_actual})$$

Ejemplo:
```
s = (25, 5, 7)
├─ 25 recursos libres
├─ Trial número 5 de la secuencia
└─ Severidad actual de la ciudad: 7
```

**Transformación en Código** (observación a índices):

```python
# observation = [available_resources, iteration, new_severity]
state_index = (obs[0], obs[1], int(obs[2]))  # Tupla indexable en Q-table
```

### 2.2 Espacio de Acciones

```python
# action_space = spaces.Discrete(11)  # 11 acciones: 0-10
action = agent_decision(state)  # Retorna: 0, 1, 2, ..., o 10 recursos
```

**Interpretación**:
- Acción 0: No asignar recursos (dejar crecer pandemia)
- Acción 5: Asignar 5 recursos
- Acción 10: Asignar máximo de recursos

---

## 3. Función Q y Q-Learning

### 3.1 Definición Teórica de Q

La función Q-value representa el **valor esperado acumulado** al tomar una acción:

$$Q(s, a) = \text{suma acumulada de recompensas futuras descontadas}$$

$$Q(s, a) = E[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... | s_t=s, a_t=a]$$

Donde:
- $r_t$ = recompensa inmediata (en PES: $-\sum \text{severity}$)
- $\gamma$ = factor de descuento (importancia de recompensas futuras)

### 3.2 Algoritmo Q-Learning

Referencia: `ext/train_rl.py:120-140` (clase `QLearning`)

**Ecuación de Actualización (Bellman)**:

$$Q_{t+1}(s_t, a_t) \leftarrow Q_t(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t)]$$

Donde:
- $\alpha$ = **learning rate** (qué tan rápido aprender)
- $\gamma$ = **discount factor** (importancia de futuro)
- $r_t$ = recompensa observada
- $\max_{a'} Q_t(s_{t+1}, a')$ = mejor acción futura

**Pseudocódigo**:

```python
def QLearning(env, learning_rate, discount_factor, epsilon, episodes):
    Q = zeros((states, actions))  # Tabla Q inicializada en 0
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Exploración vs Explotación (ε-greedy)
            if random() < epsilon:
                action = random_action()  # Exploración
            else:
                action = argmax(Q[state, :])  # Explotación
            
            # Interactuar con entorno
            next_state, reward, done = env.step(action)
            
            # Actualizar Q-table (Bellman)
            old_Q = Q[state, action]
            max_next_Q = max(Q[next_state, :])
            Q[state, action] = old_Q + alpha * (reward + gamma * max_next_Q - old_Q)
            
            state = next_state
    
    return Q  # Tabla Q entrenada
```

### 3.3 Parámetros de Q-Learning en PES

Referencia: `ext/train_rl.py:90-100`

```python
learning_rate = 0.2    # α: Cuán rápido aprender (0-1)
discount_factor = 0.9  # γ: Importancia del futuro vs presente
epsilon = 0.8          # ε: Tasa de exploración inicial
epsilon_min = 0        # ε mínimo (convergencia a explotación pura)
episodes = 20000       # Número de episodios de entrenamiento
```

**Interpretación de Parámetros**:

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `α = 0.2` | Bajo | Cambios lentos a Q-table (estable) |
| `γ = 0.9` | Alto | Valora mucho recompensas futuras |
| `ε = 0.8` | Alto | Mucha exploración inicial (80% random) |

### 3.4 Convergencia de Q-Learning

Durante entrenamiento, la curva de recompensas promedio mostrada en `train_rl.py` debería:

```
Recompensa Promedio
       │
     0 ├─────────────────── (sin asignar recursos: worst)
       │
  -500 ├───┐
       │   │     ╱╱
  -700 ├───┼────╱ (convergencia)
       │   │╱
 -1000 ├───┘     
       │
       └─────────────────────► Episodes
       0      5000    10000    15000    20000
```

**Interpretación**:
- Episodios tempranos: baja recompensa (estrategia aleatoria)
- Episodios posteriores: recompensa mejora (aprendizaje)
- Final: recompensa converge (política óptima encontrada)

---

## 4. Implementación del Agente RL

### 4.1 Estructura de Archivos para Entrenamiento

```
PES/ext/
├─ pandemic.py
│  └─ Clase Pandemic(gym.Env): Entorno
│  └─ Función QLearning(): Algoritmo Q-Learning
│  └─ Función run_experiment(): Evaluación
│
├─ train_rl.py
│  └─ Script de entrenamiento
│  └─ Genera: inputs/q.npy, inputs/rewards.npy
│
└─ tools.py
   └─ Utilidades (gráficos, conversiones)
```

### 4.2 Clase Pandemic (OpenAI Gym)

Referencia: `ext/pandemic.py:42-330`

```python
class Pandemic(gym.Env):
    """
    Entorno OpenAI Gym para el escenario de pandemia.
    
    Interfaz estándar Gym:
    - observation_space: Box (41, 13, 11)
    - action_space: Discrete(11)
    - reset(): inicializa estado
    - step(action): ejecuta acción, retorna (obs, reward, done, info)
    """
    
    def __init__(self):
        # Definición de espacios
        self.observation_space = spaces.Box(...)  # Estado continuo
        self.action_space = spaces.Discrete(11)   # Acciones 0-10
        
        # Variables de ambiente
        self.severities = []           # Severidades actuales
        self.resources = []            # Recursos gastados
        self.available_resources = 40  # Presupuesto restante
    
    def reset(self):
        """Reinicia el entorno para una nueva secuencia."""
        self.severities = []
        self.resources = []
        self.available_resources = 40
        new_severity = self.new_city()  # Primera ciudad
        self.severities.append(new_severity)
        return [self.available_resources, 0, int(new_severity)]
    
    def step(self, action):
        """Ejecuta una acción y retorna nueva observación."""
        # 1. Clip acción a recursos disponibles
        action = min(action, self.available_resources)
        self.resources.append(action)
        self.available_resources -= action
        
        # 2. Actualizar severidades
        self.severities = get_updated_severity(
            len(self.severities),
            self.resources,
            self.severities
        )
        
        # 3. Calcular recompensa (negativa de severidad total)
        reward = -sum(self.severities)
        
        # 4. Determinar si secuencia terminó
        done = (self.iteration >= self.seq_length)
        
        # 5. Siguiente ciudad
        if not done:
            new_severity = self.new_city()
            self.severities.append(new_severity)
        
        # 6. Retornar observación
        obs = [self.available_resources, self.iteration, int(new_severity)]
        return obs, reward, done, {}
```

### 4.3 Función Q-Learning

Referencia: `ext/pandemic.py:220-270`

```python
def QLearning(env, alpha, gamma, epsilon, epsilon_min, episodes):
    """
    Entrena un Q-Learning agent en el entorno Pandemic.
    
    Parámetros:
        alpha: learning rate (0.2 por defecto)
        gamma: discount factor (0.9 por defecto)
        epsilon: exploración inicial (0.8 por defecto)
        epsilon_min: exploración mínima (0 por defecto)
        episodes: número de episodios (20000 por defecto)
    
    Retorna:
        rewards: histórico de recompensas por episodio
        Q: tabla Q entrenada (shape: (5863, 11))
        confidences: histórico de confianzas
    """
    
    # Inicializar Q-table
    Q = numpy.zeros((
        env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2],
        env.action_space.n
    ))
    
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_idx = tuple(state)  # Convertir observación a índice
            
            # ε-greedy: exploración vs explotación
            if random.random() < epsilon:
                action = env.action_space.sample()  # Random
            else:
                action = numpy.argmax(Q[state_idx])  # Óptimo
            
            # Ejecutar acción
            next_state, reward, done, _ = env.step(action)
            next_state_idx = tuple(next_state)
            
            # Bellman Update
            current_Q = Q[state_idx][action]
            max_next_Q = numpy.max(Q[next_state_idx])
            
            Q[state_idx][action] = current_Q + alpha * (
                reward + gamma * max_next_Q - current_Q
            )
            
            state = next_state
            episode_reward += reward
        
        # Reducir ε gradualmente (menos exploración)
        epsilon = max(epsilon_min, epsilon - (epsilon / episodes))
        
        rewards.append(episode_reward)
    
    return rewards, Q
```

---

## 5. Inferencia del Agente (Ejecución)

### 5.1 Cargar Q-table Entrenado

Referencia: `src/pygameMediator.py:1400-1420`

```python
def provide_rl_agent_response(img, response_timeout, sequence_no, trial_no):
    """
    Genera respuesta del agente durante experimento.
    
    Carga Q-table previamente entrenado y realiza búsqueda
    de mejor acción para el estado actual.
    """
    
    global Q_table  # Cargar tabla Q global (cacheada)
    
    if Q_table is None:
        Q_path = os.path.join(INPUTS_PATH, 'q.npy')
        Q_table = numpy.load(Q_path)  # Cargar desde archivo
    
    # Construir estado actual
    obs = [available_resources, trial_number, current_severity]
    state = tuple(obs)
    
    # Obtener acción óptima de Q-table
    action = numpy.argmax(Q_table[state, :])  # Argmax de Q-values
    
    return action
```

### 5.2 Transformación de Acción a Respuesta

El `action` (0-10) se convierte a `response` (asignación real) a través de múltiples transformaciones:

**Paso 1: Metacognitive Confidence** (Agent.py:24-50)

```python
def agent_meta_cognitive(action, output_value_range, resources_left, response_timeout):
    """
    Mapea acción de RL a respuesta con confianza metacognitiva.
    
    La confianza se basa en la distancia a decisión más cercana:
    - Si está en el centro: alta confianza (1.0)
    - Si está en el borde: baja confianza (0.0)
    """
    
    # Centros de decisión (puntos de máxima confianza)
    centers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Distancia al centro más cercano
    closer = numpy.abs(numpy.array(centers) - action)
    closervalue = centers[numpy.argmin(closer)]
    
    # Clip acción a recursos disponibles
    action = numpy.clip(action, 1, resources_left)
    
    # Confianza inversamente proporcional a distancia
    distance = numpy.clip(abs(closervalue - action), 0, 0.5)
    confidence = distance * (-2) + 1  # Escala: 1.0 → 0.0
    
    # Respuesta redondeada
    response = int(round(action))
    
    # Tiempos realistas basados en distancia
    mu, sigma = int(distance * 10), 3
    rt_hold = numpy.random.normal(mu, sigma)
    rt_release = rt_hold + numpy.random.normal(mu, 1)
    
    return response, confidence, rt_hold, rt_release
```

**Interpretación**:
```
action = 7.3
centers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
distancias = [7.3, 6.3, 5.3, 4.3, 3.3, 2.3, 1.3, 0.3, 0.7, 1.7, 2.7]
closervalue = 7  (distancia mínima: 0.3)
distance = |7 - 7.3| = 0.3
confidence = 0.3 * (-2) + 1 = 0.4  (baja confianza)
response = round(7.3) = 7
```

**Paso 2: Response Decay con Ruido** (Agent.py:51-70)

```python
def adjust_response_decay(resp, decay, resources_left):
    """
    Añade variabilidad para simular comportamiento menos óptimo.
    
    decay: factor de decaimiento (0 = óptimo, 1 = aleatorio)
    """
    
    # Varianza proporcional a decaimiento
    variance = int((1.0 - decay) * AGENT_NOISE_VARIANCE)
    
    # Añadir ruido gaussiano
    delta = numpy.random.normal(resp, variance)
    
    # Clip a recursos disponibles
    response = numpy.clip(delta, 0, min(resources_left, MAX_ALLOCATABLE_RESOURCES))
    
    return response
```

**Decaimiento de Boltzmann**:

```python
def boltzmann_decay(global_seq_no):
    """
    Reduce ruido a lo largo del experimento (aprendizaje).
    
    T(seq) = exp(-75 / (4 * seq_number))
    
    T(seq=1) ≈ 1.0    (máximo ruido)
    T(seq=32) ≈ 0.5   (ruido medio)
    T(seq=64) ≈ 0.27  (mínimo ruido)
    """
    return numpy.exp(-75.0 / (4.0 * global_seq_no))
```

**Visualización**:

```
Decaimiento Boltzmann
       │
    1.0├────┐
       │     ╲
    0.5├─────╲────
       │      ╲
    0.0├───────────────────► seq_number
       0    20   40    60
```

### 5.3 Flujo Completo de Decisión

```
Estado actual: s = (25, 5, 7)
    │
    ▼
Q-table lookup: Q[s, :]
    │ Q-values para cada acción
    ▼
[0.5, 1.2, 2.1, 1.8, 3.5, 2.7, 1.9, 1.1, 0.8, 0.3, 0.1]
    │ Máximo Q-value
    ▼
action = 4 (argmax)
    │
    ▼
agent_meta_cognitive(4, ...)
    ├─ distance = 0.2
    ├─ confidence = 0.6
    └─ response ≈ 4
    │
    ▼
boltzmann_decay(seq_5) = 0.95
    │
    ▼
adjust_response_decay(4, 0.95, 25)
    ├─ variance = 1.0 * 0.95 = 0.95
    ├─ delta = normal(4, 0.95)
    └─ final_response = 4 ± ruido
    │
    ▼
Asignar ~4 recursos a ciudad
```

---

## 6. Ciclo Completo: Entrenamiento → Ejecución

### 6.1 Fase de Entrenamiento

```
1. Inicializar Q-table en ceros
   Q = zeros((5863, 11))  # Todos los estados × acciones

2. Para cada episodio (20000 iteraciones):
   
   a. Iniciar secuencia nueva
      state = env.reset()  # [40, 0, severidad_random]
   
   b. Para cada trial en secuencia:
      
      i.   ε-greedy: random o Q-argmax
           action = sample() si random < 0.8, else argmax
      
      ii.  Ejecutar en Pandemic
           reward = -sum(severities)
      
      iii. Bellman update
           Q[s, a] += 0.2 * (reward + 0.9*max(Q[s', :]) - Q[s, a])
      
      iv.  Siguiente estado
           state = next_state
   
   c. Guardar recompensa por episodio
      rewards.append(episode_total_reward)
   
   d. Reducir ε (menos exploración con tiempo)
      epsilon = max(0, epsilon - epsilon/20000)

3. Guardar Q-table entrenada
   numpy.save('inputs/q.npy', Q)
```

**Salida**: Archivo `inputs/q.npy` (tabla de decisiones óptimas)

### 6.2 Fase de Ejecución del Experimento

```
1. Cargar Q-table entrenada
   Q = numpy.load('inputs/q.npy')

2. Para cada bloque (8):
   
   a. Para cada secuencia (8):
      
      i.   Inicializar estado
           resources_left = 40
           severities = [init1, init2, init3]  # 3 ciudades iniciales
      
      ii.  Para cada trial (3-10 ciudades adicionales):
           
           - State = (resources_left, trial_num, severity_actual)
           - Action = argmax(Q[State, :])
           - Meta-cognitive transform → Confidence
           - Boltzmann decay → Variabilidad
           - Gaussian noise → Response
           - Guardar en CSV: (severity, response, confidence, times)
           - Actualizar severities para siguiente trial
      
      iii. Calcular Performance
           final_severity = sum(severities_finales)
           perf = 1 - (final_severity / initial_severity_sum)
           MyPerformances.append(perf)

3. Generar gráficos y estadísticas
```

---

## 7. Optimización y Mejora

### 7.1 Variaciones de Q-Learning

El código actual usa **Q-Learning clásico**, pero hay variantes más avanzadas:

| Variante | Ventaja | Desventaja |
|----------|---------|-----------|
| Q-Learning | Simple, convergencia garantizada | Lento, requiere muchos episodios |
| Double Q-Learning | Reduce sobrestimación | Más complejo |
| Dueling DQN | Mejor para estados complejos | Requiere red neuronal |
| Policy Gradient | Mejor para espacios continuos | Varianza alta |

### 7.2 Mejoras Potenciales

```python
# Actual
if random() < epsilon:
    action = random()
else:
    action = argmax(Q[s, :])

# Mejora: Softmax exploration
probabilities = softmax(Q[s, :] / temperature)
action = sample_from(probabilities)
```

### 7.3 Hiperparámetros Sugeridos

```python
# Conservador (aprendizaje lento, convergencia estable)
alpha = 0.1
gamma = 0.95
epsilon = 0.5
episodes = 50000

# Agresivo (aprendizaje rápido, menos estable)
alpha = 0.5
gamma = 0.8
epsilon = 0.9
episodes = 10000

# Actual (balanceado)
alpha = 0.2
gamma = 0.9
epsilon = 0.8
episodes = 20000
```

---

## 8. Análisis del Aprendizaje

### 8.1 Curva de Recompensas

Durante `train_rl.py`, se genera gráfico de recompensas promedio vs episodios:

```
Reward vs Episodes
     0 ├────────────────────
       │
 -500  ├────┐
       │    │╲
-1000  ├────┼─╲──────────
       │    │  ╲
-1500  ├────┼───╲───╲
       │    │   │   │╲
-2000  ├────┼───┼───┼─╲────
       │    │   │   │  ╲
       └────┴───┴───┴───┴──► Episodes
       0  5K  10K  15K  20K
       
Fases:
1. (0-2K): Aprendizaje rápido (estrategia aleatoria)
2. (2K-15K): Mejora gradual (refinamiento)
3. (15K-20K): Convergencia (saturación de aprendizaje)
```

### 8.2 Evaluación del Agente

Después de entrenamiento, ejecutar:

```python
# Evaluar agente entrenado
def evaluate_trained_agent(Q_table, num_episodes=100):
    env = Pandemic()
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = argmax(Q_table[state, :])  # Explotación pura
            state, reward, done, _ = env.step(action)
            total_reward += reward
    
    avg_reward = total_reward / num_episodes
    print(f"Average Reward (Trained): {avg_reward}")
    return avg_reward

# Comparar con agente aleatorio
def evaluate_random_agent(num_episodes=100):
    env = Pandemic()
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = random.randint(0, 10)  # Aleatorio
            state, reward, done, _ = env.step(action)
            total_reward += reward
    
    avg_reward = total_reward / num_episodes
    print(f"Average Reward (Random): {avg_reward}")
    return avg_reward
```

**Métrica de Mejora**:
```
Improvement = (Trained - Random) / |Random| * 100%
```

---

## 9. Troubleshooting

### Problema: Q-table no converge

**Causas**:
1. Learning rate muy alto → actualiza demasiado
2. Learning rate muy bajo → aprende lentamente
3. Recompensa no normalizada → escala inconsistente

**Soluciones**:
```python
# Normalizar recompensas
reward = -sum(severities) / max_severity  # Escala [-1, 0]

# Ajustar learning rate
alpha = 0.1  # Si anterior divergía
alpha = 0.5  # Si anterior convergía lentamente
```

### Problema: Agent toma acciones aleatorias durante ejecución

**Causa**: Q-table no cargado o archivo no existe

**Verificación**:
```bash
ls -l PES/inputs/q.npy  # Debe existir
file PES/inputs/q.npy   # Debe ser archivo válido numpy
```

### Problema: Performance muy baja (< 0.2)

**Causas**:
1. Q-table mal entrenado
2. Parámetros de pandemia incorrectos
3. Configuración de ruido demasiado alta

**Soluciones**:
```python
# Re-entrenar con más episodios
python3 -m PES.ext.train_rl
# (ejecutar con episodes=50000 en lugar de 20000)

# Verificar PANDEMIC_PARAMETER en CONFIG.py
PANDEMIC_PARAMETER = 0.4  # Por defecto

# Reducir AGENT_NOISE_VARIANCE
AGENT_NOISE_VARIANCE = 1.0  # En lugar de 2.0
```

---

## 10. Comparación con Alternativas

### 10.1 Q-Learning vs Policy Gradient

| Aspecto | Q-Learning (Actual) | Policy Gradient |
|--------|------------------|-----------------|
| Convergencia | Garantizada | Puede divergir |
| Datos necesarios | 20,000 episodios | 5,000-10,000 |
| Complejidad | Simple | Más compleja |
| Espacios continuos | Discretizar (aquí: 0-10) | Nativo |
| Interpretabilidad | Q-table legible | Red neuronal (caja negra) |

### 10.2 Q-Learning vs Deep Q-Network (DQN)

```
Q-Learning (Actual):
├─ Representación: Tabla explícita
├─ Tamaño: 5863 × 11 = 64,493 valores
├─ Memoria: ~250 KB (numpy array)
└─ Tiempo entrenamiento: 5-10 minutos

Deep Q-Network (DQN):
├─ Representación: Red neuronal
├─ Tamaño: 100-10,000+ parámetros (dependiendo arquitectura)
├─ Memoria: ~1-100 MB
└─ Tiempo entrenamiento: 30-120 minutos
```

**Decisión de usar Q-Learning**: Espacio de estados suficientemente pequeño (5863) para tabla explícita, evitando complejidad de redes neuronales.

---

## 11. Extensiones Futuras

### 11.1 Multi-Agent RL

Actualmente singleagent. Posible extensión:

```python
# Múltiples agentes aprendiendo simultáneamente
agents = [Agent(i) for i in range(num_agents)]

for episode in range(episodes):
    states = env.reset()
    
    for agent in agents:
        action = agent.select_action(state)
        reward = calculate_reward(action, states)
        agent.update_Q_table(...)
```

### 11.2 Transfer Learning

Usar Q-table entrenado en diferente PANDEMIC_PARAMETER:

```python
# Entrenar con PANDEMIC_PARAMETER=0.4
Q_04 = train_rl(0.4)  # 20,000 episodios

# Fine-tune para PANDEMIC_PARAMETER=0.6
Q_06 = Q_04  # Inicializar con entrenamiento previo
Q_06 = train_rl(0.6, initial_Q=Q_06, episodes=5000)  # Menos episodios
```

### 11.3 Asistencia Online

Aprender durante ejecución (no recomendado para experimento, pero posible para otras aplicaciones):

```python
# Durante experimento, actualizar Q-table en tiempo real
for trial in range(trials):
    action = argmax(Q[state])
    state_new, reward, done = env.step(action)
    
    # Update on-the-fly
    Q[state, action] += alpha * (reward + gamma * max(Q[state_new]) - Q[state, action])
    
    state = state_new
```

---

## Conclusión

El agente RL en PES implementa **Q-Learning clásico** con refinamientos metacognitivos (confianza, ruido humanizado, decaimiento Boltzmann). La arquitectura es clara, eficiente y permite experimentos reproducibles.

Para profundizar en teoría subyacente, ver `RL_THEORY.md`.

Para entender la simulación completa, ver `HOWTO_PES.md`.
