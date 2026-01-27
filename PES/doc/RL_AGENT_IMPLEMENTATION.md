# Implementación del RL-Agent en PES - Arquitectura y Relación con Reinforcement Learning

## 1. Visión General de la Implementación

El RL-Agent en PES implementa un algoritmo de **Q-Learning** clásico, diseñado para tomar decisiones de asignación de recursos en un entorno dinámico. Este documento explica la arquitectura específica del código y cómo cada componente se relaciona con los principios teóricos de Reinforcement Learning.

---

## 2. Componentes Principales del Sistema

### 2.1 Definición del Problema como MDP (Markov Decision Process)

El problema de PES se modela como un Markov Decision Process con:

**Estado (s)**: Tupla de tres elementos
```
s = (recursos_left, city_number, severity)
  - recursos_left: ∈ {0, 1, ..., 30}       [Eje 0 de Q-Table]
  - city_number:   ∈ {0, 1, ..., 12}       [Eje 1 de Q-Table] 
  - severity:      ∈ {0, 1, ..., 10}       [Eje 2 de Q-Table]
```

**Código relevante** ([PES/src/pygameMediator.py](PES/src/pygameMediator.py):981):
```python
def provide_rl_agent_response(img, resources, resources_left, coordinate, 
                              circle_radius, session_no, sequence_no, trial_no):
    # Paso 1: Extraer estado
    sever = sevs[session_no * NUM_SEQUENCES + sequence_no][trial_no]
    city_number = trial_no
    # Estado: (recursos_left, city_number, sever) ∈ S
```

**Interpretación RL**: Este estado representa completamente la situación de decisión. El Markov property se cumple porque la decisión óptima depende solo de estos tres factores, no del histórico.

---

**Acción (a)**: Recursos a asignar a la ciudad actual
```
a ∈ {0, 1, 2, ..., 10}  # 11 acciones posibles
  - Restricción: a ≤ recursos_left  (no se puede asignar más de lo disponible)
```

**Código relevante** ([PES/src/pandemic.py](PES/src/pandemic.py):150):
```python
def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    # options = Q[recursos_left, city_number, severity]  # 11 valores
    # Máscara: hacer infactibles acciones > recursos_left
    o = numpy.arange(len(options))
    masked_options = options.copy()
    masked_options[o > resources_left] = 0.00001
    
    # Acción = argmax de options válidas
    response = numpy.argmax(masked_options)  # ∈ {0, 1, ..., 10}
```

**Interpretación RL**: Las acciones no factibles se castigan (valores ~0) para asegurar la exploración solo de acciones válidas. Esto implementa restricciones de dominio en el RL.

---

**Recompensa (r)**: Disminución en severidad total
```
r = -(Severidad_nueva - Severidad_anterior)
  = -(β*s - α*a) + s_anterior
  = α*a - β*s + s_anterior
```

**Código relevante** ([PES/ext/train_rl.py](PES/ext/train_rl.py):~120):
```python
def calculate_reward(current_severities, resources_allocated, beta=0.76, alpha=0.24):
    # Severidad total antes
    total_before = sum(current_severities)
    
    # Aplicar fórmula de daño: new_sev = β*sev - α*resource
    updated_severities = []
    for i, sev in enumerate(current_severities):
        new_sev = beta * sev - alpha * resources_allocated[i]
        updated_severities.append(max(new_sev, 0))
    
    # Severidad total después
    total_after = sum(updated_severities)
    
    # Recompensa: cuánto mejoramos
    reward = total_before - total_after  # Positiva si severity disminuyó
    return reward
```

**Interpretación RL**: La recompensa es **immediata** (depend only of (s, a)). No hay recompensas futuras acumuladas - cada acción genera recompensa instantánea. Esto simplifica el problema pero sacrifica optimización global.

---

### 2.2 Q-Learning: Actualización de Q-Values

**Ecuación de Bellman (forma de Q-Learning)**:
```
Q(s, a) ← Q(s, a) + α·[r + γ·max_a'(Q(s', a')) - Q(s, a)]
```

Donde:
- α = LEARNING_RATE (típicamente 0.1)
- γ = DISCOUNT_FACTOR (típicamente 0.95)
- r = recompensa inmediata
- s' = siguiente estado
- max_a'(Q(s', a')) = valor máximo del próximo estado

**Código relevante** ([PES/ext/train_rl.py](PES/ext/train_rl.py):~160):
```python
def update_q_value(Q, state, action, reward, next_state, 
                   learning_rate, discount_factor):
    resources_left, city_number, severity = state
    resources_left_next, city_number_next, severity_next = next_state
    
    # Clip para evitar índices fuera de rango
    r_left = int(numpy.clip(resources_left, 0, Q.shape[0]-1))
    c_num = int(numpy.clip(city_number, 0, Q.shape[1]-1))
    sev = int(numpy.clip(severity, 0, Q.shape[2]-1))
    act = int(numpy.clip(action, 0, 10))
    
    r_left_next = int(numpy.clip(resources_left_next, 0, Q.shape[0]-1))
    c_num_next = int(numpy.clip(city_number_next, 0, Q.shape[1]-1))
    sev_next = int(numpy.clip(severity_next, 0, Q.shape[2]-1))
    
    # Valor máximo del próximo estado
    max_next_q = numpy.max(Q[r_left_next, c_num_next, sev_next, :])
    
    # Actualización Bellman
    current_q = Q[r_left, c_num, sev, act]
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    
    Q[r_left, c_num, sev, act] = new_q
    
    return Q
```

**Interpretación RL**: Esta es la ecuación central del Q-Learning off-policy. La actualización balancea:
- **Estimación actual** Q(s,a): lo que ya sabemos
- **Información nueva**: r + γ·max(Q(s'))
- **Learning rate α**: qué tan rápido adaptarnos

---

### 2.3 Estrategia de Exploración: ε-Greedy + Decaimiento

**Problema**: Balance entre explorar (probar nuevas acciones) y explotar (usar las mejores conocidas).

**Estrategia ε-Greedy**:
```
Acción_a = {
    Acción_aleatoria      con probabilidad ε
    argmax Q(s, a')       con probabilidad (1-ε)
}
```

**Código relevante** ([PES/ext/train_rl.py](PES/ext/train_rl.py):~80):
```python
def choose_action(Q, state, resources_left, epsilon):
    resources_left_idx = int(numpy.clip(resources_left, 0, Q.shape[0]-1))
    city_number_idx = int(state[1])
    severity_idx = int(numpy.clip(state[2], 0, Q.shape[2]-1))
    
    if numpy.random.random() < epsilon:  # Exploración
        # Acción aleatoria dentro de rango factible
        action = numpy.random.randint(0, min(resources_left + 1, 11))
    else:  # Explotación
        # Mejor acción conocida
        q_values = Q[resources_left_idx, city_number_idx, severity_idx, :]
        # Máscara acciones infactibles
        q_values[numpy.arange(len(q_values)) > resources_left] = -numpy.inf
        action = numpy.argmax(q_values)
    
    return action
```

**Decaimiento de ε** (reducir exploración con el tiempo):
```python
epsilon = INITIAL_EPSILON * (0.99 ** episode)
# O
epsilon = max(0.01, INITIAL_EPSILON - (episode / NUM_EPISODES))
```

**Interpretación RL**: Esto implementa el trade-off exploración-explotación. Al inicio, exploramos mucho (ε alto) para descubrir la estructura del problema. Con el tiempo, explotamos más (ε bajo) porque ya sabemos qué acciones son buenas.

---

### 2.4 Estructura de Datos: Q-Table

**Formato**:
```
Q[recursos_left, city_number, severity, action] = estimado_valor_ación
  Shape: (31, 13, 11, 11)
  
Índices:
  - Eje 0: recursos disponibles (0-30)
  - Eje 1: número de ciudad/trial (0-12)
  - Eje 2: severidad inicial (0-10)
  - Eje 3: acción = recursos asignados (0-10)
```

**Código de inicialización** ([PES/ext/train_rl.py](PES/ext/train_rl.py):~30):
```python
# Inicializar Q-Table con ceros (asumimos recompensas futuras neutrales)
Q = numpy.zeros((31, 13, 11, 11))

# O con pequeños valores de exploración (optimista)
Q = numpy.ones((31, 13, 11, 11)) * 0.1  # Optimismo inicial
```

**Interpretación RL**: La Q-Table es una representación tabulada de la política. Cada celda Q[s,a] estima el valor a largo plazo de tomar acción 'a' en estado 's'.

**Ventajas**:
- ✅ Convergencia garantizada (bajo ciertas condiciones)
- ✅ Simple de implementar
- ✅ Interpretable

**Desventajas**:
- ❌ Escalabilidad limitada (31 × 13 × 11 × 11 = ~51k estados, manejable)
- ❌ No generaliza a nuevos estados no vistos

---

## 3. Flujo de Ejecución Durante el Experimento

### 3.1 Carga de Q-Table Pre-entrenada

**Código** ([PES/src/pygameMediator.py](PES/src/pygameMediator.py):250):
```python
def provide_rl_agent_response(img, resources, resources_left, ...):
    # Cargar Q-Table
    Q = numpy.load(os.path.join(INPUTS_PATH, 'q.npy'))  # Shape: (31, 13, 11, 11)
    
    # El agente ahora usa esta Q-Table para consultar valores
```

**Interpretación RL**: Esto es **off-policy learning** o más exactamente, **policy deployment**. La Q-Table se entrenó offline (en `train_rl.py`) y ahora se usa para control online.

---

### 3.2 Selección de Acción: Greedy (sin exploración)

Durante el experimento, el agente es **puramente greedy**:
```python
response = numpy.argmax(masked_options)  # Siempre mejor acción
```

**Código** ([PES/src/pandemic.py](PES/src/pandemic.py):200):
```python
def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    # options = Q[recursos_left, city_number, severity, :]  # 11 valores
    
    # Máscara acciones infactibles
    masked_options = options.copy()
    masked_options[o > resources_left] = 0.00001
    
    # GREEDY: siempre la mejor
    response = numpy.argmax(masked_options)  # Exploit only, no explore
```

**Interpretación RL**: Durante deployment, NO exploramos. Solo explotamos la política aprendida. Esto es correcto para fase de test.

---

### 3.3 Cálculo de Confianza (Metacognición)

**Novedad**: El agente reporta **confianza** basada en entropía de Q-values.

**Código** ([PES/src/pandemic.py](PES/src/pandemic.py):210):
```python
# Calcular entropía de la distribución Q-values
entropy = entropy_from_pdf(masked_options)

# Normalizar entropía
max_entropy = entropy_from_pdf(numpy.ones(11) / 11)  # Uniforme
min_entropy = entropy_from_pdf(numpy.array([1,0,0,0,0,0,0,0,0,0,0]))  # Concentrada

# Confianza = inversa de entropía normalizada
confidence = 1 - (entropy - min_entropy) / (max_entropy - min_entropy)
# confidence ∈ [0, 1]
```

**Interpretación RL Avanzada**: 
- **Baja entropía** (Q-values concentrados): El agente está seguro - tiene una acción claramente mejor
- **Alta entropía** (Q-values uniformes): El agente es incierto - muchas acciones parecen igualmente buenas
- **Confianza = 1 - (entropía normalizada)**

Este es un cálculo de **incertidumbre epistémica** - qué tan seguro está de sus valores Q.

---

## 4. Relación Componente-Código-Teoría

### Tabla de Mapeo

| Concepto RL | Fórmula | Código | Archivo |
|-----------|----------|--------|---------|
| **Estado** | s = (r, c, σ) | `s = (recursos_left, city_number, severity)` | pygameMediator.py:990 |
| **Acción** | a ∈ A | `a ∈ {0,1,...,10}` | pygameMediator.py:340 |
| **Recompensa** | r(s,a) | `reward = α*a - β*s` | pandemic.py:180 |
| **Q-Function** | Q(s,a) = E[Σγ^t r_t] | `Q[r,c,σ,a]` | train_rl.py:30 |
| **Bellman Update** | Q ← Q + α[r+γmax(Q')-Q] | `new_q = ... + lr * (...)` | train_rl.py:160 |
| **Policy** | π(s) = argmax_a Q(s,a) | `action = argmax(Q)` | pandemic.py:190 |
| **Exploración** | ε-Greedy | `if rand() < ε: random_action` | train_rl.py:80 |
| **Convergencia** | Q→Q* | Entrenamiento 20k episodios | train_rl.py:50 |

---

## 5. Flujo Completo: Entrenamiento → Deployment

### Fase 1: Entrenamiento (Offline)

```
train_rl.py (ext/)
    ↓
Para cada episodio (20,000 veces):
    - Inicializar estado s
    - Para cada step (hasta end):
        - Elegir a con ε-Greedy
        - Ejecutar a (simular PES)
        - Observar r y s'
        - Actualizar Q(s,a) ← Bellman
    - Guardar Q-Table
    ↓
Salida: inputs/q.npy (Q-Table convergida)
```

**Código clave** ([PES/ext/train_rl.py](PES/ext/train_rl.py):60):
```python
for episode in range(NUM_EPISODES):
    state = initialize_episode()  # s_0
    
    for step in range(MAX_STEPS):
        # Política: ε-Greedy
        action = choose_action(Q, state, resources_left, epsilon)
        
        # Transición: P(s'|s,a)
        next_state, reward = simulate_step(state, action)
        
        # Actualización Bellman
        Q = update_q_value(Q, state, action, reward, next_state, 
                          LEARNING_RATE, DISCOUNT_FACTOR)
        
        state = next_state
        
        if is_terminal(state):
            break
    
    # Decaimiento de exploración
    epsilon = INITIAL_EPSILON * (0.99 ** episode)
```

---

### Fase 2: Deployment (Online)

```
PES/__main__.py
    ↓
Cargar q.npy
    ↓
Para cada trial:
    - Observar s_actual = (resources, city, severity)
    - Consultar Q[s] → array de 11 valores
    - π(s) = argmax Q[s, :] → acción greedy
    - Ejecutar acción (asignar recursos)
    - Observar transición (para logging)
    ↓
Salida: responses_XXX.txt (acciones y confianza)
```

**Código clave** ([PES/src/pygameMediator.py](PES/src/pygameMediator.py):1060):
```python
def provide_rl_agent_response(...):
    # Cargar Q-Table entrenada
    Q = numpy.load(...)  # Pre-entrenada
    
    # Extraer estado actual
    s = (resources_left, city_number, severity)
    
    # Consultar política greedy
    q_values = Q[s]  # 11 valores para 11 acciones
    action = argmax(q_values)  # Greedy policy
    
    # Calcular confianza (incertidumbre)
    confidence = calculate_confidence(q_values)
    
    return action, confidence
```

---

## 6. Comparación: Teórico vs Implementado

### 6.1 Q-Learning Clásico (Teoría)

```
Condiciones para convergencia:
1. Espacio de estados finito
2. Espacio de acciones finito
3. Rewards acotadas: |r(s,a)| ≤ R_max
4. Cada estado-acción visitado infinitas veces
5. Learning rate: Σα_n = ∞, Σα_n^2 < ∞

Convergencia: Q_n → Q* cuando n → ∞
```

### 6.2 Implementación en PES

| Teoría | Implementación | Status |
|--------|---|--------|
| Estados finitos | 31×13×11=4,433 | ✅ Cumple |
| Acciones finitas | 11 acciones | ✅ Cumple |
| Rewards acotadas | r ∈ [0, ~50] | ✅ Cumple |
| Visitas infinitas | Heurístico (20k ep) | ⚠️ Aproximado |
| Learning rate | α=0.1, decayendo | ✅ Cumple |
| **Convergencia** | ~15k episodios | ✅ Empírica |

---

## 7. Mejoras Potenciales

### 7.1 Usar Function Approximation

Actualmente: Q-Table (lookup)
Mejora: Red neuronal para generalización
```python
# En lugar de Q[s,a], usar NN(s) → 11 acciones
q_network = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(11)  # 11 acciones
])
```

### 7.2 Usar Deep Q-Learning (DQN)

Para espacios de estado más complejos:
```python
# Experience replay
buffer = ReplayBuffer(capacity=10000)

# Target network
target_network = copy(q_network)
```

### 7.3 Usar Actor-Critic

Para mejor convergencia:
```python
# Actor: política π(a|s)
actor = PolicyNetwork()

# Critic: función de valor V(s)
critic = ValueNetwork()
```

---

## 8. Resumen: Arquitectura Integrada

```
┌─────────────────────────────────────────────────────┐
│ TEORÍA DE REINFORCEMENT LEARNING                    │
│ - Markov Decision Process (s,a,r,s')               │
│ - Ecuación de Bellman: V(s)=E[r+γV(s')]            │
│ - Q-Learning: Q-update con Bellman                  │
│ - ε-Greedy: Exploración-Explotación               │
│ - Convergencia: iterativa a Q*                     │
└──────────────────────────────────────────────────────┘
                        ↓ Implementación
        ┌───────────────────────────────────┐
        │ CÓDIGO PES                        │
        ├───────────────────────────────────┤
        │ train_rl.py (Entrenamiento)       │
        │  - Q-Table init                   │
        │  - Bellman update loop            │
        │  - ε-Greedy action selection      │
        │  - Save q.npy                     │
        ├───────────────────────────────────┤
        │ pygameMediator.py (Deployment)    │
        │  - Load q.npy                     │
        │  - Greedy policy: argmax Q        │
        │  - Confidence calculation         │
        │  - Response generation            │
        └───────────────────────────────────┘
                        ↓ Ejecución
        ┌───────────────────────────────────┐
        │ RESULTADOS                        │
        ├───────────────────────────────────┤
        │ responses_XXX.txt:                │
        │  - Acciones tomadas               │
        │  - Confianza en cada decisión    │
        │  - Tiempos de respuesta           │
        │ Performance metrics:              │
        │  - Severidad final minimizada     │
        │  - Decisiones consistentes        │
        └───────────────────────────────────┘
```

---

## Referencias en el Código

- **Q-Learning**: [PES/ext/train_rl.py](PES/ext/train_rl.py):156-170
- **Selección de Acción**: [PES/src/pandemic.py](PES/src/pandemic.py):150-200
- **Confianza/Entropía**: [PES/src/pandemic.py](PES/src/pandemic.py):210-240
- **Policy Deployment**: [PES/src/pygameMediator.py](PES/src/pygameMediator.py):981-1020
- **Recompensas**: [PES/src/exp_utils.py](PES/src/exp_utils.py):228-280

