# Theoretical Foundations of Reinforcement Learning

## Introducción

Este documento desarrolla los conceptos teóricos fundamentales de **Reinforcement Learning (RL)** necesarios para comprender el agente implementado en PES. Se enfoca en la teoría formal, con referencias concretas a la implementación.

**Audiencia**: Científicos de datos con conocimiento de probabilidad y cálculo, interesados en entender RL desde principios.

---

## 1. Conceptos Fundamentales de RL

### 1.1 El Problema General de Aprendizaje por Refuerzo

En RL, un **agente** interactúa con un **entorno** mediante:

1. **Observar estado** $s \in \mathcal{S}$ (información actual del mundo)
2. **Ejecutar acción** $a \in \mathcal{A}$ (decisión del agente)
3. **Recibir recompensa** $r \in \mathbb{R}$ (feedback del ambiente)
4. **Transicionar a nuevo estado** $s' \in \mathcal{S}$ (evolución del mundo)

**Objetivo**: Encontrar una **política** $\pi$ que maximice la **recompensa acumulada**.

### 1.2 Definición Formal: Markov Decision Process (MDP)

Un MDP se define por la tupla $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

| Elemento | Símbolo | Descripción |
|---------|---------|------------|
| Espacio de Estados | $\mathcal{S}$ | Conjunto de observaciones posibles |
| Espacio de Acciones | $\mathcal{A}$ | Conjunto de decisiones posibles |
| Modelo de Transición | $\mathcal{P}(s'\ \vert\ s, a)$ | Probabilidad de ir de $s$ a $s'$ con acción $a$ |
| Función de Recompensa | $\mathcal{R}(s, a, s')$ | Recompensa por transición |
| Factor de Descuento | $\gamma \in [0, 1)$ | Importancia relativa de recompensas futuras |

**Propiedad Markoviana**: El futuro depende únicamente del estado actual, no del historial:

$$P(s_{t+1}\ \vert\ s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1}\ \vert\ s_t, a_t)$$

### 1.3 Aplicación al PES

En el Pandemic Experiment Scenario:

$$\mathcal{S} = \{\text{(recursos disponibles, número trial, severidad actual)}\}$$
$$\mathcal{S} = \mathbb{R}^3_{[0,40] \times [0,12] \times [0,10]}$$

$$\mathcal{A} = \{0, 1, 2, \ldots, 10\}$$
$$\text{(recursos a asignar)}$$

$$\mathcal{R}(s, a, s') = -\sum_{i} \text{severity}_i'$$
$$\text{(negativa de severidad total)}$$

$$\gamma = 0.9$$
$$\text{(futuro es 90% tan importante como presente)}$$

---

## 2. Políticas y Value Functions

### 2.1 Política $\pi$

Una **política** es una estrategia que mapea estados a acciones:

$$\pi: \mathcal{S} \to \mathcal{A}$$

**Dos tipos**:

1. **Política Determinista**: $a = \pi(s)$ → acción fija por estado
   ```python
   action = argmax(Q[state, :])  # Siempre el mejor
   ```

2. **Política Estocástica**: $\pi(a \vert s)$ → distribución de probabilidad
   ```python
   probs = softmax(Q[state, :] / temperature)
   action = sample(probs)  # Aleatorio pero ponderado
   ```

**Objetivo**: Encontrar política óptima $\pi^*$ que maximiza recompensa acumulada.

### 2.2 Value Functions

#### 2.2.1 State Value Function $V^{\pi}(s)$

Valor esperado de estar en estado $s$ bajo política $\pi$:

$$V^{\pi}(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t \bigg\vert s_0 = s\right]$$

**Interpretación**: "Si estoy en estado $s$ y sigo política $\pi$, ¿cuánta recompensa total puedo esperar?"

**Ecuación de Bellman para $V$**:

$$V^{\pi}(s) = \sum_a \pi(a \vert s) \sum_{s'} \mathcal{P}(s' \vert s, a) [\mathcal{R}(s,a,s') + \gamma V^{\pi}(s')]$$

**Descomposición**:
```
V(s) = Σ π(a|s) [ recompensa_inmediata + γ * V(s') ]
       acciones
```

#### 2.2.2 Action Value Function $Q^{\pi}(s, a)$

Valor esperado de ejecutar acción $a$ en estado $s$, luego seguir política $\pi$:

$$Q^{\pi}(s, a) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t \bigg\vert s_0 = s, a_0 = a\right]$$

**Interpretación**: "Si estoy en estado $s$ y hago acción $a$, ¿cuánta recompensa espero?"

**Ecuación de Bellman para $Q$**:

$$Q^{\pi}(s, a) = \sum_{s'} \mathcal{P}(s' \vert s, a) [\mathcal{R}(s,a,s') + \gamma V^{\pi}(s')]$$

**Relación con $V$**:

$$V^{\pi}(s) = \sum_a \pi(a \vert s) Q^{\pi}(s, a)$$

### 2.3 Optimal Value Functions

$$V^*(s) = \max_{\pi} V^{\pi}(s)$$
$$Q^*(s, a) = \max_{\pi} Q^{\pi}(s, a)$$

**Bellman Optimality Equation para $V^*$**:

$$V^*(s) = \max_a \sum_{s'} \mathcal{P}(s' \vert s, a) [\mathcal{R}(s,a,s') + \gamma V^*(s')]$$

**Bellman Optimality Equation para $Q^*$**:

$$Q^*(s, a) = \sum_{s'} \mathcal{P}(s' \vert s, a) [\mathcal{R}(s,a,s') + \gamma \max_{a'} Q^*(s', a')]$$

**Política Óptima**:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

---

## 3. Métodos de Iteración sobre Valores

### 3.1 Value Iteration

**Algoritmo**:

Inicializar $V(s) = 0$ para todo $s$

Repetir hasta convergencia:
$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} \mathcal{P}(s' \vert s, a) [\mathcal{R}(s,a,s') + \gamma V_k(s')]$$

**Convergencia**: Garantizada en $O(\vert \mathcal{S} \vert^2 \vert \mathcal{A} \vert)$ iteraciones

**Pseudocódigo**:

```python
V = {s: 0 for s in states}
θ = 0.001  # Precisión

while True:
    delta = 0
    for s in states:
        v_old = V[s]
        V[s] = max([
            sum(P[s'][s, a] * (R[s, a, s'] + gamma * V[s']) 
                for s' in states)
            for a in actions
        ])
        delta = max(delta, |v_old - V[s]|)
    
    if delta < θ:
        break

return V
```

### 3.2 Policy Iteration

**Algoritmo**:

Inicializar política arbitraria $\pi$

Repetir:
1. **Policy Evaluation**: Calcular $V^{\pi}(s)$ para actual $\pi$
2. **Policy Improvement**: $\pi' \leftarrow \arg\max_a Q^{\pi}(s, a)$

**Convergencia**: Típicamente más rápido que value iteration

**Comparación**:

| Método | Iteraciones | Tiempo/Iteración | Total |
|--------|-------------|-----------------|-------|
| Value Iteration | 20 | Rápido | Rápido |
| Policy Iteration | 5 | Lento | Lento |
| Resultado | Mismo $V^*$ | Mismo $\pi^*$ | - |

### 3.3 Aplicación en PES: No Usado Directamente

El PES **no usa** estos métodos directamente porque:

1. **Espacio de estados grande**: $5863$ estados (continuo discretizado)
2. **Modelo de transición desconocido**: No tenemos $\mathcal{P}(s' \vert s, a)$ explícitamente
3. **Recompensas complejas**: No lineal en acciones

**Solución**: Usar **Q-Learning**, que es **model-free**.

---

## 4. Q-Learning: Aprendizaje sin Modelo

### 4.1 Problemas del Model-Based RL

**Requerimientos**:
- Conocer $\mathcal{P}(s' \vert s, a)$ (modelo de transición)
- Conocer $\mathcal{R}(s, a, s')$ (función de recompensa)

**En PES**: No tenemos modelo explícito del proceso de pandemia.

**Solución**: **Model-Free RL** = Aprender directamente de experiencias sin modelo.

### 4.2 Ecuación de Bellman para Q-Learning

En lugar de:

$$Q^{\pi}(s, a) = \mathbb{E}[r + \gamma V^{\pi}(s')]$$

Usamos ecuación de **Bellman esperada**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

**Término de Error** (TD Error):

$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

**Interpretación**:
- $r + \gamma \max_{a'} Q(s', a')$ = lo que **debería** ser $Q(s, a)$ (target)
- $Q(s, a)$ = lo que **actualmente** es (estimate)
- $\delta$ = discrepancia (error)

### 4.3 Algoritmo Q-Learning Completo

```python
def Q_Learning(env, alpha, gamma, epsilon, episodes):
    """
    Aprende Q-table mediante Q-Learning (Watkins, 1989).
    
    Parámetros:
        alpha: learning rate (velocidad de aprendizaje)
        gamma: discount factor (importancia del futuro)
        epsilon: exploración (probabilidad de acción aleatoria)
        episodes: número de episodios de entrenamiento
    """
    
    # 1. Inicializar Q-table
    Q = defaultdict(lambda: [0.0] * len(env.action_space))
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 2. Seleccionar acción (ε-greedy)
            if random.random() < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = argmax(Q[state])           # Explotación
            
            # 3. Ejecutar acción en entorno
            next_state, reward, done, _ = env.step(action)
            
            # 4. Bellman Update
            old_Q = Q[state][action]
            max_next_Q = max(Q[next_state])
            new_Q = old_Q + alpha * (reward + gamma * max_next_Q - old_Q)
            Q[state][action] = new_Q
            
            # 5. Transicionar
            state = next_state
        
        # 6. Decaer ε (exploración gradualmente → explotación)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    return Q  # Policy: π(s) = argmax_a Q(s, a)
```

### 4.4 Propiedades Teóricas

#### Convergencia

**Teorema**: Si se visita cada par $(s, a)$ infinitamente a menudo y $\alpha$ decae apropiadamente:

$$\lim_{t \to \infty} Q_t(s, a) = Q^*(s, a) \text{ con probabilidad 1}$$

**Condiciones**:

1. $\sum_{t=0}^{\infty} \alpha_t = \infty$ (learning rate total infinito)
2. $\sum_{t=0}^{\infty} \alpha_t^2 < \infty$ (learning rate decrece)

**En PES**: $\alpha = 0.2$ (constante) → no converge formalmente, pero funciona en práctica

#### Off-Policy Learning

Q-Learning es **off-policy** = política usada para exploración ≠ política evaluada:

```
Política de Comportamiento (Behavior):   ε-greedy
Política Evaluada (Target):              argmax (greedy)
```

**Ventaja**: Puede aprender de experiencias exploratorias sin seguir completamente esa política.

### 4.5 Explorando vs Explotando

#### Dilema Exploration-Exploitation

En cada paso, decidir:

```
Exploración (Risk)              Explotación (Safe)
├─ Probar acciones nuevas       ├─ Usar mejor acción conocida
├─ Descubrir estrategias mejor  ├─ Maximizar recompensa inmediata
└─ Largo plazo                  └─ Corto plazo
```

#### ε-Greedy Policy

```python
def epsilon_greedy(Q, state, epsilon):
    """
    Con probabilidad ε: acción aleatoria (exploración)
    Con probabilidad 1-ε: mejor acción (explotación)
    """
    if random.random() < epsilon:
        return random_action()
    else:
        return argmax(Q[state, :])
```

**Ventaja**: Simple, intuitivo

**Desventaja**: No pondera intentos basados en confianza

#### Softmax (Boltzmann) Policy

```python
def softmax_policy(Q, state, temperature=1.0):
    """
    Probabilidades proporcionales a Q-values.
    
    T alto: distribución uniforme (exploración)
    T bajo: distribución concentrada en máximo (explotación)
    """
    Q_exp = np.exp(Q[state, :] / temperature)
    probs = Q_exp / np.sum(Q_exp)
    return np.random.choice(actions, p=probs)
```

**En PES**: Se usa ε-greedy durante entrenamiento, luego explotación pura.

---

## 5. Temporal Difference (TD) Learning

### 5.1 Problema del Monte Carlo

**Monte Carlo**: Esperar hasta fin de episodio para actualizar:

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

Donde $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ (retorno real)

**Problema**: Varianza alta, lento convergencia

### 5.2 Temporal Difference: Bootstrap

**TD**: Actualizar usando estimado de estados futuros:

$$V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]$$

**Ventajas sobre Monte Carlo**:
- Actualización en cada paso (no solo al final)
- Menor varianza
- Converge más rápido

**Comparación**:

$$\text{Monte Carlo: } G_t = r_t + \gamma r_{t+1} + \ldots$$
$$\text{TD: } r_t + \gamma V(s')$$

**TD Error** (notación):

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 5.3 TD-λ: Eligibility Traces

Generalización que interpola entre Monte Carlo y TD:

$$\text{MC: } \lambda = 1$$
$$\text{TD(0): } \lambda = 0$$

Para valores intermedios, usa **eligibility traces** que dan más peso a estados visitados recientemente.

**En PES**: No se usa (Q-Learning con $\lambda = 0$)

---

## 6. Deep Reinforcement Learning (DRL)

### 6.1 Limitaciones de Tabular RL

**Problemas cuando espacio de estados es grande**:

| Método | Estados | Memoria | Ejemplo |
|--------|---------|---------|---------|
| Tabular | ~1000 | KB | PES (5863) ✓ |
| Tabular | ~100K | MB | Go (10^170) ✗ |
| Tabular | ~10M | GB | Atari píxeles ✗ |

**Solución**: Usar **función aproximadora** (red neuronal) en lugar de tabla:

$$Q_\theta(s, a) = \text{NN}_\theta(s, a)$$

### 6.2 Deep Q-Network (DQN)

**Arquitectura**:

```
Input: State s
  ↓
Dense(128)→ReLU
  ↓
Dense(128)→ReLU
  ↓
Output: Q-values [0, 1, 2, ..., 10]
```

**Entrenamiento**:

```python
# Target
y = r + γ * max_a' Q_target(s', a')

# Loss
L = (y - Q_theta(s, a))^2

# Backprop
∇_θ L
```

**Innovaciones DQN (Mnih et al., 2015)**:

1. **Experience Replay**: Guardar transiciones, muestrear minibatches
2. **Target Network**: Red separada para calcular targets

**Por qué no en PES**: Espacio de estados lo suficientemente pequeño para tabular

### 6.3 Policy Gradient Methods

En lugar de aprender $V$ o $Q$, aprender directamente política:

$$\max_\theta \mathbb{E}_{\pi_\theta}[G_t]$$

**Actor-Critic**: Combinación de policy gradient + value function

**En PES**: No usado (Q-Learning más simple y efectivo)

---

## 7. Teoría de Exploración

### 7.1 Regret

Define eficiencia de algoritmo de RL:

$$\text{Regret}(T) = \sum_{t=1}^{T} (V^*(s_t) - V^{\pi_t}(s_t))$$

Mide diferencia acumulada entre política óptima y política aprendida.

**Óptimo**: Sublineal (e.g., $O(\sqrt{T})$)

### 7.2 Upper Confidence Bound (UCB)

Alternativa a ε-greedy:

$$a_t = \arg\max_a \left[Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}}\right]$$

Equilibra Q-value + incertidumbre (contadas a menudo)

### 7.3 Thompson Sampling

Bayesian approach: mantener posterior sobre $Q(s, a)$, samplear

**Ventaja**: Automáticamente balancea exploración/explotación

**En PES**: No usado (complejidad vs ε-greedy)

---

## 8. Casos de Convergencia Teórica

### 8.1 Cuando Q-Learning Converge a $Q^*$

**Teorema (Watkins & Dayan, 1992)**: Q-Learning converge a $Q^*$ si:

1. Espacio finito de estados y acciones
2. Cada $(s, a)$ visitado infinitamente
3. Learning rate $\alpha_t$ satisface:
   - $\sum_t \alpha_t = \infty$
   - $\sum_t \alpha_t^2 < \infty$

**Tasa de Convergencia**: $O(t^{-1})$ en promedio

### 8.2 Aplicabilidad en PES

✓ **Satisfecho**: Estados y acciones finitos (5863 × 11)

✗ **No satisfecho**: 
- Learning rate constante ($\alpha = 0.2$, no decae)
- No garantiza visitar todos $(s, a)$ (stochastic dynamics)

**Resultado**: Converge en práctica, pero no garantizado teóricamente

### 8.3 Sample Complexity

¿Cuántos samples (transiciones) necesarios para aprender $\pi^*$?

**Teorema (Kearns & Singh)**: $O\left(\frac{\vert S \vert \vert A \vert}{\epsilon^2 (1-\gamma)^3} \ln \frac{1}{\delta}\right)$

Para PES:
$$O\left(\frac{5863 \times 11}{\epsilon^2 (1-0.9)^3} \ln \frac{1}{\delta}\right) \approx O(10^8)$$

Con 20,000 episodios × 5-7 trials/episodio ≈ 100,000-140,000 samples → bajo para teórico

---

## 9. Variantes y Extensiones

### 9.1 Expected SARSA vs Q-Learning

**Q-Learning** (usado en PES):
$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_a Q(s', a) - Q(s, a)]$$

**Expected SARSA**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \sum_{a'} \pi(a' \vert s') Q(s', a') - Q(s, a)]$$

**Diferencia**: SARSA espera sobre política actual, Q-Learning asume greedy.

**Resultado**: SARSA menos optimista (mejor para problemas ruidosos)

### 9.2 Double Q-Learning

Evita sobrestimación de Q-values:

$$Q_1(s,a) \leftarrow Q_1 + \alpha[r + \gamma Q_2(s', \arg\max_a Q_1(s',a)) - Q_1(s,a)]$$

**Beneficio**: Estimados más estables

### 9.3 Dueling DQN

Separa estimación:

$$Q(s,a) = V(s) + A(s,a) - \text{mean}(A(s))$$

- $V(s)$: valor de estar en estado
- $A(s,a)$: ventaja de acción

**Beneficio**: Mejor representación para ciertos problemas

---

## 10. Comparación de Algoritmos

```
                    Sample Eff   Convergence   Complexity   Aplicabilidad
                    ──────────────────────────────────────────────────────
Value Iteration      Excelente    Garantizada   Baja         Modelo conocido
Policy Iteration     Excelente    Garantizada   Media        Modelo conocido
Q-Learning           Buena        Asintótica    Media        Modelo desconocido ✓
SARSA                Media        Garantizada   Media        Exploración segura
Actor-Critic         Media        No garantida  Alta         Espacios continuos
DQN                  Buena        No garantida  Muy alta     Espacios grandes
PPO                  Buena        No garantida  Muy alta     SOTA (actual)
```

**PES elige Q-Learning**: Buen balance entre simplicidad y eficacia

---

## 11. Matemática Avanzada

### 11.1 Contraction Mapping

La actualización de Bellman es **contraction**:

$$\left\|T V_1 - T V_2\right\|_{\infty} \leq \gamma \left\|V_1 - V_2\right\|_{\infty}$$

Donde $T V(s) = \max_a \sum_{s'} P(s'\vert s,a)[R(s,a,s') + \gamma V(s')]$

**Implicación**: Garantiza convergencia única a $V^*$

### 11.2 Stochastic Approximation

Q-Learning es instancia de **stochastic approximation**:

$$\theta_{t+1} = \theta_t + \alpha_t (y_t - f_\theta(\mathbf{x}_t)) \nabla f_\theta(\mathbf{x}_t)$$

**Convergencia**: Análisis de martingalas garantiza convergencia a fijo

### 11.3 Función de Ventaja

Define ventaja relativa de acción:

$$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$$

**Propiedad**: $\mathbb{E}_a[A^{\pi}(s, a)] = 0$

**Uso**: Reduce varianza en policy gradient methods

---

## 12. Aplicación Específica a PES

### 12.1 Por Qué Q-Learning es Óptimo para PES

```
Criterio                          Satisfecho?   Implicación
─────────────────────────────────────────────────────────
Espacio de estados finito         ✓            Q-Learning aplicable
Espacio conocido a priori         ✓            Inicializar Q-table
Dinámicas estocásticas            ✓            Q-Learning robusto
Función recompensa simple          ✓            Aprendizaje estable
No modelo explícito disponible    ✓            Model-free necesario
Interpretabilidad importante       ✓            Tabla Q legible
```

### 12.2 Flujo Teórico → Implementación

```
Bellman Optimality Equation (Teórica)
         ↓
Q(s,a) = E[r + γ max_a' Q(s',a')]

Bellman Update Rule (Algorithm)
         ↓
Q ← Q + α(r + γ max_a' Q(s',a') - Q)

Implementación en PES (ext/pandemic.py:230)
         ↓
Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
```

---

## 13. Preguntas Teóricas y Respuestas

### P1: ¿Por qué γ < 1?

**R**: Convergencia matemática + realismo económico
- Si $\gamma = 1$: futuro tan importante como presente (infinito horizonte)
- Si $\gamma < 1$: futuro importante pero descontado (horizonte finito)

### P2: ¿Es Q-Learning on-policy u off-policy?

**R**: **Off-policy**. Evalúa política greedy ($\max_a$) mientras sigue ε-greedy.

```
Behavior Policy:  ε-greedy (exploración)
Target Policy:    greedy (evaluación)
```

### P3: ¿Converge si no exploramos suficiente?

**R**: **No**. Q-Learning requiere experiencia suficiente de todos $(s,a)$ pares para convergencia.

### P4: ¿Qué pasa si ε nunca decae?

**R**: Converge a política cercana óptima pero no óptima (límite es π que sigue ε-greedy).

---

## Conclusión

Q-Learning implementado en PES es instancia de **Reinforcement Learning model-free** que:

1. **Aprende función Q** de transiciones observadas
2. **Converge** a política óptima bajo condiciones teóricas
3. **Balancea** exploración-explotación via ε-greedy
4. **Escala** a espacios de tamaño medio (5863 estados)

La teoría subyacente es sólida (Bellman, contraction, stochastic approximation) aunque la implementación PES no satisface todos los supuestos teóricos (ej: ε no decae, no todos $(s,a)$ visitados).

En la práctica, funciona robustamente debido a estructura del problema y suficiente data.

---

## Lecturas Recomendadas

### Textbooks Clásicos
- **Sutton & Barto (2018)**: "Reinforcement Learning: An Introduction" 
- **Bertsekas & Tsitsiklis (1996)**: "Neuro-Dynamic Programming"

### Papers Seminales
- Watkins & Dayan (1992): "Q-Learning"
- Mnih et al. (2015): "Human-level control via DQN"
- Schulman et al. (2017): "PPO"

### Recursos Online
- David Silver's RL Course (UCL)
- OpenAI Spinning Up in Deep RL
- DeepMind blog

---

## Glosario

| Término | Símbolo | Definición |
|---------|---------|-----------|
| Bellman Equation | | Ecuación recursiva para value functions |
| Discount Factor | γ | Peso relativo de recompensas futuras |
| Eligibility Trace | e(s,a) | Frecuencia de visita a (s,a) |
| Entropy | H(π) | Aleatoriedad de política |
| Exploration | | Intentar acciones nuevas |
| Exploitation | | Usar mejor acción conocida |
| Learning Rate | α | Velocidad de convergencia |
| Markov Property | | Futuro depende solo de presente |
| MDP | | Formalismo de RL |
| Off-Policy | | Evaluar ≠ Comportamiento |
| On-Policy | | Evaluar = Comportamiento |
| Policy | π | Estrategia (mapeo s→a) |
| Regret | | Pérdida vs política óptima |
| Reward | r | Feedback del ambiente |
| State Value | V(s) | Valor esperado de estado |
| TD Error | δ | Discrepancia target-estimate |
| Value Function | | Función de valor de estados/acciones |
