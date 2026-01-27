# Marco Teórico de Reinforcement Learning

**Documento:** Fundamentos matemáticos y conceptuales de Reinforcement Learning aplicados en PES  
**Autor:** Análisis de Ciencia de Datos  
**Nivel:** Avanzado (Máster/PhD)

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Fundamentos: Procesos de Decisión Markovianos](#2-fundamentos-procesos-de-decisión-markovianos)
3. [Retorno y Función de Valor](#3-retorno-y-función-de-valor)
4. [Ecuación de Bellman](#4-ecuación-de-bellman)
5. [Métodos de Aproximación: Programación Dinámica](#5-métodos-de-aproximación-programación-dinámica)
6. [Métodos de Control: Q-Learning](#6-métodos-de-control-q-learning)
7. [Exploración vs Explotación](#7-exploración-vs-explotación)
8. [Convergencia y Garantías Teóricas](#8-convergencia-y-garantías-teóricas)
9. [Aplicación en PES](#9-aplicación-en-pes)
10. [Extensiones y Limitaciones](#10-extensiones-y-limitaciones)

---

## 1. Introducción

### Definición

**Reinforcement Learning (RL)** es el subcampo del aprendizaje automático que estudia cómo un agente aprende a tomar decisiones secuenciales mediante interacción con un entorno, maximizando una señal de recompensa acumulada.

### Características Clave

1. **Secuencialidad**: Las decisiones afectan estados futuros
2. **Recompensa retardada**: La calidad de una acción se evalúa con el tiempo
3. **Exploración**: El agente debe balancear conocimiento actual vs descubrimiento
4. **Incertidumbre**: El entorno puede ser estocástico

### Diferencia con Otros Paradigmas

| Aspecto | Supervised Learning | Unsupervised Learning | RL |
|--------|-------|---------|-----|
| **Señal de aprendizaje** | Labels correctos | Estructura interna | Recompensas |
| **Tipo de datos** | Estático (i.i.d) | Estático | Dinámico, secuencial |
| **Objetividad** | Minimizar error | Encontrar patrones | Maximizar retorno |
| **Feedback** | Inmediato | Ninguno | Retardado |
| **Optimización** | Offline | Offline | Online |

---

## 2. Fundamentos: Procesos de Decisión Markovianos

### 2.1 Definición Formal de MDP

Un **Markov Decision Process** es una tupla (S, A, P, R, γ) donde:

- **S**: Espacio de estados, conjunto finito de estados posibles
- **A**: Espacio de acciones, conjunto finito de acciones disponibles  
- **P(s'|s,a)**: Función de transición, probabilidad de transición
  $$P(s'|s,a) = \Pr[S_{t+1}=s' | S_t=s, A_t=a]$$
  
- **R(s,a,s')**: Función de recompensa, recompensa esperada
  $$R(s,a,s') = \mathbb{E}[R_t | S_t=s, A_t=a, S_{t+1}=s']$$
  
- **γ**: Factor de descuento, $\gamma \in [0,1]$, importancia de recompensas futuras

### 2.2 Propiedad de Markov

**Definición**: Un proceso satisface la propiedad de Markov si y solo si:

$$\Pr[S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0] = \Pr[S_{t+1}|S_t, A_t]$$

**Interpretación**: El futuro es independiente del pasado dado el presente.

**En PES**: 
```
Estado: (recursos_left, city_number, severity_actual)

Propiedad de Markov: ¿Decisión óptima depende solo de estos 3 factores?
- Sí: No importa cómo llegamos a este estado, la decisión es la misma
- Formalización: π(a|s) = π(a|s) ∀ histórico que lleve a s
```

### 2.3 Política

**Definición**: Una política π es una distribución de probabilidad sobre acciones dado un estado:

$$\pi(a|s) = \Pr[A_t = a | S_t = s]$$

**Tipos**:
- **Determinística**: π(a|s) ∈ {0, 1} (la política elige exactamente una acción)
- **Estocástica**: π(a|s) ∈ [0, 1] (distribución general)

**En PES**:
- Política durante entrenamiento: ε-Greedy (estocástica)
- Política durante deployment: Greedy determinística (π(a*|s)=1, donde a* = argmax Q)

---

## 3. Retorno y Función de Valor

### 3.1 Retorno (Return)

**Definición**: El retorno G_t es la suma descontada de recompensas futuras:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Intuición**:
- γ = 0: Solo importa recompensa inmediata (miope)
- γ = 1: Todas las recompensas igualmente importantes (infinite horizon)
- γ = 0.95: Balance (típico)

**En PES**: 
```
Caso especial: Recompensa inmediata (local)

En cada trial, la recompensa es local a ese trial:
r_t = α*a_t - β*s_t

No hay acumulación explícita entre trials.
Esto es una simplificación que reduce dimensionalidad.
```

### 3.2 Función de Valor de Estado

**Definición**: El valor de un estado bajo política π es:

$$V_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s\right]$$

**Interpretación**: ¿Cuánta recompensa acumulada espero si estoy en estado s y sigo política π?

**Propiedades**:
- Recurrencia: V_π(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a)[r + γV_π(s')]
- Optimalidad: V*(s) = max_π V_π(s) (valor con mejor política)

### 3.3 Función de Valor de Acción (Q-Function)

**Definición**: El valor de una acción en un estado bajo política π:

$$Q_\pi(s,a) = \mathbb{E}_\pi[G_t | S_t=s, A_t=a]$$

**Interpretación**: ¿Cuánta recompensa acumulada espero si estoy en s, tomo acción a, y luego sigo π?

**Relación**:
$$V_\pi(s) = \sum_a \pi(a|s) Q_\pi(s,a)$$

(El valor de estado es el promedio ponderado de valores de acciones)

**Optimalidad**:
$$Q^*(s,a) = \max_\pi Q_\pi(s,a)$$

---

## 4. Ecuación de Bellman

### 4.1 Ecuación de Bellman para Función de Valor

**Deducción**:
$$V_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) | S_t=s]$$

Expandiendo la expectativa:
$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a)[r + \gamma V_\pi(s')]$$

**Significado**: El valor presente es la recompensa inmediata más el valor descontado del siguiente estado.

**En notación simplificada**:
$$V_\pi(s) = \mathbb{E}_\pi[R_t + \gamma V_\pi(S_{t+1})]$$

### 4.2 Ecuación de Bellman Óptima

Para la política óptima π*:

$$V^*(s) = \max_a \sum_{s',r} P(s',r|s,a)[r + \gamma V^*(s')]$$

Para Q-values:
$$Q^*(s,a) = \sum_{s',r} P(s',r|s,a)[r + \gamma \max_{a'} Q^*(s',a')]$$

**Interpretación**: En cada estado, la política óptima es elegir la acción que maximiza valor esperado.

### 4.3 Ecuación de Bellman en Forma Iterativa

**Aproximación iterativa** (usado en algoritmos):

$$V_{t+1}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a)[r + \gamma V_t(s')]$$

Converge: V_t → V_π cuando t → ∞

**Error de Bellman** (TD Error):
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

Mide qué tan bien predecimos el valor.

---

## 5. Métodos de Aproximación: Programación Dinámica

### 5.1 Iteración de Política (Policy Iteration)

**Algoritmo**:
```
1. Inicializar política π arbitraria
2. Repetir hasta convergencia:
   a) Evaluación: Calcular V_π resolviendo Bellman
   b) Mejora: π'(s) = argmax_a Σ P(s'|s,a)[r(s,a,s') + γV_π(s')]
   c) Si π' = π, salir (converged)
   d) π ← π'
```

**Complejidad**: O(|S|² |A|) por iteración

**Convergencia**: Garantizada en |A|^|S| iteraciones máximo

### 5.2 Iteración de Valor (Value Iteration)

**Algoritmo**:
```
1. Inicializar V(s) = 0 ∀s
2. Repetir hasta convergencia:
   Para cada s ∈ S:
       V(s) ← max_a Σ P(s'|s,a)[r(s,a,s') + γV(s')]
```

**Complejidad**: O(|S|² |A|) por iteración

**Ventaja**: No requiere calcular política explícitamente (implícita en argmax)

### 5.3 Limitaciones

- Requiere **modelo completo**: P(s'|s,a) y R(s,a,s')
- **Computacionalmente intensivo** para espacios grandes
- No es **online**: Requiere datos completos

---

## 6. Métodos de Control: Q-Learning

### 6.1 Aprendizaje por Diferencias Temporales (TD)

**Idea clave**: No resolver Bellman exactamente, estimar incrementalmente.

**TD Update**:
$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**Componentes**:
- $\alpha$: Learning rate (0.1 típicamente)
- $[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$: **TD Error**, señal de actualización

### 6.2 Q-Learning (Off-Policy)

**Definición**: Q-Learning es un método TD que estima Q-values directamente.

**Actualización Q**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]$$

**Características clave**:
1. **Off-policy**: Puede aprender de cualquier política mientras usa greedy
2. **Model-free**: No necesita P(s'|s,a) ni R(s,a,s')
3. **Bootstrapping**: Usa estimación de V_t+1 para mejorar V_t

**Ecuación en forma matricial**:
$$\mathbf{Q} \leftarrow \mathbf{Q} + \alpha [\mathbf{R} + \gamma \max_a \mathbf{Q}' - \mathbf{Q}]$$

### 6.3 SARSA (On-Policy)

**Actualización SARSA** (State-Action-Reward-State-Action):
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Diferencia con Q-Learning**:
- Q-Learning: Usa max_{a'} Q(s', a') (acción óptima)
- SARSA: Usa Q(s', a_{t+1}) (acción actual)

**Implicación**: SARSA es más conservador (on-policy), Q-Learning es más agresivo (off-policy)

### 6.4 Convergencia de Q-Learning

**Teorema**: Bajo condiciones:
1. Espacio de estados finito: |S| < ∞
2. Espacio de acciones finito: |A| < ∞
3. Learning rate: Σ_n α_n(s,a) = ∞, Σ_n α_n^2(s,a) < ∞
4. Cada (s,a) visitado infinitas veces

Entonces: Q_n(s,a) → Q*(s,a) with probability 1 cuando n → ∞

**En práctica**: 
- PES: 20,000 episodios (no infinito pero suficiente)
- Convergencia empírica: ~15,000 episodios

---

## 7. Exploración vs Explotación

### 7.1 El Trade-off Fundamental

**Problema**: Siendo greedy siempre (explotación) puede quedar atrapado en óptimos locales.

**Necesidad**: Probar acciones peor conocidas (exploración) para aprender mejor política.

### 7.2 Estrategia ε-Greedy

**Definición**:
$$A_t = \begin{cases}
\text{argmax}_a Q(S_t, a) & \text{con probabilidad } 1-\varepsilon \\
\text{acción aleatoria} & \text{con probabilidad } \varepsilon
\end{cases}$$

**Ventajas**:
- Simple de implementar
- Garantía teórica: con ε>0, cada (s,a) visitado ∞ veces
- Funciona bien en práctica

**Desventajas**:
- ε fijo: Explora uniformemente (ineficiente)
- Mejor: ε decreciente (exploit-dominant con el tiempo)

### 7.3 ε-Decreciente

**Variante 1** (exponencial):
$$\varepsilon_n = \varepsilon_0 \cdot \rho^n, \quad \rho \in (0,1)$$

Por ejemplo: ε_n = 0.1 × 0.99^n

**Variante 2** (lineal):
$$\varepsilon_n = \max(\varepsilon_{min}, \varepsilon_0 - \beta \cdot n)$$

**Interpretación**: Al principio exploramos mucho (ε alto), luego explotamos más (ε→0)

### 7.4 Estrategia UCB (Upper Confidence Bound)

**Alternativa más sofisticada**:
$$A_t = \arg\max_a \left[ Q(S_t, a) + c\sqrt{\frac{\ln N_t}{N_a(t)}} \right]$$

Donde:
- N_t: Número total de acciones tomadas
- N_a(t): Veces que acción a fue tomada

**Intuición**: Prioriza acciones que se han explorado menos pero parecen buenas.

**Ventaja**: Adaptativo - no requiere ajustar ε manualmente

---

## 8. Convergencia y Garantías Teóricas

### 8.1 Condiciones Suficientes para Convergencia

Para Q-Learning tabular:

1. **Finitud**: |S| < ∞, |A| < ∞
2. **Rewards acotadas**: ∃R_max tal que |R(s,a,s')| ≤ R_max
3. **Learning rate decreciente**:
   - Σ_t α_t(s,a) = ∞ (suma diverge)
   - Σ_t α_t^2(s,a) < ∞ (suma cuadrados converge)
   
   Ejemplo: α_t = 1/t satisface ambas
   
4. **Exploración uniforme**: Cada (s,a) visitado ∞ veces

### 8.2 Tasa de Convergencia

**Análisis empírico** (PES):
- Primeros 5,000 episodios: Mejora rápida (exploración dominante)
- Episodios 5,000-15,000: Mejora gradual (refinamiento)
- Episodios 15,000-20,000: Saturación (casi convergido)

**Justificación matemática**:
- TD error inicial: δ = R + γmax(Q') - Q ≈ R (grande)
- TD error final: δ ≈ 0 (convergencia)

### 8.3 Optimalidad

**Garantía**: Q_learned → Q* (con probabilidad 1)

**Implicación**:
- Política derivada: π(s) = argmax_a Q*(s,a)
- Valor derivado: V*(s) = max_a Q*(s,a)
- Son **óptimos globales**

**Caveat**: Requiere convergencia completa (infinitos episodios)

---

## 9. Aplicación en PES

### 9.1 Mapeo Teórico → Implementación

| Concepto Teórico | Parámetro PES | Valor |
|----------------|-------|-------|
| |S| (estados) | 31 × 13 × 11 | 4,433 |
| |A| (acciones) | 11 | {0,1,...,10} |
| Espacio de MDP | (s,a,r,s') | Finite, acyclic |
| Rewards | r = α×a - β×s | Bounded [-5, 50] |
| Factor descuento | γ | 0.95 |
| Learning rate | α | 0.1 (fijo) |
| Exploración | ε-Greedy | ε_0=0.1, ρ=0.99 |
| Episodios | Num episodes | 20,000 |

### 9.2 Función de Recompensa Diseñada

**Definición**:
$$r_t = \alpha \cdot a_t - \beta \cdot s_t$$

Donde:
- α = 0.24 (eficacia de recursos)
- β = 0.76 (tasa de propagación natural)
- a_t = recursos asignados
- s_t = severidad actual

**Propiedades**:
1. Localidad: Solo depende de (a_t, s_t), no de histórico
2. Acotación: -5 ≤ r ≤ 50 (asegura convergencia)
3. Interpretabilidad: Más recursos → más recompensa, más severidad → menos recompensa

### 9.3 Política Aprendida

**Durante entrenamiento**:
$$\pi(a|s) = \begin{cases}
\frac{1}{|A|} & \text{con probabilidad } \varepsilon_n \\
\mathbf{1}[\arg\max_a Q_n(s,a)] & \text{con probabilidad } 1-\varepsilon_n
\end{cases}$$

**Durante evaluación** (deployment):
$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

Completamente determinística.

### 9.4 Confianza Metacognitiva

**Novedad**: Calcular confianza de la decisión.

**Fundamento**:
- Si Q(s,a) tiene una acción con valor mucho mayor → Alta confianza
- Si Q(s,a) es casi uniforme → Baja confianza (incertidumbre)

**Formalización**:
$$\text{Confidence}(s) = 1 - \frac{H(Q(s,:)) - H_{min}}{H_{max} - H_{min}}$$

Donde H(·) es entropía Shannon:
$$H(p) = -\sum_i p_i \log p_i$$

---

## 10. Extensiones y Limitaciones

### 10.1 Limitaciones del Q-Learning Tabular

1. **Escalabilidad**: O(|S| × |A|) memoria
   - PES: 4,433 × 11 ≈ 50K, manejable
   - Pero no escala a espacios continuos

2. **Generalización**: No interpola entre estados
   - PES: Solo conocemos Q(s,a) para estados visitados
   - Nuevos estados: Cero conocimiento

3. **Velocidad de aprendizaje**: Baja para espacios grandes
   - PES: 20,000 episodios necesarios
   - En problemas grandes: Millones de episodios

### 10.2 Extensión: Function Approximation

**Idea**: Usar función parametrizada en lugar de tabla

$$Q(s,a,\theta) \approx Q^*(s,a)$$

Por ejemplo, red neuronal:
$$Q_\theta(s,a) = \text{NN}_\theta(s, a)$$

**Ventajas**:
- ✅ Escalable a espacios grandes (S^d)
- ✅ Generaliza a nuevos estados
- ✅ Aprendizaje más rápido

**Desventajas**:
- ❌ Convergencia no garantizada
- ❌ Mayor complejidad computacional

### 10.3 Extensión: Deep Q-Networks (DQN)

**Innovaciones**:
1. **Experience Replay**: Guardar transiciones en buffer, sample aleatoriamente
2. **Target Network**: Usar red separada para computar targets
3. **Estabilidad**: Mitigar divergencia debido a correlaciones

**Actualización DQN**:
$$\theta \leftarrow \theta + \alpha \nabla_\theta [R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)]^2$$

Donde θ^- es red objetivo (actualizada cada C pasos).

### 10.4 Extensión: Policy Gradient Methods

**Alternativa**: En lugar de aprender Q, aprender π directamente.

**Ejemplos**:
- REINFORCE
- Actor-Critic
- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)

**Ventaja**: Pueden aprender políticas estocásticas, mejor exploración

### 10.5 Limitación en PES Actual

**Estructura del problema**:
- Estados visibles: Sí (recursos, severidad conocida)
- Dinámica determinística: Basicalmente sí (β,α fijos)
- Horizon finito: Sí (64 secuencias × 8 bloques)

**Implicación**: Q-Learning es **suficiente** para PES.

Extensiones necesarias solo si:
- ✗ Estados parcialmente observables
- ✗ Dinámica muy estocástica
- ✗ Espacio de estados enorme

---

## 11. Resumen Comparativo: Métodos RL

| Método | Tipo | Convergencia | Complejidad | En PES |
|--------|------|---------|-----------|--------|
| **Policy Iteration** | DP | O(k²) | O(\|S\|²\|A\|) | Teórico |
| **Value Iteration** | DP | O(k) | O(\|S\|²\|A\|) | Teórico |
| **SARSA** | TD | ε-greedy | O(\|S\|\|A\|) | ✓ Viable |
| **Q-Learning** | TD | ε-greedy | O(\|S\|\|A\|) | ✓ Usado |
| **DQN** | FA | Adam/SGD | O(d²) | Overkill |
| **A3C** | PG | Async | O(1/ε²) | Overkill |

---

## 12. Ecuaciones Resumidas

### Ecuaciones Fundamentales

**Ecuación de Bellman**:
$$V(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$$

**Bellman Óptima**:
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

**Q-Bellman**:
$$Q(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q(s',a')]$$

**TD Update**:
$$V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$$

**Q-Learning Update** (usada en PES):
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**ε-Greedy**:
$$\pi(a|s) = \begin{cases} 1-\varepsilon + \varepsilon/|A| & a = a^* \\ \varepsilon/|A| & a \neq a^* \end{cases}$$

---

## 13. Referencias Teóricas

### Libros Fundamentales
1. Sutton & Barto (2018). **Reinforcement Learning: An Introduction** (2nd Ed.)
   - Capítulo 3-6: MDP, Bellman, TD Learning
   - Capítulo 6: Q-Learning
   - Capítulo 2: Exploration-Exploitation

2. Puterman (1994). **Markov Decision Processes**
   - Teoría exhaustiva de MDP
   - Programación dinámica
   - Convergencia

### Papers Clave
1. Watkins & Dayan (1992). "Q-Learning"
   - *Machine Learning 8:279-292*
   - Prueba de convergencia de Q-Learning off-policy

2. Van Hasselt, Guez & Silver (2016). "Deep Reinforcement Learning with Double Q-Learning"
   - *AAAI 2016*
   - Soluciona sobreestimación en DQN

---

## 14. Conclusión

El marco teórico de Reinforcement Learning proporciona garantías sólidas para el aprendizaje secuencial:

1. **Optimalidad**: Q-Learning converge a política óptima
2. **Finitud**: Con estados/acciones finitos, algoritmo termina
3. **Eficiencia**: TD learning más rápido que DP
4. **Flexibilidad**: Fácil de extender a problemas más complejos

En PES específicamente, Q-Learning es **suficiente y adecuado** dado:
- Espacio de estados manejable
- Dinámica conocida
- Recompensas locales

Las extensiones a function approximation o policy gradients son innecesarias para el problema actual pero permitirían escalabilidad futura.

