# Teoría de Reinforcement Learning para Científicos de Datos

**Nivel**: Intermedio | **Duración de lectura**: 40 minutos | **Requisitos**: Probabilidad, Álgebra lineal, Python

---

## 1. Introducción: El Problema de Control Óptimo

### 1.1 Motivación

¿Cómo enseña a una máquina a tomar **decisiones secuenciales óptimas** en un entorno dinámico?

**Ejemplos de Aplicación**:
- 🎮 Juegos: Agente aprende a ganar (AlphaGo, Atari)
- 🚗 Robots: Control motor y navegación
- 💊 Medicina: Selección de tratamientos secuenciales
- 💰 Finanzas: Portfolio optimization
- 🦠 **COVID/Pandemia**: Asignación de recursos (nuestro caso)

### 1.2 ¿Por qué Reinforcement Learning?

| Paradigma ML | Requiere | Aprende |
|------------|----------|--------|
| **Supervisado** | Etiquetas (X, y) | Mapeo: entrada → salida |
| **No supervisado** | Solo X | Estructura de datos |
| **Refuerzo** | Recompensas (feedback) | Secuencia óptima de acciones |

En Reinforcement Learning:
- No hay etiquetas correctas
- El agente **explora** el ambiente
- Recibe **recompensa** como feedback
- Aprende a **maximizar recompensa acumulada**

---

## 2. Cadenas de Markov

### 2.1 Propiedad de Markov

**Definición**: Un proceso tiene la **propiedad de Markov** si el futuro es independiente del pasado, dado el presente:

$$P(s_{t+1} | s_t, s_{t-1}, \ldots, s_0) = P(s_{t+1} | s_t)$$

"El futuro solo depende del presente, no de la historia."

### 2.2 Cadena de Markov (CM)

Una **Cadena de Markov** es tupla $\langle S, P \rangle$:
- $S$: Conjunto finito de estados
- $P$: Matriz de transición $P(s' | s)$ para todo $(s, s')$

**Ejemplo: Clima**:
```
Estados: {Soleado, Nublado, Lluvia}

Matriz de transición P:
         A     Soleado  Nublado  Lluvia
      Soleado   0.8      0.15     0.05
P =   Nublado   0.2      0.6      0.2
      Lluvia    0.1      0.3      0.6

P(Mañana=Soleado | Hoy=Soleado) = 0.8
P(Mañana=Lluvia | Hoy=Nublado) = 0.2
```

### 2.3 Cadena de Markov con Recompensas (MRP)

Una **Markov Reward Process** es tupla $\langle S, P, R, \gamma \rangle$:
- $S$: Estados
- $P$: Transición
- $R(s)$: Recompensa esperada en estado $s$
- $\gamma \in [0,1]$: **Factor de descuento**

**Factor de descuento $\gamma$**:
- $\gamma = 1$: Recompensas futuras igual de valiosas que presentes
- $\gamma = 0.9$: Recompensa en 1 paso = 0.9 × recompensa hoy
- $\gamma = 0$: Solo importa recompensa inmediata

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots = \sum_{i=0}^{\infty} \gamma^i r_{t+i}$$

**Interpretación**: El **return** $G_t$ es la suma **descontada** de recompensas futuras.

### 2.4 Función de Valor en MRP

La **función de valor** predice el return esperado desde un estado:

$$V(s) = \mathbb{E}[G_t | s_t = s] = \mathbb{E}\left[\sum_{i=0}^{\infty} \gamma^i r_{t+i} | s_t = s\right]$$

**Ecuación de Bellman para MRP**:

$$V(s) = R(s) + \gamma \sum_{s'} P(s'|s) V(s')$$

"El valor de un estado = recompensa inmediata + valor descontado de los próximos estados"

**Ejemplo numérico**:
```
Estado: Soleado
R(Soleado) = +10 (es agradable)
Transiciones: P(Soleado|Soleado)=0.8, P(Nublado|Soleado)=0.15, P(Lluvia|Soleado)=0.05
γ = 0.9
V(Nublado) = 5, V(Lluvia) = -5

V(Soleado) = 10 + 0.9 * (0.8*V(Soleado) + 0.15*5 + 0.05*(-5))
           = 10 + 0.9 * (0.8*V(Soleado) + 0.75 - 0.25)
           = 10 + 0.9 * (0.8*V(Soleado) + 0.5)

Resolviendo: V(Soleado) = 10 + 0.9*0.8*V(Soleado) + 0.45
             V(Soleado) - 0.72*V(Soleado) = 10.45
             0.28*V(Soleado) = 10.45
             V(Soleado) ≈ 37.3
```

---

## 3. Procesos de Decisión de Markov (MDP)

### 3.1 Definición

Una **Markov Decision Process** es tupla $\langle S, A, P, R, \gamma \rangle$:
- $S$: Conjunto finito de estados
- $A$: Conjunto finito de acciones
- $P(s' | s, a)$: Probabilidad de transición
- $R(s, a, s')$ o $R(s, a)$: Recompensa por acción en estado
- $\gamma$: Factor de descuento

**Diferencia vs. MRP**: El agente **elige acciones**, generando un camino a través del MDP.

### 3.2 Intuición del MDP

```
Tiempo t: Estado s_t

Agente decide: Acción a_t

Ambiente responde:
  - Nuevo estado s_{t+1} ~ P(·|s_t, a_t)
  - Recompensa r_t ~ R(s_t, a_t)

Agente observa: s_{t+1}, r_t

Tiempo t+1: Repetir con estado s_{t+1}
```

### 3.3 Historia vs. Markov Property

En un **MDP**, aunque hay historia $\tau = (s_0, a_0, r_0, \ldots, s_t)$, solo importa el **estado actual**:

$$P(s_{t+1} | \tau) = P(s_{t+1} | s_t, a_t)$$

---

## 4. Políticas y Valor Esperado

### 4.1 Política

Una **política** $\pi$ es la estrategia del agente: "qué acción tomar en cada estado"

Dos tipos:
1. **Política Determinística**: $\pi(s) = a$ (única acción por estado)
2. **Política Estocástica**: $\pi(a|s) = P(a|s)$ (probabilidad sobre acciones)

### 4.2 Estado-Value Function

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{i=0}^{\infty} \gamma^i r_{t+i} | s_t = s\right]$$

"El valor esperado del return, siguiendo política $\pi$ desde estado $s$"

**Ecuación de Bellman para V**:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} P(s', r | s, a) [r + \gamma V^\pi(s')]$$

### 4.3 Acción-Value Function (Q-Function)

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{i=0}^{\infty} \gamma^i r_{t+i} | s_t = s, a_t = a\right]$$

"El valor esperado de tomar acción $a$ en estado $s$, luego seguir $\pi$"

**Ecuación de Bellman para Q**:

$$Q^\pi(s, a) = \sum_{s', r} P(s', r | s, a) [r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]$$

### 4.4 Relación V-Q

$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$$

"El valor del estado = promedio ponderado de valores de acciones"

---

## 5. Control Óptimo

### 5.1 Valores Óptimos

La **política óptima** es aquella que maximiza el return esperado:

$$\pi^* = \arg\max_\pi V^\pi(s) \quad \forall s$$

Los **valores óptimos** son:

$$V^*(s) = \max_\pi V^\pi(s) = \max_a Q^*(s, a)$$

$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

### 5.2 Ecuaciones de Bellman Óptimas

**Para V***:

$$V^*(s) = \max_a \sum_{s', r} P(s', r | s, a) [r + \gamma V^*(s')]$$

"El valor óptimo = máxima recompensa inmediata + valor descontado del mejor próximo estado"

**Para Q***:

$$Q^*(s, a) = \sum_{s', r} P(s', r | s, a) [r + \gamma \max_{a'} Q^*(s', a')]$$

### 5.3 Política Óptima

Una vez tenemos $Q^*$, la **política óptima es determinística**:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

"En cada estado, ejecutar la acción con mayor Q-value"

---

## 6. Algoritmos de Control: Q-Learning (FOCO PRINCIPAL)

### 6.1 Motivación

Los algoritmos vistos requieren conocer:
- $P(s' | s, a)$ (dinámica del sistema)
- $R(s, a)$ (función de recompensa)

**En RL práctico**, generalmente:
- No conocemos $P$ ni $R$ exactamente
- Interactuamos con el ambiente ("exploración")
- Aprendemos de la experiencia

### 6.2 Temporal Difference (TD) Learning

**Idea central**: Usar diferencia temporal como error de aprendizaje.

Observamos una transición: $(s, a, r, s')$

El **TD-error** es:

$$\delta = r + \gamma V(s') - V(s)$$

"Diferencia entre lo que esperabamos y lo que observamos"

Actualizamos V:

$$V(s) \leftarrow V(s) + \alpha \delta$$

Donde $\alpha$ es el **learning rate** (cuánto pesar la nueva observación).

### 6.3 Q-Learning Off-Policy

**Q-Learning** es un algoritmo de **aprendizaje directo de $Q^*$** sin necesitar $P$ ni $R$.

**Actualización Q-Learning**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

Desglosada:
- $Q(s, a)$ = estimación actual (antes de actualización)
- $\alpha$ = learning rate (velocidad de aprendizaje)
- $r + \gamma \max_{a'} Q(s', a')$ = target (lo que "debería" valer)
- Término entre corchetes = TD-error

**Pseudocódigo**:
```python
Q = tabla con valores iniciales (ej. ceros o uniformes)
α = learning_rate  (ej. 0.1 a 0.3)
γ = discount_factor  (ej. 0.9 a 0.99)
ε = exploration_rate  (ej. 0.1)

for episode in range(num_episodes):
    s = initial_state()
    
    while not terminal(s):
        # Política ε-greedy: explorar vs explotar
        if random() < ε:
            a = random_action()  # Exploración
        else:
            a = argmax_a Q[s, a]  # Explotación
        
        # Ejecutar acción, observar resultado
        s', r = environment(s, a)
        
        # Q-Learning update
        Q[s, a] = Q[s, a] + α * (r + γ * max_a Q[s', a] - Q[s, a])
        
        s = s'
```

### 6.4 Interpretación de Q-Learning

La **ecuación de actualización** implementa la ecuación de Bellman óptima:

$$Q(s, a) \approx Q^*(s, a)$$

Con suficientes actualizaciones, la tabla Q **converge** a $Q^*$.

---

## 7. Convergencia y Garantías Teóricas

### 7.1 Convergencia de Q-Learning

**Teorema**: Q-Learning converge a $Q^*$ bajo condiciones:

1. **Exploración suficiente**: Cada (s, a) se visita infinitamente
   - Garantizado por ε-greedy con ε > 0
   
2. **Learning rate decreciente**: $\sum_t \alpha_t = \infty$ y $\sum_t \alpha_t^2 < \infty$
   - Ejemplo: $\alpha_t = 1/t$ converge
   - Nuestro caso PES: $\alpha = 0.2$ fijo (conservador)
   
3. **Recompensas acotadas**: $|R(s, a)| < R_{max}$
   - En PES: $R_{max}$ = suma máxima severidades ✓

**Conclusión**: Q-Learning es **garantizado converger** en espacios finitos.

### 7.2 Tasa de Convergencia

La convergencia es **lenta** en espacios grandes:
- Con $n$ estados y $m$ acciones: $O(nm)$ complejidad
- En PES: 31 × 11 × 10 × 11 = 37,620 entradas en Q
- 1M episodios es razonable para garantizar convergencia

### 7.3 Maldición de la Dimensionalidad

Al aumentar complejidad:
- Más estados → más datos necesarios
- Espacio de estado exponencial → imposible tabular Q

**Soluciones modernas**:
- **Function Approximation**: Q(s,a) ≈ w·φ(s) (regresión lineal)
- **Deep Q-Networks (DQN)**: Red neuronal para aproximar Q
- **Policy Gradient**: Aproximar política directamente

PES usa la versión más simple: **tabla exacta** (viable porque espacio es pequeño).

---

## 8. Exploración vs. Explotación

### 8.1 Dilema Fundamental

En todo punto de decisión:

**Explotación** = usar acción que creo es mejor
$$a = \arg\max_a Q(s, a)$$

**Exploración** = probar acción nueva para mejorar estimación
$$a = \text{random}$$

¿Cuál es mejor?
- Demasiada explotación → converge a óptimo local
- Demasiada exploración → ignora lo que ya aprendió

### 8.2 Epsilon-Greedy

**Estrategia ε-Greedy**:

Con probabilidad $\varepsilon$: acción aleatoria (explorar)
Con probabilidad $1-\varepsilon$: mejor acción (explotar)

$$\pi_\varepsilon(a|s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{m} & \text{si } a = \arg\max Q(s, a) \\
\frac{\varepsilon}{m} & \text{en otro caso}
\end{cases}$$

Donde $m = |A|$ es número de acciones.

### 8.3 Epsilon Decay

La exploración es más importante **al inicio**:
- Inicio: no sabemos nada, exploración máxima
- Medio: vamos aprendiendo, balance
- Final: confiamos en lo aprendido, explotación pura

**Esquemas de decay**:

1. **Lineal**: $\varepsilon_t = \varepsilon_0 - \frac{\varepsilon_0 - \varepsilon_{min}}{T} \cdot t$
   - PES usa esto
   
2. **Exponencial**: $\varepsilon_t = \varepsilon_0 e^{-\lambda t}$
   
3. **Hiperbólico**: $\varepsilon_t = \frac{\varepsilon_0}{1 + \lambda t}$

**Análisis en PES**:
```
ε_0 = 0.8, ε_min = 0.0, T = 1,000,000
reduction = (0.8 - 0) / 1,000,000 = 0.0000008 por episodio

Inicio (t=0):     ε = 0.8   (80% exploración)
t=250,000:        ε = 0.6   (60% exploración)
t=500,000:        ε = 0.4   (40% exploración)
t=750,000:        ε = 0.2   (20% exploración)
Final (t=1M):     ε = 0.0   (0% exploración)
```

---

## 9. Off-Policy vs. On-Policy Learning

### 9.1 Definiciones

**On-Policy**: Aprender de la política que estamos siguiendo
- Algoritmo: SARSA
- Mejora convergencia lenta

**Off-Policy**: Aprender de política diferente
- Algoritmo: Q-Learning
- Permite exploración audaz

### 9.2 Q-Learning es Off-Policy

En Q-Learning, el **target** usa:
$$\max_{a'} Q(s', a')$$

No $\sum_{a'} \pi(a'|s') Q(s', a')$ (que sería on-policy)

Esto es **off-policy** porque:
- Estamos siguiendo una política exploratoria (ε-greedy)
- Pero aprendiendo la política greedy codificada en $\max Q$

**Ventaja**: Podemos explorar sin comprometer lo que aprendemos.

---

## 10. Función de Recompensa en Dominios Específicos

### 10.1 Diseño de Recompensas

La función de recompensa codifica el **objetivo del problema**.

En RL, el agente aprende a:
$$\max \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \ldots]$$

**Lo que haga el agente depende enteramente de R.**

### 10.2 Recompensa en PES

```python
# Objetivo: Minimizar severidades pandémicas
# Código en pandemic.py línea 330:

severities = get_updated_severity(...)  # Calcular nuevas severidades
reward = (-1) * numpy.sum(severities)   # Penalizar severidad

# Ejemplo:
# Si severidades = [2.5, 3.0, 1.8] → suma = 7.3
# reward = -7.3
```

**Análisis**:
- Recompensa negativa → agente quiere minimizar
- Proporcional a severidad total → incentiva buenas decisiones
- Inmediata → feedback rápido sobre cada acción

### 10.3 Shaping de Recompensas

En práctica, diseñar R es un **arte**:

1. **Recompensa muy esparsa** (ej. solo al final)
   - Problema: aprendizaje lento, muchos pasos sin feedback
   
2. **Recompensa muy densa** (ej. por cada acción)
   - Problema: agente puede buscar "atajos" (reward hacking)
   
3. **Balance**: Recompensa por progreso + recompensa final
   - En PES: severidad en cada step == balance automático

---

## 11. Entropía y Confianza Meta-cognitiva

### 11.1 Entropía de Shannon

La **entropía** mide incertidumbre en una distribución:

$$H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

Donde $p_i = P(X = x_i)$ es probabilidad.

**Propiedades**:
- $H = 0$ si una categoría tiene probabilidad 1 (determinística)
- $H = \log_2(n)$ máxima si todas categorías equiprobables
- Para 11 acciones: $H_{max} = \log_2(11) \approx 3.46$ bits

**Ejemplo**:
```python
# Distribución 1: Determinística
p1 = [1.0, 0.0, 0.0, ...]  # H = 0 bits (sin incertidumbre)

# Distribución 2: Uniforme
p2 = [1/11, 1/11, ..., 1/11]  # H ≈ 3.46 bits (máxima incertidumbre)

# Distribución 3: Concentrada
p3 = [0.7, 0.15, 0.1, 0.05, ...]  # H ≈ 1.2 bits (media)
```

### 11.2 Q-Values como Distribución

En estado $s$, los Q-values para cada acción:
$$Q_s = [Q(s,0), Q(s,1), \ldots, Q(s,10)]$$

Podemos normalizarlos a probabilidades (softmax):
$$p_a = \frac{e^{Q(s,a)}}{Z}$$

Donde $Z = \sum_a e^{Q(s,a)}$ es factor de normalización.

### 11.3 Confianza Basada en Entropía

**Intuición**:
- Si Q-values muestran **claro ganador** (ej. Q=[0.1, 0.9, 0.2, ...])
  - Entropía baja
  - Agente confiado
  
- Si Q-values **similares** (ej. Q=[0.4, 0.45, 0.42, ...])
  - Entropía alta
  - Agente inseguro

$$\text{Confianza} = 1 - \frac{H_{decision}}{H_{max}}$$

O normalización más sofisticada en PES:

$$\text{Confianza} = \frac{H_{decision} - H_{max}}{H_{min} - H_{max}}$$

### 11.4 Tiempos de Reacción Humano-realistas

La **confianza baja** debería correlacionar con **tiempos lentos**.

En PES (pandemic.py líneas 410-420):
```python
# Mapeo: confianza → tiempo de reacción
# confidence alta (>0.8) → respuesta rápida (<100ms)
# confidence baja (<0.3) → respuesta lenta (>500ms)

mu_response = confidence * coefficient + offset
rt_hold = numpy.random.normal(mu=mu_response, sigma=3)
```

Esto simula que un agente **cognitivamente realista** tarda más cuando es incierto.

---

## 12. Métricas de Evaluación

### 12.1 Return Acumulado

La métrica más fundamental es el **return acumulado**:

$$G = \sum_{t=0}^{T} r_t$$

En PES: suma de recompensas negativas = severidad total.

### 12.2 Performance Normalizado

Para comparar entre problema, normalizamos al rango teórico:

$$\text{Performance} = \frac{\text{worst} - \text{actual}}{\text{worst} - \text{best}}$$

- 0 = rendimiento más pobre posible
- 0.5 = mitad de lo óptimo
- 1 = rendimiento óptimo

### 12.3 Curva de Aprendizaje

Durante entrenamiento, graficar return promedio vs. episodios:

$$\text{Avg Return}_{\text{window}} = \frac{1}{N} \sum_{i=t-N/2}^{t+N/2} G_i$$

Esperado: curva creciente hasta convergencia en forma de **S**.

---

## 13. Comparación con Supervisado y No Supervisado

| Aspecto | Supervisado | No Supervisado | Refuerzo |
|--------|-----------|---------|----------|
| **Datos** | (X, y) etiquetados | Solo X | Trayectorias (s, a, r, s') |
| **Feedback** | Etiqueta correcta | Ninguno | Recompensa |
| **Objetivo** | Predecir y | Encontrar estructura | Maximizar return |
| **Ejemplo** | Clasificación | Clustering | Control, decisiones |
| **Desafío** | Overfitting | Validación | Exploración-Explotación |
| **Métrica** | Accuracy, F1 | Silhueta | Return promedio |

---

## 14. Limitations y Desafíos

### 14.1 Maldición de la Dimensionalidad

**Problema**: En espacios de estado grandes, tabla Q es infactible.

Ejemplo: Estado de 10 variables continuas, 100 valores cada una
- Tamaño de Q: $100^{10} \times |A|$ (¡¡¡astronómico!!!)

**Soluciones**:
1. Discretización (reducir precisión)
2. Function approximation (Red neuronal)
3. Dimensionalidad reducida

PES mitiga con discretización (31 × 11 × 10 = pequeño).

### 14.2 Non-Stationary Environments

Si el ambiente cambia con el tiempo (dinámica no-estacionaria), supuestos de MDP fallan.

En PES: Ambiente fijo (pandemia simulada), así que no problema.

### 14.3 Sample Efficiency

Q-Learning puede ser **muy ineficiente** (millions de muestras necesarios).

En robótica, obtener muestras es caro. Problemas:
- **Sample efficiency**: Minimizar muestras requeridas
- **Off-policy learning**: Reutilizar historiales
- **Model-based RL**: Aprender modelo del ambiente

### 14.4 Reward Hacking

Si recompensa mal diseñada, agente encuentra "truco" no intencional.

Ejemplo: En juego, si recompensa = velocidad, agente aprende a dar vueltas rápidamente (no objetivo).

---

## 15. Extensiones Modernas

### 15.1 Deep Q-Networks (DQN)

Reemplazar tabla Q con red neuronal:
$$Q(s, a; w) \approx Q^*(s, a)$$

Ventajas:
- Maneja espacios continuos
- Generaliza entre estados
- Permite transfer learning

Desventajas:
- Más parámetros
- Menos interpretable
- Requiere más datos

### 15.2 Policy Gradient Methods

Aprender política directamente, no vía Q-values:
$$\pi_\theta(a|s) = P(a|s; \theta)$$

Actualizar parámetros $\theta$ usando gradient:
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \cdot R$$

**Ventaja**: Aplica a espacios continuos de acción.

### 15.3 Actor-Critic

Combinar policy gradient + value function:
- **Actor**: Política (qué acción tomar)
- **Critic**: Función de valor (qué tan buena es acción)

---

## 16. Resumen: PES Implementa Q-Learning Clásico

```
┌──────────────────────────────────────────────────────┐
│ ARQUITECTURA DE PES                                  │
├──────────────────────────────────────────────────────┤
│ 1. Espacio de Estados (S)                            │
│    - [recursos disponibles, trial actual, severidad] │
│    - Tamaño: 31 × 11 × 10 = 3,410                   │
│                                                      │
│ 2. Espacio de Acciones (A)                           │
│    - Recursos a asignar: 0-10                        │
│    - Tamaño: 11 acciones                             │
│                                                      │
│ 3. Dinámica (P, R)                                   │
│    - Severidad evoluciona según fórmula              │
│    - Recompensa = -severidad_total                   │
│    - Determinista con respecto a recompensa          │
│                                                      │
│ 4. Entrenamiento: Q-Learning                         │
│    - 1,000,000 episodios                             │
│    - α = 0.2, γ = 0.9                                │
│    - ε-greedy con decay lineal                       │
│                                                      │
│ 5. Ejecución: Política Greedy                        │
│    - Consultar Q-table, tomar argmax                 │
│    - Confianza desde entropía Q-values               │
│    - Tiempos realistas                               │
│                                                      │
│ 6. Evaluación: Performance Normalizado                │
│    - (worst - actual) / (worst - best)               │
│    - Rango [0, 1]                                    │
└──────────────────────────────────────────────────────┘
```

---

## 17. Para Seguir Aprendiendo

### Libros Recomendados
1. **"Reinforcement Learning: An Introduction"** (Sutton & Barto, 2018)
   - Biblia de RL, matemática rigurosa
   
2. **"Deep Reinforcement Learning Hands-On"** (Lapan, 2020)
   - Implementación práctica con código
   
3. **"Probabilistic Graphical Models"** (Koller & Friedman, 2009)
   - Profundizar en probabilidades

### Cursos Online
- Stanford CS234: Reinforcement Learning
- DeepMind UCL Course on RL
- OpenAI Spinning Up in Deep RL

### Implementación
- OpenAI Gym: Entornos estándar
- Stable Baselines3: Algoritmos RL
- TensorFlow/PyTorch: Redes neuronales

---

## Conclusión

**Q-Learning es algoritmo fundamental** que aprende qué acción es mejor en cada estado.

En PES:
1. Entrenamos Q-table en 1M episodios
2. Aprendemos la política óptima (argmax Q)
3. La ejecutamos en el experimento real
4. Medimos cuán bien funciona

La teoría garantiza convergencia, datos validados por la práctica, y código implementa fielmente los conceptos matemáticos.

**Eso es RL aplicado.** 🚀
