# Mejoras implementadas en Q-Learning: Teoría y Código

## Índice

1. [Introducción](#1-introducción)
2. [Double Q-Learning](#2-double-q-learning)
3. [Decaimiento exponencial de ε con warm-up](#3-decaimiento-exponencial-de-ε-con-warm-up)
4. [Reward Shaping basado en potencial (PBRS)](#4-reward-shaping-basado-en-potencial-pbrs)
5. [Integración de las tres mejoras](#5-integración-de-las-tres-mejoras)
6. [Referencias bibliográficas](#6-referencias-bibliográficas)

---

## 1. Introducción

Este documento describe las tres mejoras implementadas en la función
`QLearning()` del módulo `pandemic.py`, relacionando cada modificación
con su fundamento teórico y señalando las líneas de código donde se
materializa. Las tres mejoras son:

1. **Double Q-Learning** — elimina el sesgo de sobreestimación
2. **Decaimiento exponencial de ε con warm-up** — mejora la distribución
   temporal de exploración vs explotación
3. **Reward Shaping basado en potencial (PBRS)** — acelera la convergencia
   sin alterar la política óptima

### Contexto del problema

El entorno Pandemic es un MDP discreto con:

| Componente | Valor |
|---|---|
| Estados | $31 \times 11 \times 10 = 3{,}410$ |
| Acciones | 11 (asignar 0–10 recursos) |
| Steps por episodio | 3–10 |
| Recompensa | $r_t = -\sum_{i=0}^{t} S_i$ (negativo de la severidad total) |
| Transición | $S'_i = \max(0,\; 1.4 \cdot S_i - 0.4 \cdot a_i)$ |

Todo el código reside en `PES_QLv2/ext/pandemic.py`, en la función `QLearning()`.

---

## 2. Double Q-Learning

### 2.1 Teoría

#### El problema: sesgo de maximización

Q-Learning estándar (Watkins & Dayan, 1992) actualiza la Q-table con:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$$

El operador $\max$ cumple un doble rol: **selecciona** la mejor acción y
**evalúa** su valor usando la misma tabla. Cuando las estimaciones $Q(s',a')$
tienen ruido (inevitable al inicio del entrenamiento), $\max$ selecciona
sistemáticamente los valores con error positivo:

$$\mathbb{E}\!\left[\max_{a'} Q(s',a')\right] \geq \max_{a'} \mathbb{E}\!\left[Q(s',a')\right]$$

Esta desigualdad (consecuencia directa de la desigualdad de Jensen aplicada
al operador $\max$) produce **Q-valores inflados** que no corresponden al
rendimiento real. El resultado práctico es una política que sobreasigna
recursos a acciones que parecen buenas pero no lo son.

#### La solución: desacoplamiento selección–evaluación

Van Hasselt (2010) propuso mantener **dos Q-tables independientes** $Q_A$ y
$Q_B$. En cada step, con probabilidad 0.5 se actualiza una u otra:

$$Q_A(s,a) \leftarrow Q_A(s,a) + \alpha\left[r + \gamma\, Q_B\!\left(s',\, \underset{a'}{\arg\max}\; Q_A(s',a')\right) - Q_A(s,a)\right]$$

$$Q_B(s,a) \leftarrow Q_B(s,a) + \alpha\left[r + \gamma\, Q_A\!\left(s',\, \underset{a'}{\arg\max}\; Q_B(s',a')\right) - Q_B(s,a)\right]$$

El mecanismo es:
- Una tabla **selecciona** la acción (argmax): decide cuál es la mejor acción
- La otra tabla **evalúa** esa acción: estima cuánto vale

Como los errores de estimación de $Q_A$ y $Q_B$ son independientes (fueron
inicializadas con valores aleatorios distintos), el sesgo optimista se cancela
en promedio. Intuitivamente, si $Q_A$ sobreestima la acción $a^*$, es improbable
que $Q_B$ también la sobreestime por la misma cantidad.

#### Convergencia

Double Q-Learning converge a los Q-valores óptimos bajo las mismas condiciones
que Q-Learning estándar (programación estocástica de Robbins-Monro):

$$\sum_t \alpha_t = \infty, \quad \sum_t \alpha_t^2 < \infty$$

Con tasa de aprendizaje fija ($\alpha$ constante), converge a una vecindad del
óptimo cuyo radio depende de $\alpha$, igual que Q-Learning estándar.

### 2.2 Implementación en código

#### Inicialización de las dos tablas

```python
# pandemic.py — QLearning() — Inicialización
q_shape = (env.available_resources_states,   # 31
           env.trial_no_states,              # 11
           env.severity_states,              # 10
           env.action_space.n)               # 11

if double_q:
    Q_A = numpy.random.uniform(low=-1, high=1, size=q_shape)
    Q_B = numpy.random.uniform(low=-1, high=1, size=q_shape)
```

Las tablas se inicializan de forma **independiente** con distribución uniforme
$\mathcal{U}(-1, 1)$. La independencia es esencial: si ambas tablas empezaran
con los mismos valores, las primeras actualizaciones serían idénticas y el
desacoplamiento se perdería.

Cada tabla ocupa $31 \times 11 \times 10 \times 11 \times 8$ bytes $= 37{,}510$
bytes $\approx 37$ KB. El total durante entrenamiento es $\approx 74$ KB
(trivial para cualquier máquina).

#### Selección de acciones (ε-greedy)

```python
# pandemic.py — QLearning() — Selección de acción
if numpy.random.random() < 1 - epsilon and state[0] is not None:
    if double_q:
        action = numpy.argmax(Q_A[s] + Q_B[s])
    else:
        action = numpy.argmax(Q[s])
```

Se usa $\arg\max_a (Q_A(s,a) + Q_B(s,a))$ para seleccionar la acción greedy.
Esto es equivalente a $\arg\max_a \frac{Q_A + Q_B}{2}$ porque argmax es
invariante a escalado positivo. La suma de ambas tablas produce una estimación
más estable que cualquiera individualmente, suavizando los errores de cada una.

#### Actualización (el núcleo de Double Q-Learning)

```python
# pandemic.py — QLearning() — Actualización Double Q
if double_q:
    if done:
        if numpy.random.random() < 0.5:
            Q_A[s + (action,)] = reward
        else:
            Q_B[s + (action,)] = reward
    else:
        if numpy.random.random() < 0.5:
            # Update Q_A: Q_A selects best action, Q_B evaluates it
            best_action = numpy.argmax(Q_A[s2])
            target = reward + discount * Q_B[s2 + (best_action,)]
            Q_A[s + (action,)] += learning * (target - Q_A[s + (action,)])
        else:
            # Update Q_B: Q_B selects best action, Q_A evaluates it
            best_action = numpy.argmax(Q_B[s2])
            target = reward + discount * Q_A[s2 + (best_action,)]
            Q_B[s + (action,)] += learning * (target - Q_B[s + (action,)])
```

Correspondencia directa con la teoría:

| Paso teórico | Código |
|---|---|
| Con prob 0.5, actualizar $Q_A$ | `if numpy.random.random() < 0.5:` |
| $a^* = \arg\max_{a'} Q_A(s', a')$ | `best_action = numpy.argmax(Q_A[s2])` |
| $\text{target} = r + \gamma \cdot Q_B(s', a^*)$ | `target = reward + discount * Q_B[s2 + (best_action,)]` |
| $Q_A(s,a) \mathrel{+}= \alpha \cdot (\text{target} - Q_A(s,a))$ | `Q_A[s + (action,)] += learning * (target - Q_A[...])` |

En el estado terminal (`done=True`), se asigna $Q_X(s,a) = r$ directamente,
sin bootstrap, porque $V(s_{\text{terminal}}) = 0$ por definición.

#### Tabla final para evaluación

```python
# pandemic.py — QLearning() — Promediado final
if double_q:
    Q = (Q_A + Q_B) / 2.0
```

El archivo `q.npy` guardado contiene una sola tabla promediada. Esto no afecta
la política ya que $\arg\max_a \frac{Q_A + Q_B}{2} = \arg\max_a (Q_A + Q_B)$.

---

## 3. Decaimiento exponencial de ε con warm-up

### 3.1 Teoría

#### Exploración vs explotación

En ε-greedy, el parámetro $\varepsilon$ controla la probabilidad de tomar una
acción aleatoria (exploración) vs la mejor acción conocida (explotación):

$$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|A|} & \text{si } a = \arg\max_{a'} Q(s,a') \\ \frac{\varepsilon}{|A|} & \text{en caso contrario} \end{cases}$$

donde $|A| = 11$ es el número de acciones.

Un buen schedule de $\varepsilon$ debe explorar intensamente al inicio (cuando la
Q-table no contiene información útil) y explotar intensamente al final (cuando la
tabla ya convergió).

#### Decaimiento lineal vs exponencial

**Lineal** (implementación original):

$$\varepsilon_t^{\text{lin}} = \max\!\left(\varepsilon_{\min},\; \varepsilon_0 - t \cdot \frac{\varepsilon_0 - \varepsilon_{\min}}{N}\right)$$

Distribuye la exploración uniformemente en el tiempo: cada episodio reduce $\varepsilon$
en la misma cantidad $\Delta = (\varepsilon_0 - \varepsilon_{\min})/N$.

**Exponencial con warm-up** (implementación mejorada):

$$\varepsilon_t = \begin{cases} \varepsilon_0 & \text{si } t < W \quad \text{(warm-up)} \\ \max\!\left(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t-W}\right) & \text{si } t \geq W \end{cases}$$

El decaimiento exponencial tiene dos propiedades clave:

1. **Rápido al inicio**: la función $\varepsilon_0 \cdot \lambda^t$ decrece
   rápidamente cuando $t$ es pequeño, concentrando ~80% del descenso en el
   primer tercio del entrenamiento.

2. **Asintótico al final**: la tasa de cambio $\frac{d\varepsilon}{dt} = \varepsilon_0 \cdot \lambda^t \cdot \ln\lambda$
   tiende a cero, lo que produce una transición suave hacia explotación pura.

#### El warm-up: cobertura mínima del espacio de estados

La fase de warm-up ($t < W$) mantiene $\varepsilon = \varepsilon_0$ constante.
Su propósito es garantizar que el agente visite una fracción suficiente de los
3,410 estados antes de empezar a comprometerse con una política.

El criterio de cobertura mínima es:

$$W \geq \frac{|S|}{E[\text{estados por episodio}]}$$

Con $|S| = 3{,}410$ y ~6.5 estados/episodio: $W_{\min} \approx 525$ episodios.
Usando $W = 5\% \cdot N$ con $N = 1{,}000{,}000$, el warm-up dura 50,000
episodios ($\approx 95 \times W_{\min}$), garantizando cobertura amplia.

#### Calibración automática de λ

El parámetro $\lambda$ se calcula para que $\varepsilon$ alcance $\varepsilon_{\min}$
en un episodio objetivo $T = f \cdot N$ (fracción $f$ del total). Despejando:

$$\varepsilon_{\min} = \varepsilon_0 \cdot \lambda^{T - W}$$

$$\lambda = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{\frac{1}{T - W}} = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{\frac{1}{(f - w) \cdot N}}$$

donde $f$ es `target_ratio` y $w$ es `warmup_ratio`. Esta fórmula tiene la
ventaja de adaptar $\lambda$ automáticamente al número de episodios $N$: si se
duplica $N$, λ se recalcula para mantener las mismas proporciones de
exploración/explotación.

#### Ejemplo numérico

Con $\varepsilon_0 = 0.4915$, $\varepsilon_{\min} = 0.0710$, $N = 1{,}000{,}000$,
$w = 0.05$, $f = 0.66$:

$$\lambda = \left(\frac{0.0710}{0.4915}\right)^{\frac{1}{(0.66 - 0.05) \cdot 1{,}000{,}000}} = 0.1444^{1/610{,}000} \approx 0.9999968$$

| Episodio | Fracción de N | ε exponencial | ε lineal |
|---|---|---|---|
| 0 – 50,000 | 0 – 5% | 0.4915 (warm-up) | 0.4705 |
| 250,000 | 25% | 0.2608 | 0.3865 |
| 500,000 | 50% | 0.1264 | 0.2813 |
| 660,000 | 66% | **0.0710** (ε_min) | 0.2136 |
| 1,000,000 | 100% | 0.0710 | 0.0710 |

El exponencial alcanza $\varepsilon_{\min}$ al 66% de N, dejando 340,000 episodios
(34%) de explotación pura. El lineal no tiene zona de explotación pura — sigue
explorando hasta el último episodio.

### 3.2 Implementación en código

#### Cálculo automático de λ

```python
# pandemic.py — QLearning() — Configuración del decay
use_exponential_decay = (decay_rate != 'linear')

if use_exponential_decay:
    if decay_rate is None:
        decay_rate = (min_eps / epsilon) ** (1.0 / ((target_ratio - warmup_ratio) * episodes))
    warmup_episodes = int(warmup_ratio * episodes)
    target_episodes = int(target_ratio * episodes)
```

Correspondencia con la teoría:

| Fórmula | Código |
|---|---|
| $\varepsilon_{\min} / \varepsilon_0$ | `min_eps / epsilon` |
| $(f - w) \cdot N$ | `(target_ratio - warmup_ratio) * episodes` |
| $\lambda = (\varepsilon_{\min}/\varepsilon_0)^{1/((f-w)\cdot N)}$ | `(min_eps / epsilon) ** (1.0 / (...))` |
| $W = w \cdot N$ | `int(warmup_ratio * episodes)` |

Si se provee un `decay_rate` explícito, se usa directamente sin recalcular.
Si `decay_rate='linear'`, se usa el decaimiento lineal legacy.

#### Aplicación del decay en el bucle de entrenamiento

```python
# pandemic.py — QLearning() — Decay al final de cada episodio
if use_exponential_decay:
    if i < warmup_episodes:
        epsilon = epsilon_initial                                    # warm-up
    else:
        epsilon = max(min_eps, epsilon_initial * (decay_rate ** (i - warmup_episodes)))
else:
    if epsilon > min_eps:
        epsilon -= reduction                                         # lineal
```

Correspondencia directa:

| Fase teórica | Condición | Código |
|---|---|---|
| Warm-up: $\varepsilon = \varepsilon_0$ | $t < W$ | `if i < warmup_episodes: epsilon = epsilon_initial` |
| Exponencial: $\varepsilon_0 \cdot \lambda^{t-W}$ | $t \geq W$ | `epsilon_initial * (decay_rate ** (i - warmup_episodes))` |
| Piso: $\max(\varepsilon_{\min}, \cdot)$ | siempre | `max(min_eps, ...)` |

Nótese que `epsilon_initial` almacena el valor original de $\varepsilon_0$ para
recalcular desde la base en cada episodio, evitando la acumulación de errores
numéricos que ocurriría con `epsilon *= decay_rate`.

---

## 4. Reward Shaping basado en potencial (PBRS)

### 4.1 Teoría

#### El problema: falta de señal sobre el efecto compuesto

La recompensa original del entorno es $r_t = -\sum_{i=0}^{t} S_i$ (el negativo de
la severidad total acumulada tras aplicar la acción). Esta recompensa evalúa cada
step de forma local: el agente sabe cuánta severidad hay **ahora**, pero no recibe
feedback diferencial sobre si su acción **mejoró o empeoró** la situación respecto
al step anterior.

Con `SEVERITY_MULTIPLIER = 1.4`, una ciudad ignorada crece exponencialmente:

| Step | Severidad de una ciudad ignorada ($S_0 = 8$) |
|---|---|
| 0 | 8.0 |
| 1 | $1.4 \times 8.0 = 11.2$ |
| 2 | $1.4 \times 11.2 = 15.7$ |
| 3 | $1.4 \times 15.7 = 22.0$ |

Este crecimiento compuesto hace que la intervención temprana sea mucho más valiosa
que la tardía, pero la recompensa local no refleja esta diferencia con suficiente
claridad.

#### Reward shaping: modificar la recompensa sin cambiar la política

La idea del reward shaping es agregar una señal adicional a la recompensa para
guiar al agente durante el aprendizaje:

$$r'_t = r_t + F(s_t, s_{t+1})$$

donde $F$ es la función de shaping. El riesgo fundamental es que una función $F$
arbitraria puede **cambiar la política óptima** — el agente podría aprender a
maximizar $F$ en lugar de la recompensa real.

#### El teorema de Ng et al. (1999): invarianza de política

Ng, Harada y Russell (1999) demostraron que existe una forma específica de $F$
que **garantiza** que la política óptima no cambia. Se define una **función
potencial** $\Phi: S \to \mathbb{R}$ y la función de shaping como:

$$F(s, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

**Teorema (Ng et al., 1999):** Sea $M = (S, A, T, \gamma, R)$ un MDP con
recompensa $R$. Sea $\Phi: S \to \mathbb{R}$ cualquier función de potencial
acotada, y $M' = (S, A, T, \gamma, R')$ el MDP con recompensa modificada
$R'(s,a,s') = R(s,a,s') + \gamma \cdot \Phi(s') - \Phi(s)$. Entonces la
política óptima de $M'$ es idéntica a la política óptima de $M$.

La demostración se basa en que la suma telescópica de los potenciales a lo largo
de una trayectoria colapsa:

$$\sum_{t=0}^{T} \gamma^t F(s_t, s_{t+1}) = \sum_{t=0}^{T} \gamma^t \left[\gamma \Phi(s_{t+1}) - \Phi(s_t)\right] = \gamma^{T+1} \Phi(s_{T+1}) - \Phi(s_0)$$

El primer término es una constante que depende solo del estado terminal (y se
descuenta exponencialmente), y el segundo es una constante que depende solo del
estado inicial. Ninguno depende de las acciones tomadas, por lo que no afectan
qué política maximiza el retorno acumulado.

#### Por qué no un shaping aditivo ingenuo

Una penalización aditiva como $r' = r - \beta \sum S_i$ parece intuitiva, pero
colapsa a un simple escalado:

$$r' = -\sum S_i - \beta \sum S_i = -(1 + \beta) \sum S_i$$

Esto no introduce información nueva: solo amplifica la señal existente. Peor aún,
puede cambiar la política óptima en MDPs con descuento (porque el escalado no
uniforme de recompensas en diferentes steps altera los valores relativos de las
trayectorias).

El PBRS, en cambio, computa una **diferencia de potenciales** que captura
*cómo cambió* la severidad entre steps consecutivos, lo cual sí es información
nueva.

#### Elección de la función potencial

Para el entorno Pandemic, se eligió:

$$\Phi(s) = -\sum_{i=0}^{t} S_i$$

Es decir, el potencial es el negativo de la severidad total. La función de shaping
resultante es:

$$F(s, s') = \beta \left(\gamma \cdot \Phi(s') - \Phi(s)\right) = \beta \left(-\gamma \sum S_i^{\text{después}} + \sum S_i^{\text{antes}}\right)$$

$$= \beta \left(\sum S_i^{\text{antes}} - \gamma \sum S_i^{\text{después}}\right)$$

donde $\beta > 0$ es un coeficiente de escalado. La interpretación es:

- Si la acción **reduce** la severidad total: $\sum S_i^{\text{antes}} > \gamma \sum S_i^{\text{después}}$
  → $F > 0$ → **bonus**
- Si la acción **deja crecer** la severidad: $\sum S_i^{\text{antes}} < \gamma \sum S_i^{\text{después}}$
  → $F < 0$ → **penalización**

La presencia de $\gamma$ en el término del estado futuro refleja que la severidad
futura se descuenta (consistente con el MDP descontado).

#### Estado terminal

Se define $\Phi(s_{\text{terminal}}) = 0$. En el estado terminal, la función se
reduce a:

$$F(s, s_{\text{terminal}}) = \beta \left(0 - \Phi(s)\right) = \beta \sum S_i^{\text{antes}}$$

Esto penaliza terminar con severidad alta, incentivando al agente a resolver
las ciudades antes de que acabe la secuencia.

#### Efecto sobre la señal de aprendizaje

Con el crecimiento exponencial de severidades ($1.4\times$ por step), la
diferencia de potenciales se amplifica en cada step:

| Step | $\sum S_i$ antes | $\sum S_i$ después | $F$ ($\beta=0.1$, $\gamma=0.9$) |
|---|---|---|---|
| 0 | 8.0 | 18.2 | $0.1 \cdot (8.0 - 0.9 \cdot 18.2) = -0.84$ |
| 1 | 18.2 | 32.5 | $0.1 \cdot (18.2 - 0.9 \cdot 32.5) = -1.11$ |
| 2 | 32.5 | 52.5 | $0.1 \cdot (32.5 - 0.9 \cdot 52.5) = -1.50$ |

Las penalizaciones crecientes crean un **gradiente temporal** que incentiva la
intervención temprana: cuanto antes actúe el agente, menor es la penalización
total acumulada. Si el agente interviene y reduce la severidad, las penalizaciones
se convierten en bonus.

### 4.2 Implementación en código

#### Cálculo del potencial antes del step

```python
# pandemic.py — QLearning() — Potencial pre-step
if penalty_coeff > 0.0:
    phi_s = -sum(max(0.0, sv) for sv in env.severities)
```

$\Phi(s) = -\sum \max(0, S_i)$. Se usa `max(0, ·)` como protección contra
severidades negativas (que no deberían ocurrir por la transición
$S' = \max(0, 1.4 S - 0.4 a)$, pero se protege por robustez).

La variable `env.severities` contiene la lista de severidades de todas las
ciudades activas **antes** de ejecutar la acción.

#### Ejecución del step (sin modificar)

```python
# pandemic.py — QLearning() — Step del entorno
state2, reward, done, info = env.step(action)
```

El step se ejecuta sin modificaciones. El entorno retorna la recompensa
original $r_t = -\sum S_i^{\text{después}}$. El PBRS se aplica **después**,
en la capa de entrenamiento, no en el entorno.

Esta decisión de diseño es importante: si se modificara `step()`, las
recompensas distorsionadas afectarían a `pygameMediator.py` (visualización)
y `run_experiment()` (evaluación), confundiendo métricas de rendimiento
con señales de entrenamiento.

#### Cálculo del shaping reward

```python
# pandemic.py — QLearning() — PBRS (Ng et al., 1999)
if penalty_coeff > 0.0:
    if done:
        phi_s_prime = 0.0
    else:
        phi_s_prime = -sum(max(0.0, sv) for sv in env.severities)
    reward += penalty_coeff * (discount * phi_s_prime - phi_s)
```

Correspondencia con la teoría:

| Fórmula | Código |
|---|---|
| $\Phi(s) = -\sum S_i^{\text{antes}}$ | `phi_s = -sum(max(0.0, sv) for sv in env.severities)` (pre-step) |
| $\Phi(s') = -\sum S_i^{\text{después}}$ | `phi_s_prime = -sum(max(0.0, sv) for sv in env.severities)` (post-step) |
| $\Phi(s_{\text{terminal}}) = 0$ | `if done: phi_s_prime = 0.0` |
| $F = \beta(\gamma \Phi(s') - \Phi(s))$ | `penalty_coeff * (discount * phi_s_prime - phi_s)` |
| $r' = r + F$ | `reward += ...` |

Nótese que `env.severities` cambia entre el pre-step y el post-step: antes del
step contiene las severidades previas, y después del step contiene las nuevas
(actualizadas por `get_updated_severity()`).

#### Desactivación por defecto

```python
# pandemic.py — QLearning() — Firma de la función
def QLearning(env, learning, discount, epsilon, min_eps, episodes,
              decay_rate=None, warmup_ratio=0.05, target_ratio=0.66,
              double_q=True, penalty_coeff=0.0, ...):
```

Con `penalty_coeff=0.0` (valor por defecto), todas las secciones protegidas
por `if penalty_coeff > 0.0:` se saltan, y la función se comporta exactamente
igual que antes de la implementación. Esto garantiza retrocompatibilidad total.

---

## 5. Integración de las tres mejoras

### 5.1 Arquitectura combinada

Las tres mejoras están integradas en una sola función `QLearning()`. El flujo
de un episodio es:

```
┌─ Episodio i ────────────────────────────────────────────────────┐
│                                                                 │
│  1. Generar secuencia aleatoria                                 │
│  2. Reset del entorno                                           │
│                                                                 │
│  ┌─ Step t ──────────────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  a) Selección ε-greedy:                                    │  │
│  │     • ε determinado por decay exponencial + warm-up        │  │
│  │     • acción = argmax(Q_A + Q_B) con prob (1-ε)            │  │
│  │                                                            │  │
│  │  b) Calcular Φ(s) = −Σ sᵢ  (PBRS pre-step)               │  │
│  │                                                            │  │
│  │  c) Ejecutar env.step(action) → s', r, done                │  │
│  │                                                            │  │
│  │  d) Calcular Φ(s') y F = β(γ·Φ(s') − Φ(s))  (PBRS)       │  │
│  │     r' = r + F                                             │  │
│  │                                                            │  │
│  │  e) Double Q-Learning update:                              │  │
│  │     Con prob 0.5: Q_A[s,a] += α(r' + γ·Q_B[s',a*] − Q_A) │  │
│  │     Con prob 0.5: Q_B[s,a] += α(r' + γ·Q_A[s',a*] − Q_B) │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                 │
│  3. Decay ε (exponencial post-warmup)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Sinergia entre mejoras

Las tres mejoras no solo son independientes sino que se refuerzan mutuamente:

#### Double Q-Learning + Reward Shaping

El PBRS modifica la magnitud de las recompensas. Con Q-Learning estándar, las
recompensas amplificadas podrían exacerbar el sesgo de maximización (Q-valores
sobreestimados se inflan aún más). Double Q-Learning neutraliza este riesgo:
ambas tablas reciben la misma recompensa shaped, pero sus estimaciones
independientes cancelan el sesgo.

#### Double Q-Learning + Decaimiento exponencial

Double Q-Learning requiere más exploración inicial que el estándar, porque debe
llenar **dos** tablas con estimaciones independientes. El warm-up garantiza esta
exploración adicional, y el decaimiento exponencial la reduce rápidamente una
vez que ambas tablas tienen estimaciones razonables.

#### Reward Shaping + Decaimiento exponencial

El PBRS es más informativo cuando el agente explora transiciones diversas: si
solo explota una política fija, las diferencias de potencial se estabilizan y
la señal de shaping pierde variedad. El warm-up asegura que el agente
experimente muchas transiciones diferentes antes de comprometerse, generando
señales de shaping más ricas.

### 5.3 Hiperparámetros del modelo combinado

| Símbolo | Parámetro | Tipo | Rango | Origen |
|---|---|---|---|---|
| $\alpha$ | `learning_rate` | float | $[0.2, 0.4]$ | Original |
| $\gamma$ | `discount_factor` | float | $[0.80, 0.99]$ | Original |
| $\varepsilon_0$ | `epsilon_initial` | float | $[0.4, 1.0]$ | Original |
| $\varepsilon_{\min}$ | `epsilon_min` | float | $[0.05, 0.1]$ | Original |
| $N$ | `num_episodes` | int | $[800k, 1.2M]$ | Original |
| $w$ | `warmup_ratio` | float | $[0.01, 0.10]$ | Nuevo — Mejora 3 |
| $f$ | `target_ratio` | float | $[0.50, 0.80]$ | Nuevo — Mejora 3 |
| $\beta$ | `penalty_coeff` | float | $[0.001, 0.5]$ | Nuevo — Mejora 4 |

Los hiperparámetros $\lambda$ y $W$ no aparecen como parámetros independientes:
$\lambda$ se calcula automáticamente de $\varepsilon_0$, $\varepsilon_{\min}$, $w$, $f$ y $N$;
y $W = w \cdot N$.

La variable `double_q` no es un hiperparámetro optimizable: se fija en `True`
porque Double Q-Learning es estrictamente superior o igual a Q-Learning estándar
(nunca peor).

Todos los trials de entrenamiento usan la misma semilla (`SEED` de `CONFIG.py`,
default 42) para garantizar que las diferencias de rendimiento reflejen
exclusivamente los hiperparámetros.

---

## 6. Referencias bibliográficas

- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy invariance under reward
  transformations: Theory and application to reward shaping." *Proceedings of the
  16th International Conference on Machine Learning (ICML 1999)*, pp. 278–287.

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
  (2nd ed.). MIT Press. Capítulos 6.5 (ε-greedy), 6.7 (Q-learning).

- van Hasselt, H. (2010). "Double Q-learning." *Advances in Neural Information
  Processing Systems 23 (NeurIPS 2010)*.

- Watkins, C. J. C. H., & Dayan, P. (1992). "Q-learning." *Machine Learning*,
  8(3-4), 279–292.

- Brockman, G. et al. (2016). "OpenAI Gym." *arXiv:1606.01540*.

- Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic
  Dynamic Programming*. Wiley.

- Shannon, C. E. (1948). "A Mathematical Theory of Communication." *Bell System
  Technical Journal*, 27(3), 379–423.
