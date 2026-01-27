# Documentación de PES - Índice

## Bienvenida

Esta carpeta contiene documentación completa del **Pandemic Experiment Scenario (PES)**, un sistema de simulación autónomo basado en Reinforcement Learning para experimentación en asignación de recursos bajo pandemia.

---

## 📚 Documentos Principales

### 1. **HOWTO_PES.md** - Guía Operativa (Principiante)
**Audiencia**: Científicos de datos que necesitan **entender cómo funciona** la simulación

**Contenido**:
- Concepto fundamental de PES (problema, solución, matemática básica)
- Estructura jerárquica (bloques, secuencias, trials)
- Flujo de ejecución detallado con líneas de código referenciadas
- Configuración de parámetros
- Flujos de datos (entrada/salida)
- Análisis de resultados (gráficos, estadísticas)
- Modos de ejecución (experimento principal vs entrenamiento)
- Troubleshooting y debugging
- Ejemplos prácticos numéricos

**Secciones clave**:
```
1. Concepto Fundamental
2. Estructura Jerárquica (Bloques→Secuencias→Trials)
3. Flujo de Ejecución (Inicialización→Loop Principal→Análisis)
4. Parámetros de Configuración
5. Flujos de Datos
6. Análisis de Resultados
7. Modos de Ejecución
8. Integración del Agente RL
9. Validación y Debugging
10. Ejemplos Prácticos
11. Preguntas Frecuentes
12. Referencia de Código
```

**Cuándo leer**: 
- ✅ Necesitas entender cómo ejecutar un experimento
- ✅ Quieres saber qué hace cada parámetro
- ✅ Buscas interpretar resultados
- ❌ No necesitas: profundidad teórica de RL

---

### 2. **RL_AGENT.md** - Implementación del Agente (Intermedio)
**Audiencia**: Científicos de datos que necesitan **entender cómo el agente toma decisiones**

**Contenido**:
- Arquitectura general del agente RL
- Definición formal del problema como MDP
- Espacios de estados y acciones
- Función Q y algoritmo Q-Learning
- Implementación en código (clases, funciones, parámetros)
- Ciclo completo: Entrenamiento → Inferencia
- Transformación de acción a respuesta (metacognición, ruido, confianza)
- Optimización y mejora
- Análisis del aprendizaje
- Troubleshooting

**Secciones clave**:
```
1. Arquitectura General
2. Espacio de Estados y Acciones
3. Función Q y Q-Learning
4. Implementación del Agente RL
5. Inferencia (Ejecución)
6. Ciclo Completo: Training→Execution
7. Optimización y Mejora
8. Análisis del Aprendizaje
9. Troubleshooting
10. Comparación con Alternativas
11. Extensiones Futuras
```

**Cuándo leer**:
- ✅ Entiendes PES pero quieres saber cómo decide el agente
- ✅ Necesitas modificar parámetros de entrenamiento (learning rate, épocas)
- ✅ Quieres re-entrenar el agente
- ❌ No necesitas: matemática formal, derivaciones de ecuaciones

---

### 3. **RL_THEORY.md** - Fundamentos Teóricos (Avanzado)
**Audiencia**: Científicos de datos con **sólida base en probabilidad/cálculo** que necesitan teoría formal

**Contenido**:
- Conceptos fundamentales: MDP, políticas, value functions
- Bellman equations (derivaciones, intuición)
- Métodos de iteración sobre valores (value iteration, policy iteration)
- Q-Learning: ecuaciones, convergencia, pruebas
- Temporal Difference learning
- Deep RL (teórico)
- Teoría de exploración
- Convergencia y sample complexity
- Variantes y extensiones (Double Q-Learning, Dueling DQN, etc.)
- Análisis teórico específico a PES

**Secciones clave**:
```
1. Conceptos Fundamentales (MDP, Políticas, Value Functions)
2. Políticas y Value Functions
3. Métodos de Iteración sobre Valores
4. Q-Learning: Aprendizaje sin Modelo
5. Temporal Difference Learning
6. Deep Reinforcement Learning (DRL)
7. Teoría de Exploración
8. Casos de Convergencia Teórica
9. Variantes y Extensiones
10. Comparación de Algoritmos
11. Matemática Avanzada
12. Aplicación Específica a PES
```

**Cuándo leer**:
- ✅ Necesitas entender **por qué** Q-Learning funciona
- ✅ Quieres cambiar a otro algoritmo (DQN, Policy Gradient, etc.)
- ✅ Necesitas demostrar convergencia
- ❌ No necesitas: uso práctico, configuración

---

## 🔗 Relaciones Entre Documentos

```
Iniciante → Intermedio → Avanzado
   │           │             │
   ↓           ↓             ↓
HOWTO_PES → RL_AGENT → RL_THEORY
   (Qué)    (Cómo)      (Por qué)
```

### Flujo de Lectura Recomendado

**Scenario A: Solo necesito ejecutar experimentos**
```
HOWTO_PES (Sección 3, 7, 10)
```

**Scenario B: Necesito entender el agente**
```
HOWTO_PES (completo) → RL_AGENT (1-5) → RL_THEORY (1-4)
```

**Scenario C: Voy a mejorar/modificar el código**
```
HOWTO_PES (completo) → RL_AGENT (completo) → RL_THEORY (completo)
```

**Scenario D: Solo interesa teoría**
```
RL_THEORY (completo)
```

---

## 📊 Estadísticas de Documentación

| Documento | Líneas | Palabras* | Secciones | Complejidad |
|-----------|--------|-----------|-----------|------------|
| HOWTO_PES.md | 639 | ~4500 | 12 | Beginner |
| RL_AGENT.md | 879 | ~6200 | 11 | Intermediate |
| RL_THEORY.md | 729 | ~5300 | 13 | Advanced |
| **Total** | **2247** | **~16000** | **36** | - |

*Estimado

---

## 🎯 Guía de Referencias Cruzadas

### Concepto: "Q-Learning"

| Referencia en | Ubicación | Profundidad |
|---------------|-----------|-----------|
| HOWTO_PES | Sección 8 | Conceptual |
| RL_AGENT | Sección 3-4 | Implementación |
| RL_THEORY | Sección 4 | Matemática formal |

### Concepto: "Bellman Equation"

| Referencia en | Ubicación | Profundidad |
|---------------|-----------|-----------|
| HOWTO_PES | Sección 3.4 | Intuición |
| RL_AGENT | Sección 3.2 | Código |
| RL_THEORY | Sección 2, 4 | Derivación |

### Concepto: "Parámetros de Entrenamiento"

| Referencia en | Ubicación | Profundidad |
|---------------|-----------|-----------|
| HOWTO_PES | Sección 4 | Configuración |
| RL_AGENT | Sección 3.3, 7.2 | Ajuste |
| RL_THEORY | Sección 8.2 | Justificación |

---

## 🔧 Archivos de Código Referenciados

Los documentos hacen referencia a archivos clave:

```
PES/
├─ __main__.py          (Líneas referenciadas en HOWTO_PES, RL_AGENT)
├─ src/
│  ├─ Agent.py         (Líneas: 24-70, en RL_AGENT)
│  ├─ exp_utils.py     (Líneas: 200-300, en HOWTO_PES)
│  └─ log_utils.py     (Auxiliar)
├─ ext/
│  ├─ train_rl.py      (Líneas: 35-100, en RL_AGENT, RL_THEORY)
│  ├─ pandemic.py      (Líneas: 42-330, en RL_AGENT, RL_THEORY)
│  └─ tools.py         (Auxiliar)
└─ config/
   └─ CONFIG.py        (Líneas: 1-100, en HOWTO_PES)
```

---

## 💡 Tips de Navegación

### Buscar por Concepto

```bash
grep -r "Q-Learning" *.md          # Encuentra todas referencias
grep -r "Bellman" *.md             # Encuentra todas referencias
grep -r "exploit" *.md             # Encuentra exploración vs explotación
```

### Crear Tu Versión Personalizada

Cada documento está diseñado para ser **modular**. Puedes:

1. Copiar secciones a tu propio documento
2. Adaptar ejemplos a tus datos
3. Crear diagrama con referencias específicas

---

## 🚀 Casos de Uso

### Caso 1: Investigador Científico
**Objetivo**: Publicar resultados de experimento

**Leer**:
1. HOWTO_PES (Sec 1-3, 6) → Entender simulación
2. RL_THEORY (Sec 1-4) → Fundamentar teoría
3. RL_AGENT (Sec 1-2) → Describir método

**Escribir en paper**:
> "Implementamos Q-Learning (Watkins & Dayan, 1992) para entrenar un agente 
> de asignación de recursos bajo restricciones pandémicas. El agente usa una 
> tabla Q de tamaño 5863×11 entrenada con 20,000 episodios..."

### Caso 2: Ingeniero de ML
**Objetivo**: Mejorar el agente RL

**Leer**:
1. RL_AGENT (completo)
2. RL_THEORY (Sec 6-9) → Variantes
3. HOWTO_PES (Sec 7-9) → Testing

**Cambios potenciales**:
- Cambiar a Double Q-Learning (reducir sobrestimación)
- Implementar DQN (para espacios más grandes)
- Agregar experience replay

### Caso 3: Estudiante de RL
**Objetivo**: Aprender RL en contexto real

**Leer en orden**:
1. RL_THEORY (Sec 1-4)
2. RL_AGENT (Sec 1-5)
3. HOWTO_PES (Sec 3, 8)
4. Código real en `PES/ext/train_rl.py`

---

## 📝 Preguntas Frecuentes Generales

### P: ¿Por dónde empiezo?

**R**: Depende de tu rol:
- Si eres **ejecutor**: HOWTO_PES Sec 7
- Si eres **analista**: HOWTO_PES completo
- Si eres **desarrollador**: Todo en orden (HOWTO→RL_AGENT→RL_THEORY)
- Si eres **académico**: RL_THEORY + HOWTO_PES

### P: ¿Necesito leer todo?

**R**: No. Los documentos están diseñados para lectura **a la carta**:
- Tabla de contenidos clara
- Secciones independientes
- Referencias cruzadas para profundizar

### P: ¿Dónde están las derivaciones matemáticas?

**R**: En **RL_THEORY**. HOWTO_PES y RL_AGENT son más prácticos.

### P: ¿Cómo contribuir/actualizar documentación?

**R**: 
1. Identificar sección a actualizar
2. Mantener nivel de complejidad consistente
3. Agregar referencias a código
4. Mantener estructura de secciones

---

## 📚 Recursos Adicionales Recomendados

### Textbooks
- **Sutton & Barto (2018)**: "Reinforcement Learning: An Introduction"
  - Mejor para: Aprender RL desde principios
  
- **Puterman (1994)**: "Markov Decision Processes: Discrete Stochastic Dynamic Programming"
  - Mejor para: Profundidad matemática

### Papers Seminales
- Watkins & Dayan (1992): "Q-Learning" (IEEE Transactions on Machine Learning)
- Mnih et al. (2015): "Human-level control through deep reinforcement learning" (Nature)

### Cursos Online
- **David Silver's RL Course** (UCL): Introduction to RL (video lectures)
- **OpenAI Spinning Up**: Practical deep RL
- **DeepMind RL Courses**: Advanced topics

---

## 📞 Contacto y Soporte

Para preguntas específicas sobre:

- **Configuración/Ejecución**: Ver HOWTO_PES Sec 9-10
- **Modificación del agente**: Ver RL_AGENT Sec 6-7
- **Teoría subyacente**: Ver RL_THEORY Sec 1-4
- **Troubleshooting**: Ver respective docs Sec 8-9

---

## Versión y Actualización

- **Versión**: 1.0
- **Fecha**: 27 de enero de 2026
- **Estado**: Completo
- **Próxima actualización**: Pendiente
  - Deep RL guide (DQN, PPO)
  - Multi-agent extension
  - Benchmark results

---

## Licencia y Atribución

Documentación generada para el proyecto **Pandemic Experiment Scenario (PES)**.

Basada en código original y teoría de Reinforcement Learning.

---

**Última actualización**: 27 de enero de 2026

**Tamaño total de documentación**: ~2247 líneas, ~16,000 palabras
