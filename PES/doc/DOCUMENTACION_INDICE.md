# Índice de Documentación PES v2.0 - Ciencia de Datos

## Descripción General

Este conjunto de documentos proporciona una cobertura completa del sistema PES (Pandemic Experiment Scenario) desde dos perspectivas: **implementación práctica** y **teoría fundamental de Reinforcement Learning**.

---

## 📄 Documentos Disponibles

### 1. **EXPLICACION_PES.md** (951 líneas)
**Audiencia**: Desarrolladores, investigadores interesados en el código  
**Nivel**: Intermedio

**Contenido**:
- Visión general de PES v2.0 y arquitectura
- Fases de inicialización y ejecución
- Componentes del sistema (main loop, stimulus, response)
- Parámetros de configuración para ambos modos:
  - Modo `'human'`: Ejecución con pygame (30+ minutos)
  - Modo `'RL-Agent'`: Ejecución sin gráficos (3-5 minutos)
- **NUEVO**: Sección completa sobre **Entrenamiento del RL-Agent**
  - Cómo configurar parámetros de entrenamiento
  - Comando para ejecutar train_rl.py
  - Interpretación de resultados
  - Técnicas de mejora del entrenamiento

**Mejor para**:
- Entender cómo ejecutar PES en ambos modos
- Aprender a entrenar un nuevo RL-Agent
- Debugging de problemas operacionales

---

### 2. **RL_AGENT_IMPLEMENTATION.md** (506 líneas)
**Audiencia**: Científicos de datos, ingenieros ML  
**Nivel**: Avanzado

**Contenido**:
- Definición formal del problema como MDP (Markov Decision Process)
- Mapeo directo código → teoría RL:
  - Definición de estado (3 dimensiones)
  - Definición de acción (11 acciones posibles)
  - Definición de recompensa (función reward)
  - Estructura Q-Table (31×13×11×11)
- Ecuación de Bellman y su implementación
- Estrategia ε-Greedy y decaimiento
- Flujo de ejecución: Entrenamiento → Deployment
- Cálculo de confianza metacognitiva
- Relación componente-código-teoría (tabla de mapeo)
- Mejoras potenciales (Function Approximation, DQN, Actor-Critic)

**Mejor para**:
- Entender la relación entre código y teoría RL
- Analizar arquitectura técnica del RL-Agent
- Proponer mejoras o extensiones
- Debugging teórico

---

### 3. **REINFORCEMENT_LEARNING_THEORY.md** (590 líneas)
**Audiencia**: Académicos, investigadores, estudiantes de Master/PhD  
**Nivel**: Avanzado (Teórico)

**Contenido**:
- **Sección 1**: Introducción a RL y comparación con otros paradigmas
- **Sección 2**: Procesos de Decisión Markovianos (MDP)
  - Definición formal
  - Propiedad de Markov
  - Políticas (determinística vs estocástica)
- **Sección 3**: Retorno y funciones de valor
  - Retorno con descuento
  - Función de valor de estado V(s)
  - Función de valor de acción Q(s,a)
  - Relaciones y propiedades
- **Sección 4**: Ecuación de Bellman
  - Deducción matemática completa
  - Forma iterativa
  - TD Error
- **Sección 5**: Métodos de Programación Dinámica
  - Policy Iteration
  - Value Iteration
  - Limitaciones
- **Sección 6**: Métodos TD y Q-Learning
  - Aprendizaje por Diferencias Temporales
  - Q-Learning vs SARSA
  - Teorema de convergencia
- **Sección 7**: Exploración vs Explotación
  - ε-Greedy
  - ε-Decreciente
  - UCB (Upper Confidence Bound)
- **Sección 8**: Convergencia y garantías teóricas
- **Sección 9**: Aplicación específica en PES
- **Sección 10**: Extensiones y limitaciones
- **Sección 11**: Comparativa de métodos RL
- **Sección 12**: Ecuaciones resumidas
- **Sección 13**: Referencias teóricas

**Mejor para**:
- Entender los fundamentos teóricos de RL
- Aprender sobre convergencia y garantías
- Estudiar diferentes algoritmos (Policy Iteration, Q-Learning, DQN)
- Base teórica para investigación académica

---

## 🔄 Relación Entre Documentos

```
REINFORCEMENT_LEARNING_THEORY.md (Teoría Fundamental)
        ↓ Especialización práctica
RL_AGENT_IMPLEMENTATION.md (Mapeo Código-Teoría)
        ↓ Aplicación concreta
EXPLICACION_PES.md (Uso Práctico)
        ↓ Ejecución
PES/__main__.py (Sistema Operacional)
```

**Flujo de lectura recomendado**:
1. Si eres **programador**: Comienza con EXPLICACION_PES.md
2. Si eres **ingeniero ML**: Ve a RL_AGENT_IMPLEMENTATION.md
3. Si eres **investigador/académico**: Empieza con REINFORCEMENT_LEARNING_THEORY.md
4. Si quieres **comprensión completa**: Sigue el orden 1→2→3

---

## 📊 Estadísticas de Documentación

| Documento | Líneas | Tamaño | Secciones | Ecuaciones |
|-----------|--------|---------|-----------|-----------|
| EXPLICACION_PES.md | 951 | 33 KB | 15+ | 5 |
| RL_AGENT_IMPLEMENTATION.md | 506 | 18 KB | 8 | 15+ |
| REINFORCEMENT_LEARNING_THEORY.md | 590 | 18 KB | 14 | 30+ |
| **TOTAL** | **2,047** | **69 KB** | **37+** | **50+** |

---

## 🎯 Preguntas Frecuentes - Qué Leer

### "¿Cómo ejecuto PES?"
→ **EXPLICACION_PES.md**, Secciones 1-2

### "¿Cómo cambio entre modo human y RL-Agent?"
→ **EXPLICACION_PES.md**, Parámetros de Configuración

### "¿Cómo entreno un nuevo RL-Agent?"
→ **EXPLICACION_PES.md**, Sección ENTRENAMIENTO DEL RL-AGENT

### "¿Cómo funciona el RL-Agent internamente?"
→ **RL_AGENT_IMPLEMENTATION.md**, Secciones 2-4

### "¿Cuál es la relación entre el código y Q-Learning?"
→ **RL_AGENT_IMPLEMENTATION.md**, Sección 4

### "¿Qué garantías teóricas tiene el Q-Learning?"
→ **REINFORCEMENT_LEARNING_THEORY.md**, Sección 8

### "¿Cómo funciona la ecuación de Bellman?"
→ **REINFORCEMENT_LEARNING_THEORY.md**, Sección 4

### "¿Cuál es la diferencia entre exploración y explotación?"
→ **REINFORCEMENT_LEARNING_THEORY.md**, Sección 7

### "¿Qué es un MDP y por qué es importante?"
→ **REINFORCEMENT_LEARNING_THEORY.md**, Sección 2

### "¿Cómo puedo mejorar el rendimiento del RL-Agent?"
→ **RL_AGENT_IMPLEMENTATION.md**, Sección 7

---

## 🔧 Secciones por Tema

### Configuración y Ejecución
- **Modo de ejecución**: EXPLICACION_PES.md § 1-2
- **Parámetros de configuración**: EXPLICACION_PES.md § Parámetros
- **Cambiar entre modos**: EXPLICACION_PES.md § Uso Típico

### Entrenamiento del RL-Agent
- **Cómo entrenar**: EXPLICACION_PES.md § ENTRENAMIENTO
- **Parámetros de entrenamiento**: EXPLICACION_PES.md § Paso 1
- **Evaluar convergencia**: EXPLICACION_PES.md § Monitorización
- **Mejoras**: EXPLICACION_PES.md § Técnicas de Mejora

### Implementación Técnica
- **MDP definition**: RL_AGENT_IMPLEMENTATION.md § 2.1
- **Q-Table structure**: RL_AGENT_IMPLEMENTATION.md § 2.4
- **Q-Learning update**: RL_AGENT_IMPLEMENTATION.md § 4
- **Política greedy**: RL_AGENT_IMPLEMENTATION.md § 3.2
- **Confianza metacognitiva**: RL_AGENT_IMPLEMENTATION.md § 3.3

### Teoría Fundamental
- **MDP fundamentals**: REINFORCEMENT_LEARNING_THEORY.md § 2
- **Bellman equation**: REINFORCEMENT_LEARNING_THEORY.md § 4
- **Value functions**: REINFORCEMENT_LEARNING_THEORY.md § 3
- **Q-Learning algorithm**: REINFORCEMENT_LEARNING_THEORY.md § 6
- **Convergence proofs**: REINFORCEMENT_LEARNING_THEORY.md § 8
- **Exploration strategies**: REINFORCEMENT_LEARNING_THEORY.md § 7

---

## 📚 Complementary Files

Estos documentos complementan los tres principales:

| Archivo | Propósito |
|---------|-----------|
| EXPLICACION_PES_RL_AGENT.md | Descripción v1.0 del sistema |
| PES/ext/train_rl.py | Código de entrenamiento (línea 1-198) |
| PES/src/pygameMediator.py | Código de deployment (línea 981-1020) |
| PES/src/pandemic.py | Funciones de RL (línea 150-240) |
| PES/ext/train_rl.py | Algoritmo completo Q-Learning |

---

## 🎓 Niveles de Profundidad

### Nivel 1: Usuario (Ejecutar PES)
- Lee: EXPLICACION_PES.md (secciones 1-2)
- Tiempo: 15 minutos
- Resultado: Puedes ejecutar `python3 -m PES`

### Nivel 2: Practicante (Entrenar y Personalizar)
- Lee: EXPLICACION_PES.md (todas las secciones)
- Lee: RL_AGENT_IMPLEMENTATION.md (secciones 2-4)
- Tiempo: 1-2 horas
- Resultado: Entiendes arquitectura y puedes entrenar nuevos agentes

### Nivel 3: Ingeniero (Modificar y Extender)
- Lee: Todos los documentos
- Lee: Código fuente completo
- Tiempo: 4-6 horas
- Resultado: Puedes diseñar mejoras y extensiones

### Nivel 4: Investigador (Teoría y Publicaciones)
- Lee: REINFORCEMENT_LEARNING_THEORY.md completo
- Estudia: Referencias teóricas (§13)
- Tiempo: 8+ horas
- Resultado: Fundamento para investigación

---

## ✅ Completitud de Documentación

- ✅ Configuración dual (human/RL-Agent)
- ✅ Entrenamiento del RL-Agent paso a paso
- ✅ Mapeo completo código-teoría
- ✅ Marco teórico fundamentado
- ✅ Ecuaciones matemáticas
- ✅ Ejemplos de código
- ✅ Referencias teóricas
- ✅ Guía de troubleshooting
- ✅ Tablas comparativas
- ✅ Diagrama de flujo

---

## 📞 Guía de Uso Rápido

```bash
# Para ejecutar PES en modo RL-Agent (rápido)
python3 -m PES

# Para entrenar nuevo RL-Agent
python3 -m PES.ext.train_rl

# Para ejecutar en modo humano (visual)
# Editar CONFIG.py: PLAYER_TYPE = 1
python3 -m PES
```

---

## 🔗 Navegación Rápida

- **Teoría**: REINFORCEMENT_LEARNING_THEORY.md
- **Implementación**: RL_AGENT_IMPLEMENTATION.md
- **Práctica**: EXPLICACION_PES.md
- **Índice (este archivo)**: DOCUMENTACION_INDICE.md

---

**Última actualización**: 26 de enero de 2026  
**Versión del sistema**: PES v2.0  
**Estado**: Documentación completa

