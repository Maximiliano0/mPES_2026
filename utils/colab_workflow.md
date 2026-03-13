# Optimización Bayesiana en Google Colab Pro+

Guía paso a paso para ejecutar las optimizaciones Bayesianas de **pes_dqn** y
**pes_ac** en Google Colab Pro+, usando VS Code como interfaz.

[TOC]

---

## Motivación

La máquina local (Intel i3-6006U, 7.7 GB RAM, 3.9 GB swap) es insuficiente
para correr las optimizaciones completas. El kernel de Linux (`systemd-oomd`)
termina los procesos cuando la presión de memoria supera el 50% durante más de
20 segundos. Cada optimización consume ~900 MB, y con el escritorio activo se
excede el límite rápidamente.

| Recurso         | Local         | Colab Pro+        |
|-----------------|---------------|-------------------|
| RAM             | 7.7 GB        | hasta 83 GB       |
| CPU             | i3-6006U 2C   | Xeon 2C/4C        |
| GPU             | —             | T4 / A100         |
| Runtime máximo  | ilimitado     | 24 h              |
| Disco           | SSD 256 GB    | ~200 GB efímero   |

> **Nota sobre GPU:** Los modelos DQN/A2C son pequeños (32–64 neuronas,
> 1–2 capas ocultas). La GPU no acelera significativamente el entrenamiento,
> pero las instancias GPU de Colab Pro+ incluyen más RAM y mejor CPU.

---

## Requisitos previos

1. **Cuenta Google** con suscripción **Colab Pro+**.
2. **Extensión de VS Code:** `google.colab` (Google Colab) instalada y
   autenticada con la cuenta `vegamaximiliano0@gmail.com`.
3. **Token de GitHub** (Personal Access Token) con permiso `repo` para hacer
   push automático de resultados. Crear en
   *GitHub → Settings → Developer settings → Personal access tokens*.
4. **Google Drive** con espacio suficiente (~500 MB por paquete).

---

## Estructura del notebook

El notebook `colab/mPES_Bayesian_Optimization.ipynb` tiene 11 celdas:

| #  | Tipo     | Propósito |
|----|----------|-----------|
| 1  | Markdown | Descripción y notas sobre reanudación |
| 2  | Código   | **Configuración**: paquete, trials, fecha, branch, token |
| 3  | Markdown | Encabezado "Setup del entorno" |
| 4  | Código   | **Setup**: clonar repo, restaurar DB desde Drive, instalar deps, verificar GPU |
| 5  | Código   | **Verificación**: nvidia-smi, RAM, archivos necesarios, estado de la DB |
| 6  | Markdown | Encabezado "Ejecutar optimización" |
| 7  | Código   | **Optimización**: subproceso con GPU + sincronización a Drive cada N min |
| 8  | Markdown | Encabezado "Resultados" |
| 9  | Código   | **Resultados**: trials por estado, top 10, copiar outputs a Drive |
| 10 | Código   | **Copiar modelo**: copiar `.keras` y `.npy` al path estándar |
| 11 | Código   | **Push a GitHub** (opcional): commit + push de resultados |

---

## Guía paso a paso

### Paso 1 — Conectarse a Colab desde VS Code

1. Abrir VS Code.
2. Abrir la paleta de comandos: `Ctrl+Shift+P`.
3. Buscar **"Google Colab: New Notebook"** o **"Google Colab: Connect to Runtime"**.
4. Seleccionar el tipo de runtime:
   - **Runtime type:** Python 3
   - **Hardware accelerator:** GPU (T4 recomendado)
   - **Runtime shape:** High-RAM si está disponible
5. Esperar a que VS Code se conecte al kernel remoto.

> Si ya tenés el notebook abierto, usá **"Google Colab: Connect to Runtime"**
> directamente.

### Paso 2 — Configurar la celda de parámetros

La celda #2 contiene los parámetros principales como formulario interactivo:

```python
PACKAGE = "pes_dqn"        # "pes_dqn" o "pes_ac"
N_TRIALS = 110              # Trials objetivo (DQN=110, AC=100)
RESUME_DATE = "2026-03-07"  # Fecha del estudio a reanudar
BRANCH = "dqn_and_ac"       # Rama de Git
DRIVE_SYNC_MINUTES = 5      # Intervalo de backup a Drive (min)
GITHUB_TOKEN = "Token_2026" # Reemplazar con tu Personal Access Token
```

| Parámetro | DQN | AC |
|-----------|-----|----|
| `PACKAGE` | `pes_dqn` | `pes_ac` |
| `N_TRIALS` | `110` | `100` |
| `RESUME_DATE` | `2026-03-07` | `2026-03-07` |

**Importante:** Escribir el `GITHUB_TOKEN` en la celda y **no** hacer commit
del token. La celda se limpia automáticamente antes de pushear.

### Paso 3 — Ejecutar todas las celdas

Ejecutar las celdas en orden o usar **Run All**:

1. **Celda 2 (Configuración):** Define variables y crea el directorio de backup
   en Drive.
2. **Celda 4 (Setup):**
   - Monta Google Drive en `/content/drive`.
   - Clona el repositorio (o hace `git pull` si ya existe).
   - Restaura la DB de Optuna desde Drive si es más reciente que la del repo.
   - Instala dependencias faltantes (`optuna`, `gym`, `pygame`, etc.).
   - Verifica TensorFlow y la GPU.
3. **Celda 5 (Verificación):**
   - Muestra info de GPU (`nvidia-smi`).
   - Muestra RAM total del sistema.
   - Verifica la existencia de archivos necesarios (DB, CSVs).
   - Imprime el estado actual de trials en la DB.

### Paso 4 — Monitorear la optimización

La celda 7 ejecuta la optimización como subproceso. La salida se muestra en
tiempo real:

```
============================================================
  python3 -m pes_dqn.ext.optimize_dqn 110 --resume 2026-03-07
  cwd: /content/mPES
  GPU: CUDA_VISIBLE_DEVICES=0
  Drive sync: cada 5 min
============================================================

[I 2026-03-11 ...] Trial 22 finished with value: 0.847...
```

Un hilo en segundo plano copia la DB a Google Drive cada `DRIVE_SYNC_MINUTES`
minutos como respaldo. Al finalizar, se hace un backup final.

**Tiempo estimado:**
- DQN (89 trials restantes): ~8–12 horas
- AC (62 trials restantes): ~6–10 horas

### Paso 5 — Revisar resultados

La celda 9 muestra un resumen de la optimización:
- Trials por estado (COMPLETE, FAIL, RUNNING).
- Top 10 trials ordenados por valor de la función objetivo.
- Copia todos los outputs al directorio de Drive.

### Paso 6 — Copiar el mejor modelo

La celda 10 copia el mejor modelo `.keras` al path estándar que espera
`__main__.py`:

| Paquete   | Archivo optimización       | Path estándar               |
|-----------|----------------------------|-----------------------------|
| `pes_dqn` | `dqn_best_*.keras`         | `pes_dqn/inputs/dqn_model.keras` |
| `pes_ac`  | `ac_best_*.keras`          | `pes_ac/inputs/ac_actor.keras`   |

También copia `rewards_best_*.npy` → `rewards.npy` y respalda el modelo en
Drive.

### Paso 7 — Push a GitHub (opcional)

La celda 11 hace `git add -A`, `commit` y `push` automático si se configuró
`GITHUB_TOKEN`. Si no, descargá los archivos desde Drive manualmente.

---

## Manejo de desconexiones

Colab puede desconectarse por inactividad (~90 min sin interacción) o al
alcanzar el límite de 24h.

### Qué se preserva

- **DB de Optuna en Drive:** Se sincroniza cada N minutos. Como máximo se
  pierden los trials del último intervalo de sincronización.
- **Repositorio clonado:** Se pierde (disco efímero), pero se re-clona al
  volver a ejecutar.

### Cómo reanudar

1. Reconectar a un runtime (Paso 1).
2. Ejecutar **todas las celdas** desde el principio.
3. La celda 4 restaura la DB desde Drive.
4. Optuna detecta los trials completados y retoma desde donde quedó.

> `load_if_exists=True` en el código de optimización permite reanudar
> automáticamente sin intervención manual adicional.

---

## Flujo de archivos

```
┌─────────────────────────────────────────────────────────────┐
│  Google Colab (efímero)                                     │
│                                                             │
│  /content/mPES/  ← git clone                                │
│    └─ pes_dqn/inputs/2026-03-07_BAYESIAN_OPT/              │
│        ├─ optuna_study_2026-03-07.db   ← Optuna escribe    │
│        ├─ dqn_best_2026-03-07.keras     ← mejor modelo     │
│        └─ rewards_best_2026-03-07.npy   ← rewards          │
│                                                             │
│  Sync cada 5 min ↕                                          │
│                                                             │
│  /content/drive/MyDrive/mPES_backups/pes_dqn/              │
│    ├─ optuna_study_2026-03-07.db        ← backup           │
│    └─ 2026-03-07_BAYESIAN_OPT/          ← outputs finales  │
│        ├─ optuna_study_2026-03-07.db                        │
│        ├─ dqn_best_2026-03-07.keras                         │
│        └─ rewards_best_2026-03-07.npy                       │
└─────────────────────────────────────────────────────────────┘
                          │
                    git push / Drive download
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Máquina local                                              │
│                                                             │
│  mPES/                                                      │
│    └─ pes_dqn/inputs/                                       │
│        ├─ dqn_model.keras               ← modelo final     │
│        ├─ rewards.npy                   ← rewards           │
│        └─ 2026-03-07_BAYESIAN_OPT/     ← todo el estudio   │
└─────────────────────────────────────────────────────────────┘
```

---

## Detalles técnicos

### GPU Override

El código fuente (en cada `__init__.py`) establece:

```python
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU only
```

El uso de `setdefault` hace que **no** sobreescriba la variable si ya existe.
La celda de optimización define `CUDA_VISIBLE_DEVICES='0'` en el entorno del
subproceso, habilitando la primera GPU.

### Optuna TPE + SQLite

- **Sampler:** TPE (Tree-structured Parzen Estimator) — default de Optuna.
- **Storage:** SQLite local (`optuna_study_YYYY-MM-DD.db`).
- **Reanudación:** `load_if_exists=True` continúa con los trials existentes.
- Los trials con estado `RUNNING` (interrumpidos) se ignoran: Optuna no los
  reintenta, pero tampoco bloquean el estudio.

### Variables de entorno en Colab

| Variable | Valor | Propósito |
|----------|-------|-----------|
| `CUDA_VISIBLE_DEVICES` | `0` | Habilita GPU para TensorFlow |
| `VIRTUAL_ENV` | `/content` | Evita el prompt "Press ENTER" de `__init__.py` |
| `PYTHONUNBUFFERED` | `1` | Output en tiempo real |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Suprime mensajes de oneDNN |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Solo warnings y errores de TF |

---

## Recuperar resultados localmente

Después de que la optimización termine, hay dos formas de traer los resultados
a la máquina local:

### Opción A: Git pull

Si se hizo push desde Colab (celda 11):

```bash
cd ~/Documentos/maximiliano/mPES
source linux_mpes_env/bin/activate
git pull origin dqn_and_ac
```

### Opción B: Descargar desde Google Drive

1. Ir a `Mi unidad > mPES_backups > pes_dqn` (o `pes_ac`).
2. Descargar los archivos `.keras`, `.npy` y `.db`.
3. Copiar al directorio correspon­diente en el repositorio local.

### Copiar modelo al path estándar (local)

Si el modelo no quedó en el path estándar, copiarlo manualmente:

```bash
# DQN
cp pes_dqn/inputs/2026-03-07_BAYESIAN_OPT/dqn_best_2026-03-07.keras \
   pes_dqn/inputs/dqn_model.keras

# AC
cp pes_ac/inputs/2026-03-07_BAYESIAN_OPT/ac_best_2026-03-07.keras \
   pes_ac/inputs/ac_actor.keras
```

### Ejecutar el experimento con el modelo optimizado

```bash
source linux_mpes_env/bin/activate
python3 -m pes_dqn   # o python3 -m pes_ac
```

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| "No module named google.colab" | Estás corriendo fuera de Colab. Conectate al runtime remoto. |
| GPU no detectada | Verificar que el runtime tiene GPU. Ir a Runtime → Change runtime type → GPU. |
| DB no se restaura desde Drive | El directorio de Drive puede no existir. Verificar que la fecha `RESUME_DATE` coincide. |
| `git push` falla | Verificar que `GITHUB_TOKEN` tiene permiso `repo` y no expiró. |
| OOM en Colab | Cambiar a runtime "High-RAM" o reducir el tamaño de batch en la configuración del paquete. |
| Trial RUNNING infinito | Los trials interrumpidos quedan en estado RUNNING. Optuna los ignora y crea nuevos. |
| "Press ENTER to continue" | La variable `VIRTUAL_ENV` no se definió. Asegurarse de que la celda de configuración se ejecutó. |
