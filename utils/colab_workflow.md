# Flujo de trabajo en Google Colab Pro+

Guía para ejecutar entrenamiento y optimización Bayesiana de **pes_dqn** y
**pes_ac** en Google Colab Pro+, usando VS Code como interfaz.

[TOC]

---

## Motivación

La máquina local (Intel i3-6006U, 7.7 GB RAM, 3.9 GB swap) es insuficiente
para correr entrenamientos y optimizaciones completas. El kernel de Linux
(`systemd-oomd`) termina los procesos cuando la presión de memoria supera el
50% durante más de 20 segundos.

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
   autenticada.
3. **Google Drive** con la carpeta `mPES/` subida (incluye `requirements.txt`
   y los paquetes `pes_dqn/`, `pes_ac/`, etc.).

---

## Estructura en Drive

Ambos notebooks comparten la misma convención de directorios:

```
MyDrive/
├── mPES/                              ← repo source (código + inputs)
│   ├── requirements.txt
│   ├── pes_dqn/  pes_ac/  ...
└── mPES_results/                      ← outputs organizados
    └── {PACKAGE}/
        ├── train/
        │   └── {TRAIN_DATE}/          ← modelo .keras, rewards, plots
        └── bayesian/
            └── {RESUME_DATE}/         ← DB Optuna + config + plots
```

- **`mPES/`** es el código fuente (read-only en Colab: se copia a `/content/mPES`).
- **`mPES_results/`** almacena los resultados organizados por paquete, tipo de
  ejecución y fecha. Este directorio persiste en Drive entre desconexiones.

---

## Notebooks

### Train_Colab.ipynb — Entrenamiento

Entrena DQN o A2C desde cero con GPU. 10 celdas:

| #  | Tipo     | Propósito |
|----|----------|-----------|
| 1  | Markdown | Descripción, estructura en Drive y flujo |
| 2  | Código   | **Mount Drive** |
| 3  | Código   | **Configuración**: `PACKAGE`, `NUM_EPISODES`, `DRIVE_REPO`, `DRIVE_RESULTS`, `DRIVE_SYNC_MINUTES`, `NTFY_TOPIC` |
| 4  | Markdown | Encabezado "Setup del entorno" |
| 5  | Código   | **Setup**: copiar repo desde Drive, instalar Python 3.12 + venv, instalar `requirements.txt` |
| 6  | Código   | **Diagnóstico**: nvidia-smi, RAM, archivos requeridos |
| 7  | Markdown | Encabezado "Entrenar" |
| 8  | Código   | **Entrenamiento**: subproceso con GPU + sync periódico a Drive + notificaciones ntfy |
| 9  | Markdown | Encabezado "Resultados" |
| 10 | Código   | **Resultados**: copiar outputs a `mPES_results/{PACKAGE}/train/{DATE}/` |

**Parámetros de configuración:**

```python
PACKAGE = "pes_dqn"                                    # "pes_dqn" o "pes_ac"
NUM_EPISODES = 50000                                   # Episodios de entrenamiento
DRIVE_REPO = "/content/drive/MyDrive/mPES"             # Repo en Drive
DRIVE_RESULTS = "/content/drive/MyDrive/mPES_results"  # Resultados en Drive
DRIVE_SYNC_MINUTES = 15                                # Backup periódico (min)
NTFY_TOPIC = "mpes-train"                              # Push notifications
```

### Bayesian_Colab.ipynb — Optimización Bayesiana

Ejecuta Optuna (TPE) para encontrar hiperparámetros óptimos. 10 celdas:

| #  | Tipo     | Propósito |
|----|----------|-----------|
| 1  | Markdown | Descripción, estructura en Drive y flujo |
| 2  | Código   | **Mount Drive** |
| 3  | Código   | **Configuración**: `PACKAGE`, `N_TRIALS`, `RESUME_DATE`, `DRIVE_REPO`, `DRIVE_RESULTS`, `DRIVE_SYNC_MINUTES`, `NTFY_TOPIC`, `NTFY_EVERY_N` |
| 4  | Markdown | Encabezado "Setup del entorno" |
| 5  | Código   | **Setup**: copiar repo desde Drive, restaurar DB de Optuna, instalar Python 3.12 + venv, instalar `requirements.txt` |
| 6  | Código   | **Diagnóstico**: nvidia-smi, RAM, archivos requeridos, estado de la DB |
| 7  | Markdown | Encabezado "Ejecutar optimización" |
| 8  | Código   | **Optimización**: subproceso con GPU + sync DB a Drive cada N min + notificaciones ntfy |
| 9  | Markdown | Encabezado "Resultados" |
| 10 | Código   | **Resultados**: trials por estado, top 10, copiar outputs a `mPES_results/{PACKAGE}/bayesian/{DATE}/` |

**Parámetros de configuración:**

```python
PACKAGE = "pes_dqn"                                    # "pes_dqn" o "pes_ac"
N_TRIALS = 100                                         # Trials objetivo
RESUME_DATE = "2026-03-07"                             # Fecha del estudio a reanudar
DRIVE_REPO = "/content/drive/MyDrive/mPES"             # Repo en Drive
DRIVE_RESULTS = "/content/drive/MyDrive/mPES_results"  # Resultados en Drive
DRIVE_SYNC_MINUTES = 15                                # Sync DB cada N min
NTFY_TOPIC = "mpes-bayesian"                           # Push notifications
NTFY_EVERY_N = 5                                       # Notificar cada N trials
```

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

### Paso 2 — Configurar los parámetros

La celda 3 de cada notebook contiene los parámetros como formulario interactivo.
Ajustar `PACKAGE`, número de episodios/trials, y fechas según corresponda.

### Paso 3 — Ejecutar todas las celdas

Ejecutar las celdas en orden o usar **Run All**:

1. **Mount Drive** → monta Google Drive en `/content/drive`.
2. **Configuración** → define variables y paths derivados.
3. **Setup** → copia el repo desde Drive, instala Python 3.12 en un venv
   (`/content/mpes_env`) e instala todas las dependencias de `requirements.txt`.
4. **Diagnóstico** → verifica GPU, RAM y archivos requeridos.
5. **Ejecución** → lanza el subproceso con sync periódico a Drive.
6. **Resultados** → copia outputs finales a `mPES_results/`.

### Paso 4 — Monitorear

La salida del subproceso se muestra en tiempo real. Un hilo en segundo plano
copia los outputs a Drive cada `DRIVE_SYNC_MINUTES` minutos. Las notificaciones
push se envían via [ntfy.sh](https://ntfy.sh/).

---

## Manejo de desconexiones

Colab puede desconectarse por inactividad (~90 min sin interacción) o al
alcanzar el límite de 24h.

### Qué se preserva

- **Resultados en Drive:** Se sincronizan cada N minutos. Como máximo se
  pierden los outputs del último intervalo de sincronización.
- **Repo copiado:** Se pierde (disco efímero), pero se re-copia desde Drive
  al reejecutar.
- **DB de Optuna (Bayesian):** Se restaura desde
  `mPES_results/{PACKAGE}/bayesian/{DATE}/` y Optuna retoma automáticamente.

### Cómo reanudar

1. Reconectar a un runtime.
2. Ejecutar **todas las celdas** desde el principio.
3. El setup copia el repo y restaura el estado previo desde Drive.

> **Bayesian:** `load_if_exists=True` permite que Optuna reanude el estudio
> automáticamente.
>
> **Train:** El entrenamiento se reinicia desde cero (no hay checkpointing).
> Los archivos parciales del intento anterior quedan en Drive.

---

## Flujo de archivos

```
┌─────────────────────────────────────────────────────────────┐
│  Google Colab (efímero)                                     │
│                                                             │
│  /content/mPES/  ← copytree desde Drive                    │
│    └─ {PACKAGE}/inputs/{DATE}_{SUFFIX}/                    │
│        ├─ modelo .keras / DB Optuna                         │
│        ├─ rewards .npy                                      │
│        └─ config, plots...                                  │
│                                                             │
│  Sync cada N min ↕                                          │
│                                                             │
│  /content/drive/MyDrive/mPES_results/{PACKAGE}/            │
│    ├─ train/{DATE}/         ← outputs de entrenamiento     │
│    └─ bayesian/{DATE}/      ← outputs de optimización      │
└─────────────────────────────────────────────────────────────┘
                          │
                    Drive download
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Máquina local                                              │
│                                                             │
│  mPES/                                                      │
│    └─ {PACKAGE}/inputs/                                     │
│        ├─ dqn_model.keras / ac_actor.keras  ← modelo final │
│        ├─ rewards.npy                                       │
│        └─ {DATE}_{SUFFIX}/  ← estudio completo             │
└─────────────────────────────────────────────────────────────┘
```

---

## Entorno Python en Colab

Ambos notebooks instalan **Python 3.12** (misma versión que `linux_mpes_env`)
mediante `deadsnakes/ppa` y crean un venv en `/content/mpes_env`. Las
dependencias se instalan desde `requirements.txt` con versiones fijadas.

### GPU Override

El código fuente (en cada `__init__.py`) establece:

```python
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU only
```

El uso de `setdefault` hace que **no** sobreescriba la variable si ya existe.
Las celdas de ejecución definen `CUDA_VISIBLE_DEVICES='0'` en el entorno del
subproceso, habilitando la primera GPU.

### Optuna TPE + SQLite (Bayesian)

- **Sampler:** TPE (Tree-structured Parzen Estimator) — default de Optuna.
- **Storage:** SQLite local (`optuna_study_YYYY-MM-DD.db`).
- **Reanudación:** `load_if_exists=True` continúa con los trials existentes.
- Los trials con estado `RUNNING` (interrumpidos) se ignoran: Optuna no los
  reintenta, pero tampoco bloquean el estudio.

### Variables de entorno

| Variable | Valor | Propósito |
|----------|-------|-----------|
| `CUDA_VISIBLE_DEVICES` | `0` | Habilita GPU para TensorFlow |
| `VIRTUAL_ENV` | `/content/mpes_env` | Evita el prompt "Press ENTER" de `__init__.py` |
| `PYTHONUNBUFFERED` | `1` | Output en tiempo real |
| `PYTHONIOENCODING` | `utf-8` | Evita errores Unicode en output redirigido |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Suprime mensajes de oneDNN |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Solo warnings y errores de TF |

---

## Recuperar resultados localmente

Después de que la ejecución termine, descargar los resultados desde Google Drive:

1. Ir a `Mi unidad > mPES_results > {PACKAGE} > train/` o `bayesian/`.
2. Descargar la carpeta con la fecha correspondiente.
3. Copiar al directorio del repositorio local.

### Copiar modelo al path estándar (local)

```bash
# DQN — después de optimización
cp pes_dqn/inputs/{DATE}_BAYESIAN_OPT/dqn_best_{DATE}.keras \
   pes_dqn/inputs/dqn_model.keras

# AC — después de optimización
cp pes_ac/inputs/{DATE}_BAYESIAN_OPT/ac_best_{DATE}.keras \
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
| GPU no detectada | Verificar que el runtime tiene GPU: Runtime → Change runtime type → GPU. |
| DB no se restaura desde Drive | Verificar que `RESUME_DATE` coincide con la carpeta en `mPES_results/`. |
| OOM en Colab | Cambiar a runtime "High-RAM" o reducir batch size en la configuración del paquete. |
| Trial RUNNING infinito | Los trials interrumpidos quedan en estado RUNNING. Optuna los ignora y crea nuevos. |
| "Press ENTER to continue" | La variable `VIRTUAL_ENV` no se definió. Asegurarse de que la celda de configuración se ejecutó. |
| Repo no encontrado en Drive | Verificar que la carpeta `mPES/` está en la raíz de `MyDrive`. |
