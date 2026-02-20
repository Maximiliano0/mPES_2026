# Guía de Optimización Bayesiana — Q-Learning

## 1. Lanzar la optimización

```bash
cd /home/mecatronica/Documentos/maximiliano/mPES
source linux_mpes_env/bin/activate

# Lanzar con N trials en segundo plano (ejemplo: 100)
nohup python3 -m PES_Bayesian.ext.optimize_rl 100 \
  > PES_Bayesian/inputs/bayesian_opt.log 2>&1 &
```

Los resultados se guardan en `PES_Bayesian/inputs/<FECHA>_BAYESIAN_OPT/`.

---

## 2. Evitar que la PC se suspenda

Inmediatamente después de lanzar la optimización:

```bash
# Obtener el PID del proceso
pgrep -f "PES_Bayesian.ext.optimize_rl"

# Bloquear suspensión (reemplazar <PID> con el número obtenido)
nohup systemd-inhibit \
  --what=idle:sleep:handle-lid-switch \
  --who="mPES Bayesian Optimization" \
  --why="Running Bayesian optimization" \
  --mode=block \
  tail --pid=<PID> -f /dev/null > /dev/null 2>&1 &
```

El inhibidor se desactiva automáticamente cuando la optimización termina.

Si GNOME sigue suspendiendo al cerrar la tapa, desactivar eso explícitamente:

```bash
gsettings set org.gnome.settings-daemon.plugins.power lid-close-ac-action 'nothing'
```

---

## 3. Reanudar tras interrupción (--resume)

Cada trial completado se guarda en una base de datos SQLite (`optuna_study_<FECHA>.db`).
Si el proceso se interrumpe (suspensión, apagado, crash), los trials completados
se conservan.

**Mismo día** — re-ejecutar el mismo comando:

```bash
nohup python3 -m PES_Bayesian.ext.optimize_rl 100 \
  > PES_Bayesian/inputs/bayesian_opt.log 2>&1 &
```

**Día diferente** — usar `--resume` con la fecha original:

```bash
nohup python3 -m PES_Bayesian.ext.optimize_rl 100 --resume 2026-02-12 \
  > PES_Bayesian/inputs/bayesian_opt_resume.log 2>&1 &
```

El script detecta los trials previos y ejecuta solo los restantes:

```
ℹ Resuming: 45 trials already completed, 55 remaining
```

---

## 4. Interpretar los logs

### Ver progreso

```bash
grep "Trial" PES_Bayesian/inputs/bayesian_opt.log | tail -10
```

### Formato de salida

```
  Trial   1/100  |  value=0.7563  |  best=0.7563  |  elapsed=19s
  Trial   2/100  |  value=0.6941  |  best=0.7563  |  elapsed=76s
  Trial   8/100  |  value=0.8254  |  best=0.8254  |  elapsed=274s
```

| Campo | Significado |
|-------|-------------|
| `Trial N/100` | Número de trial actual / total solicitado |
| `value` | Rendimiento medio normalizado de este trial (0–1, mayor = mejor) |
| `best` | Mejor rendimiento encontrado hasta ahora entre todos los trials |
| `elapsed` | Tiempo transcurrido desde el inicio de esta corrida |

### Ver log en tiempo real

```bash
tail -f PES_Bayesian/inputs/bayesian_opt.log
```

### Verificar que el proceso sigue vivo

```bash
pgrep -f "PES_Bayesian.ext.optimize_rl" -a
```

Si no devuelve nada, el proceso terminó (completó todos los trials o fue interrumpido).
Revisar el final del log para determinar qué ocurrió:

```bash
tail -5 PES_Bayesian/inputs/bayesian_opt.log
```

### Tiempo estimado

- Varios minutos por trial (depende del `num_episodes` muestreado entre 800k y 1M)
- 100 trials puede tomar varias horas
- 200 trials puede tomar un día completo

Usar el script `utils/run_bayesian_opt.sh` para lanzar con inhibición de suspensión automática.
