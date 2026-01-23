"""
Notificaciones push vía ntfy.sh (https://ntfy.sh).

Instala la app «ntfy» en tu celular (Android/iOS) y suscríbete al
tema configurado en NTFY_TOPIC para recibir las notificaciones.

Uso desde Python (import absoluto desde cualquier paquete):
    from utils.notify import notify
    notify("Título", "Cuerpo del mensaje")

Uso desde bash (desde la raíz del workspace):
    python3 utils/notify.py "Título" "Cuerpo del mensaje"

Uso directo con curl (referencia):
    curl -H "Title: Mi título" -d "Mi mensaje" ntfy.sh/mpes-bayesian-max
"""

import sys
import urllib.request
import urllib.error
import base64

# ── Configuración ntfy ──────────────────────────────────────────
NTFY_SERVER = "https://ntfy.sh"
NTFY_TOPIC  = "mpes-bayesian-max"


def notify(title: str, body: str, *, priority: str = "default",
           tags: str = "") -> bool:
    """Envía una notificación push vía ntfy.sh.

    Parameters
    ----------
    title : str
        Título de la notificación.
    body : str
        Cuerpo del mensaje (texto plano).
    priority : str, optional
        Prioridad: "min", "low", "default", "high", "urgent".
    tags : str, optional
        Etiquetas/emojis separadas por coma (ej: "white_check_mark,robot").

    Returns
    -------
    bool
        True si el envío fue exitoso, False en caso de error.
    """
    try:
        url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
        data = body.encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        # Encode title as base64 to avoid latin-1 header encoding issues
        # with non-ASCII characters (ntfy supports RFC 2047-style encoding)
        title_b64 = base64.b64encode(title.encode("utf-8")).decode("ascii")
        req.add_header("Title", f"=?UTF-8?B?{title_b64}?=")
        req.add_header("Priority", priority)
        if tags:
            req.add_header("Tags", tags)

        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()

        return True
    except Exception as exc:
        print(f"[notify] ERROR al enviar notificación: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Uso: python3 {sys.argv[0]} <título> <cuerpo>", file=sys.stderr)
        sys.exit(1)

    ok = notify(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
