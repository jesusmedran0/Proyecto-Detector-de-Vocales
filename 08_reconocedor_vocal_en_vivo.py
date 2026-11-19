import csv
import time
import numpy as np
from pathlib import Path
from scipy.linalg import solve_toeplitz
from scipy.signal import windows

# ============================================================
# PROGRAMA 08: RECONOCIMIENTO DE VOCALES EN VIVO
# El usuario pronuncia una vocal y el KNN dice cuál fue.
# Requiere: pip install sounddevice
# ============================================================

CARPETA_CAPTURAS = Path("capturas_vocales")
CARPETA_RESULTADOS = Path("resultados_formantes")

# Parámetros de análisis (iguales a los programas 06 y 07)
DURACION_FRAME = 0.030
ORDEN_LPC = 16
F_MIN = 90.0
F_MAX = 5000.0
ANCHO_BANDA_MAX = 400.0

# Características del KNN (iguales a las del programa 07)
CARACTERISTICAS = ["F1_Hz", "F2_Hz"]
K_VECINOS = 3
VOCALES = ["a", "e", "i", "o", "u"]

# Grabación
FS_POR_DEFECTO = 16000     # Se usa solo si no se detecta de los .npz
DURACION_GRABACION = 2.0   # Segundos que se graban
HOP_FRAME = 0.010          # Salto entre fragmentos analizados (s)
UMBRAL_ENERGIA = 0.35      # Fracción de la energía máxima para "voz"
UMBRAL_CONFIANZA = 0.60    # Por debajo se avisa de que no hay certeza


# ------------------------------------------------------------
# 1. Frecuencia de muestreo (igual a la del entrenamiento)
# ------------------------------------------------------------

def detectar_fs():
    for archivo in sorted(CARPETA_CAPTURAS.glob("*.npz")):
        try:
            with np.load(archivo, allow_pickle=False) as datos:
                return int(datos["fs"])
        except (KeyError, ValueError, OSError):
            continue
    return FS_POR_DEFECTO


# ------------------------------------------------------------
# 2. LPC y formantes (misma lógica que el programa 06)
# ------------------------------------------------------------

def calcular_lpc(senal, orden):
    senal = np.asarray(senal, dtype=np.float64)

    autocorrelacion = np.correlate(
        senal, senal, mode="full"
    )[len(senal) - 1:]

    if autocorrelacion[0] <= 1e-12:
        raise ValueError("Energía insuficiente en el fragmento.")

    r = autocorrelacion[:orden + 1]
    intermedios = solve_toeplitz(r[:orden], r[1:orden + 1])

    return np.concatenate(([1.0], -intermedios))


def estimar_formantes(coeficientes, fs):
    raices = np.roots(coeficientes)
    candidatos = []

    for raiz in raices:
        if np.imag(raiz) <= 0:
            continue

        frecuencia = np.angle(raiz) * fs / (2 * np.pi)
        radio = np.abs(raiz)

        if radio <= 0 or radio >= 1:
            continue

        ancho_banda = -fs / np.pi * np.log(radio)

        if (
            F_MIN <= frecuencia <= min(F_MAX, fs / 2)
            and 0 < ancho_banda <= ANCHO_BANDA_MAX
        ):
            candidatos.append((float(frecuencia), float(ancho_banda)))

    candidatos.sort(key=lambda x: x[0])
    return candidatos


def formantes_de_fragmento(fragmento, fs):
    """Devuelve [F1, F2, F3] (puede ser más corta) de un fragmento."""
    fragmento = fragmento.copy()
    fragmento -= np.mean(fragmento)

    if np.max(np.abs(fragmento)) < 1e-8:
        return []

    fragmento *= windows.hamming(len(fragmento), sym=False)
    orden = min(ORDEN_LPC, len(fragmento) // 2 - 1)

    try:
        coeficientes = calcular_lpc(fragmento, orden)
    except (ValueError, np.linalg.LinAlgError):
        return []

    return [f for f, _ in estimar_formantes(coeficientes, fs)[:3]]


# ------------------------------------------------------------
# 3. Grabación y selección de la zona estable de la vocal
# ------------------------------------------------------------

def grabar(fs):
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "Falta la biblioteca sounddevice. "
            "Instálala con: pip install sounddevice"
        )

    print("Prepárate...")
    time.sleep(0.7)
    print(">>> DI LA VOCAL (sostenida) <<<")

    audio = sd.rec(
        int(DURACION_GRABACION * fs),
        samplerate=fs,
        channels=1,
        dtype="float64",
    )
    sd.wait()
    print("Grabación terminada.")

    senal = audio[:, 0]
    senal = senal - np.mean(senal)

    pico = np.max(np.abs(senal))
    if pico < 1e-4:
        raise ValueError(
            "No se detectó sonido. Revisa el micrófono."
        )

    return senal / pico


def zona_de_voz(senal, fs):
    """
    Localiza la región central y estable de la vocal usando la
    energía por fragmentos. Devuelve (inicio, fin) en muestras.
    """
    n = int(DURACION_FRAME * fs)
    hop = int(HOP_FRAME * fs)

    inicios = np.arange(0, len(senal) - n, hop)
    energia = np.array([
        np.sqrt(np.mean(senal[i:i + n] ** 2)) for i in inicios
    ])

    activos = energia >= UMBRAL_ENERGIA * energia.max()

    # Tramo continuo más largo con energía suficiente.
    mejor = (0, 0)
    inicio = None
    for i, activo in enumerate(activos):
        if activo and inicio is None:
            inicio = i
        if (not activo or i == len(activos) - 1) and inicio is not None:
            fin = i + 1 if activo else i
            if fin - inicio > mejor[1] - mejor[0]:
                mejor = (inicio, fin)
            inicio = None

    a, b = mejor
    if b - a < 5:
        raise ValueError(
            "No se detectó una vocal sostenida. "
            "Habla más fuerte y mantén el sonido ~1 s."
        )

    # Recortar 20 % de cada extremo (ataque y caída de la vocal).
    margen = int(0.2 * (b - a))
    a += margen
    b -= margen

    return inicios[a], inicios[b - 1] + n


def extraer_formantes_en_vivo(senal, fs):
    """
    Analiza varios fragmentos de la zona estable y devuelve la
    mediana de cada formante, más cuántos fragmentos fueron válidos.
    """
    desde, hasta = zona_de_voz(senal, fs)
    n = int(DURACION_FRAME * fs)
    hop = int(HOP_FRAME * fs)

    muestras = []
    for i in range(desde, hasta - n + 1, hop):
        f = formantes_de_fragmento(senal[i:i + n], fs)
        if len(f) >= len(CARACTERISTICAS):
            muestras.append(f[:3] + [np.nan] * (3 - len(f)))

    if len(muestras) < 3:
        raise ValueError(
            "No se pudieron estimar formantes estables. "
            "Intenta de nuevo."
        )

    muestras = np.array(muestras, dtype=np.float64)
    medianas = np.nanmedian(muestras, axis=0)

    return {
        "F1_Hz": float(medianas[0]),
        "F2_Hz": float(medianas[1]),
        "F3_Hz": float(medianas[2]),
    }, len(muestras)


# ------------------------------------------------------------
# 4. KNN (igual al del programa 07)
# ------------------------------------------------------------

def cargar_dataset():
    X, y = [], []

    for ruta in sorted(CARPETA_RESULTADOS.glob("*_formantes.csv")):
        with open(ruta, newline="", encoding="utf-8-sig") as f:
            for fila in csv.DictReader(f):
                try:
                    valores = [float(fila[c]) for c in CARACTERISTICAS]
                except (ValueError, KeyError):
                    continue

                if fila["vocal"] in VOCALES:
                    X.append(valores)
                    y.append(fila["vocal"])

    if not X:
        raise RuntimeError(
            "No hay datos de entrenamiento. Ejecuta primero el "
            "programa 06 para varias vocales."
        )

    return np.array(X, dtype=np.float64), np.array(y)


def normalizar(X_train, X_test):
    media = X_train.mean(axis=0)
    desv = X_train.std(axis=0)
    desv[desv < 1e-9] = 1.0
    return (X_train - media) / desv, (X_test - media) / desv


def knn_predecir(X_train, y_train, x, k):
    k = min(k, len(X_train))
    distancias = np.linalg.norm(X_train - x, axis=1)
    indices = np.argsort(distancias)[:k]

    votos = {}
    for i in indices:
        peso = 1.0 / (distancias[i] + 1e-9)
        votos[y_train[i]] = votos.get(y_train[i], 0.0) + peso

    total = sum(votos.values())
    votos = {v: p / total for v, p in votos.items()}

    return max(votos, key=votos.get), votos


# ------------------------------------------------------------
# 5. Programa principal
# ------------------------------------------------------------

X, y = cargar_dataset()
fs = detectar_fs()

print("====================================")
print("RECONOCEDOR DE VOCALES EN VIVO")
print("====================================")
print(f"Muestras de entrenamiento: {len(X)}")
for v in VOCALES:
    print(f"  Vocal {v.upper()}: {int(np.sum(y == v))}")
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Características: {', '.join(CARACTERISTICAS)}  |  k = {K_VECINOS}")

if len(set(y)) < 2:
    print("\n[AVISO] Solo hay una vocal en los datos de entrenamiento.")

while True:
    orden = input(
        "\nPulsa ENTER para grabar una vocal (q para salir): "
    ).strip().lower()

    if orden == "q":
        break

    try:
        senal = grabar(fs)
        formantes, n_frames = extraer_formantes_en_vivo(senal, fs)
    except (ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        if isinstance(error, RuntimeError):
            break
        continue

    valores = np.array(
        [[formantes[c] for c in CARACTERISTICAS]], dtype=np.float64
    )
    X_tr, X_te = normalizar(X, valores)
    prediccion, votos = knn_predecir(X_tr, y, X_te[0], K_VECINOS)

    print("\n------------------------------------")
    print(
        "Formantes medidas: "
        + ", ".join(
            f"{c.replace('_Hz', '')}={formantes[c]:.0f} Hz"
            for c in CARACTERISTICAS
        )
        + f"  ({n_frames} fragmentos)"
    )
    print(f"\n   VOCAL DETECTADA: {prediccion.upper()}\n")

    for v, p in sorted(votos.items(), key=lambda t: -t[1]):
        print(f"   {v.upper()}: {100 * p:.0f} %")

    if votos[prediccion] < UMBRAL_CONFIANZA:
        print(
            "\n[AVISO] Confianza baja. Repite la vocal más clara "
            "y sostenida."
        )
    print("------------------------------------")

print("Hasta luego.")
