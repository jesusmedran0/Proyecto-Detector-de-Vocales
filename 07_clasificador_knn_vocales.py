import csv
import numpy as np
from pathlib import Path
from scipy.linalg import solve_toeplitz
from scipy.signal import windows

# ============================================================
# PROGRAMA 07: CLASIFICACIÓN DE VOCALES CON KNN
# Usa las formantes (F1, F2, F3) obtenidas con el programa 06
# ============================================================

CARPETA_CAPTURAS = Path("capturas_vocales")
CARPETA_RESULTADOS = Path("resultados_formantes")

# Parámetros de análisis (deben coincidir con el programa 06)
INICIO_FRAME = 0.80
DURACION_FRAME = 0.030
ORDEN_LPC = 16
F_MIN = 90.0
F_MAX = 5000.0
ANCHO_BANDA_MAX = 400.0

# Características que usará el KNN. F1 y F2 son las más
# discriminativas; añade "F3_Hz" si quieres probar con tres.
CARACTERISTICAS = ["F1_Hz", "F2_Hz"]

K_VECINOS = 3
VOCALES = ["a", "e", "i", "o", "u"]


# ------------------------------------------------------------
# 1. Extracción de formantes (misma lógica que el programa 06)
# ------------------------------------------------------------

def calcular_lpc(senal, orden):
    senal = np.asarray(senal, dtype=np.float64)

    autocorrelacion = np.correlate(
        senal, senal, mode="full"
    )[len(senal) - 1:]

    if autocorrelacion[0] <= 1e-12:
        raise ValueError("Energía insuficiente en el fragmento.")

    r = autocorrelacion[:orden + 1]

    coeficientes_intermedios = solve_toeplitz(
        r[:orden], r[1:orden + 1]
    )

    return np.concatenate(([1.0], -coeficientes_intermedios))


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


def extraer_formantes(archivo):
    """
    Devuelve (diccionario con F1_Hz, F2_Hz, F3_Hz, vocal_real).
    """
    with np.load(archivo, allow_pickle=False) as datos:
        fs = int(datos["fs"])
        senal = np.asarray(datos["normalizada"], dtype=np.float64)
        vocal_real = str(datos["vocal"].item())

    if senal.ndim != 1 or not np.all(np.isfinite(senal)):
        raise ValueError("La señal no es válida.")

    inicio = int(INICIO_FRAME * fs)
    fin = inicio + int(DURACION_FRAME * fs)

    if fin > len(senal):
        raise ValueError("El fragmento excede la duración del audio.")

    fragmento = senal[inicio:fin].copy()
    fragmento -= np.mean(fragmento)

    if np.max(np.abs(fragmento)) < 1e-8:
        raise ValueError("Amplitud insuficiente.")

    fragmento *= windows.hamming(len(fragmento), sym=False)

    orden = min(ORDEN_LPC, len(fragmento) // 2 - 1)
    coeficientes = calcular_lpc(fragmento, orden)
    candidatos = estimar_formantes(coeficientes, fs)[:3]

    formantes = {}
    for i, nombre in enumerate(["F1_Hz", "F2_Hz", "F3_Hz"]):
        formantes[nombre] = candidatos[i][0] if i < len(candidatos) else None

    return formantes, vocal_real


# ------------------------------------------------------------
# 2. Cargar el conjunto de entrenamiento desde los CSV
# ------------------------------------------------------------

def cargar_dataset():
    """
    Lee todos los *_formantes.csv de resultados_formantes.
    Devuelve X (n, d), etiquetas, nombres de archivo y hablantes.
    """
    X, y, archivos, hablantes = [], [], [], []

    for ruta in sorted(CARPETA_RESULTADOS.glob("*_formantes.csv")):
        with open(ruta, newline="", encoding="utf-8-sig") as f:
            for fila in csv.DictReader(f):
                try:
                    valores = [float(fila[c]) for c in CARACTERISTICAS]
                except (ValueError, KeyError):
                    continue  # fila sin formantes completas

                if fila["vocal"] not in VOCALES:
                    continue

                X.append(valores)
                y.append(fila["vocal"])
                archivos.append(fila["archivo"])
                hablantes.append(fila["hablante"])

    if not X:
        raise RuntimeError(
            "No hay datos de entrenamiento. Ejecuta primero el "
            "programa 06 para varias vocales."
        )

    return (
        np.array(X, dtype=np.float64),
        np.array(y),
        np.array(archivos),
        np.array(hablantes),
    )


# ------------------------------------------------------------
# 3. KNN
# ------------------------------------------------------------

def normalizar(X_train, X_test):
    """Estandariza con la media y desviación del entrenamiento."""
    media = X_train.mean(axis=0)
    desv = X_train.std(axis=0)
    desv[desv < 1e-9] = 1.0
    return (X_train - media) / desv, (X_test - media) / desv


def knn_predecir(X_train, y_train, x, k):
    """
    Clasifica un punto x. Voto ponderado por 1/distancia.
    Devuelve (vocal_predicha, votos, vecinos).
    """
    k = min(k, len(X_train))
    distancias = np.linalg.norm(X_train - x, axis=1)
    indices = np.argsort(distancias)[:k]

    votos = {}
    for i in indices:
        peso = 1.0 / (distancias[i] + 1e-9)
        votos[y_train[i]] = votos.get(y_train[i], 0.0) + peso

    total = sum(votos.values())
    votos = {v: p / total for v, p in votos.items()}
    prediccion = max(votos, key=votos.get)

    vecinos = [(y_train[i], float(distancias[i])) for i in indices]
    return prediccion, votos, vecinos


# ------------------------------------------------------------
# 4. Evaluación (validación leave-one-out)
# ------------------------------------------------------------

def evaluar(X, y, k):
    n = len(X)
    clases = [v for v in VOCALES if v in set(y)]
    indice = {v: i for i, v in enumerate(clases)}
    confusion = np.zeros((len(clases), len(clases)), dtype=int)

    aciertos = 0
    for i in range(n):
        mascara = np.ones(n, dtype=bool)
        mascara[i] = False

        X_tr, X_te = normalizar(X[mascara], X[i:i + 1])
        pred, _, _ = knn_predecir(X_tr, y[mascara], X_te[0], k)

        confusion[indice[y[i]], indice[pred]] += 1
        aciertos += int(pred == y[i])

    print("\n====================================")
    print(f"EVALUACIÓN LEAVE-ONE-OUT (k={k}, n={n})")
    print("====================================")
    print(f"Exactitud: {100 * aciertos / n:.1f} %\n")

    print("Matriz de confusión (filas = real, columnas = predicha)")
    print("      " + "  ".join(f"{c:>3}" for c in clases))
    for c, fila in zip(clases, confusion):
        print(f"  {c:>3} " + "  ".join(f"{v:>3}" for v in fila))


# ------------------------------------------------------------
# 5. Clasificar una grabación
# ------------------------------------------------------------

def clasificar_grabacion(X, y, archivos, k):
    hablante = input("Hablante (ej. H01): ").strip().upper()
    vocal = input("Vocal grabada (a/e/i/o/u): ").strip().lower()
    rep = input("Número de repetición: ").strip()

    if not rep.isdigit():
        print("Repetición no válida.")
        return

    patron = f"{hablante}_{vocal}_rep{int(rep):02d}_*.npz"
    candidatos = sorted(CARPETA_CAPTURAS.glob(patron))

    if not candidatos:
        print("No se encontró ninguna grabación con ese patrón.")
        return

    archivo = candidatos[-1]

    try:
        formantes, vocal_real = extraer_formantes(archivo)
    except (ValueError, np.linalg.LinAlgError) as error:
        print(f"[ERROR] {archivo.name}: {error}")
        return

    valores = [formantes[c] for c in CARACTERISTICAS]

    if any(v is None for v in valores):
        print("No se pudieron estimar todas las formantes necesarias.")
        return

    # Excluir del entrenamiento esta misma grabación, si estaba.
    mascara = archivos != archivo.name
    X_tr, X_te = normalizar(
        X[mascara], np.array([valores], dtype=np.float64)
    )

    pred, votos, vecinos = knn_predecir(X_tr, y[mascara], X_te[0], k)

    print("\n====================================")
    print(f"Archivo: {archivo.name}")
    print(
        "Formantes: "
        + ", ".join(
            f"{c.replace('_Hz', '')}={v:.0f} Hz"
            for c, v in zip(CARACTERISTICAS, valores)
        )
    )
    print(f"Vocal predicha: {pred.upper()}")
    print(f"Vocal real (etiqueta del archivo): {vocal_real.upper()}")
    print("Confianza por vocal:")
    for v, p in sorted(votos.items(), key=lambda t: -t[1]):
        print(f"  {v.upper()}: {100 * p:.0f} %")
    print("Vecinos más cercanos (vocal, distancia normalizada):")
    for v, d in vecinos:
        print(f"  {v.upper()}  {d:.3f}")
    print("====================================")


# ------------------------------------------------------------
# 6. Menú principal
# ------------------------------------------------------------

X, y, archivos, hablantes = cargar_dataset()

print("Datos de entrenamiento cargados:")
for v in VOCALES:
    print(f"  Vocal {v.upper()}: {int(np.sum(y == v))} muestras")
print(f"Características: {', '.join(CARACTERISTICAS)}")

if len(set(y)) < 2:
    print(
        "\n[AVISO] Solo hay una vocal en el conjunto de datos. "
        "Procesa más vocales con el programa 06."
    )

while True:
    print("\n1) Evaluar el clasificador (leave-one-out)")
    print("2) Clasificar una grabación")
    print("3) Cambiar k (actual: " + str(K_VECINOS) + ")")
    print("4) Salir")
    opcion = input("Opción: ").strip()

    if opcion == "1":
        evaluar(X, y, K_VECINOS)
    elif opcion == "2":
        clasificar_grabacion(X, y, archivos, K_VECINOS)
    elif opcion == "3":
        nuevo = input("Nuevo valor de k: ").strip()
        if nuevo.isdigit() and int(nuevo) >= 1:
            K_VECINOS = int(nuevo)
        else:
            print("Valor no válido.")
    elif opcion == "4":
        break
    else:
        print("Opción no válida.")
