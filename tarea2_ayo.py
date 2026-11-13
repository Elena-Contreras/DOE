import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

MATRICULA = 2225402
random.seed(MATRICULA)
r = random.random()

T_MIN, T_MAX = 1, 100
TAMANOS = [5, 10, 15]
REPLICAS = 10
N = 3000       
T_FIN = 0.01    
ALPHA = 0.05

fracciones = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
temperaturas = {nom: T_MIN + f * (T_MAX - T_MIN) for nom, f in fracciones.items()}
orden_temp = ["bajo", "medio", "alto"]

print(f"r = {r:.4f}")
for nom in fracciones:
    print(f"  {nom:>5}: fraccion = {fracciones[nom]:.4f} -> T0 = {temperaturas[nom]:.2f}")

instancias = {}
for n in TAMANOS:
    rc = random.Random(n)
    instancias[n] = [(rc.uniform(0, 100), rc.uniform(0, 100)) for _ in range(n)]

def tour_length(pts, ruta):
    return sum(math.dist(pts[ruta[i]], pts[ruta[(i + 1) % len(ruta)]])
               for i in range(len(ruta)))

def two_opt_move(ruta, i, k):
    """Reverse ruta[i..k]."""
    return ruta[:i] + ruta[i:k + 1][::-1] + ruta[k + 1:]

def simulated_annealing(pts, initial, schedule, max_iter, rng):
    current = initial[:]
    cur_value = -tour_length(pts, current)
    best, best_len = current[:], -cur_value
    for t in range(1, max_iter + 1):
        T = schedule(t)
        i, k = sorted(rng.sample(range(1, len(current)), 2))
        nxt = two_opt_move(current, i, k)
        next_value = -tour_length(pts, nxt)
        delta_E = next_value - cur_value
        if delta_E > 0:
            current, cur_value = nxt, next_value
        elif rng.random() < math.exp(delta_E / T):
            current, cur_value = nxt, next_value
        if -cur_value < best_len:
            best, best_len = current[:], -cur_value
    return best, best_len

def esquema_exponencial(T0):
    return lambda t: T0 * (T_FIN / T0) ** (t / N)

filas = []
corrida = 0
for n in TAMANOS:
    for nom in orden_temp:
        T0 = temperaturas[nom]
        sch = esquema_exponencial(T0)
        for rep in range(REPLICAS):
            corrida += 1
            rng_sa = random.Random(corrida) 
            inicial = list(range(n))
            rng_sa.shuffle(inicial)        
            _, L_fin = simulated_annealing(instancias[n], inicial, sch, N, rng_sa)
            filas.append({"temperatura": nom, "T0": T0, "tamano": n, "costo": L_fin})

df = pd.DataFrame(filas)
print(f"\nTotal de corridas: {len(df)}  (iteraciones por corrida: {N})")

modelo = ols('costo ~ C(temperatura) * C(tamano)', data=df).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print("\n=== ANOVA de dos factores ===")
print(tabla_anova)

p_temp = tabla_anova.loc["C(temperatura)", "PR(>F)"]
p_tam = tabla_anova.loc["C(tamano)", "PR(>F)"]
p_int = tabla_anova.loc["C(temperatura):C(tamano)", "PR(>F)"]
print(f"\nTemperatura : p = {p_temp:.4g} -> {'SIGNIFICATIVO' if p_temp < ALPHA else 'no significativo'}")
print(f"Tamaño      : p = {p_tam:.4g} -> {'SIGNIFICATIVO' if p_tam < ALPHA else 'no significativo'}")
print(f"Interaccion : p = {p_int:.4g} -> {'SIGNIFICATIVO' if p_int < ALPHA else 'no significativo'}")

rng_np = np.random.default_rng(MATRICULA)

def bootstrap_mediana(datos, B=10_000):
    datos = np.asarray(datos)
    meds = np.median(rng_np.choice(datos, size=(B, len(datos)), replace=True), axis=1)
    return float(np.median(meds)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

significativo = (p_temp < ALPHA) or (p_int < ALPHA)
print("\n=== Bootstrap de la mediana por punto de diseño ===")
if significativo:
    print("La temperatura o la interaccion fueron significativas (p < 0.05): se reporta el bootstrap.")
else:
    print("Ni la temperatura ni la interaccion fueron significativas (p >= 0.05).")
    print("Se reporta el bootstrap de todos modos para poder llenar la tabla final.")

boot = {}
print(f"{'n':>3} {'Temp':>6} {'T0':>7} {'Mediana boot':>13} {'IC 95%':>22} {'Media':>8}")
for n in TAMANOS:
    for nom in orden_temp:
        d = df[(df.tamano == n) & (df.temperatura == nom)]["costo"]
        med, lo, hi = bootstrap_mediana(d)
        boot[(n, nom)] = (med, lo, hi, d.mean())
        print(f"{n:>3} {nom:>6} {temperaturas[nom]:>7.2f} {med:>13.2f} [{lo:>8.2f}, {hi:>8.2f}] {d.mean():>8.2f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
m_temp = df.groupby("temperatura")["costo"].mean().reindex(orden_temp)
axs[0].plot(orden_temp, m_temp.values, "o-", color="tab:blue")
axs[0].set_xlabel("Nivel de temperatura")
axs[0].set_ylabel("Costo promedio")
axs[0].set_title("Efecto principal: temperatura")
m_tam = df.groupby("tamano")["costo"].mean()
axs[1].plot(m_tam.index, m_tam.values, "s-", color="tab:green")
axs[1].set_xticks(TAMANOS)
axs[1].set_xlabel("Tamaño del problema (n)")
axs[1].set_ylabel("Costo promedio")
axs[1].set_title("Efecto principal: tamaño")
plt.tight_layout()
plt.savefig("efectos_principales.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for n in TAMANOS:
    medias = df[df.tamano == n].groupby("temperatura")["costo"].mean().reindex(orden_temp)
    ax.plot(orden_temp, medias.values, "o-", label=f"n = {n}")
ax.set_xlabel("Nivel de temperatura")
ax.set_ylabel("Costo promedio")
ax.set_title("Interaccion temperatura x tamaño")
ax.legend()
plt.tight_layout()
plt.savefig("interaccion.png", dpi=150)
plt.show()

print("\n=== TABLA RESUMEN: temperatura óptima por tamaño ===")
print(f"{'n':>3} {'Nivel':>6} {'T* (T0)':>9} {'Mediana boot':>13} {'IC 95%':>22} {'Empate':>8}")
resumen = []
for n in TAMANOS:
    clave = lambda nom: (round(boot[(n, nom)][0], 6), 
                         round(boot[(n, nom)][3], 6),  
                         round(boot[(n, nom)][2], 6))  
    mejor_nom = min(orden_temp, key=clave)
    med, lo, hi, media = boot[(n, mejor_nom)]
    empate = sum(1 for nom in orden_temp
                 if clave(nom)[:2] == clave(mejor_nom)[:2]) > 1
    resumen.append({"n": n, "nivel": mejor_nom, "T*": temperaturas[mejor_nom],
                    "mediana": med, "IC_inf": lo, "IC_sup": hi, "media": media,
                    "empate_tecnico": empate})
    print(f"{n:>3} {mejor_nom:>6} {temperaturas[mejor_nom]:>9.2f} {med:>13.2f} "
          f"[{lo:>8.2f}, {hi:>8.2f}] {'Sí' if empate else 'No':>8}")
