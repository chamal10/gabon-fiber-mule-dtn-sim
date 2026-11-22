import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================
# 1. MOTEUR DE SIMULATION (ex E1.py)
# ==========================================
D_KM, V_KMH, SLOW = 120.0, 40.0, 2.0
T0 = D_KM / V_KMH
P_NOM, BETA, Q_PICK, P_BRK = 0.7, 0.3, 0.95, 0.03
P_BLOCK = {0: 0.02, 1: 0.15}
SEASONS = {"sec": (0.03, 0.5), "pluie": (0.25, 0.25)}
DAYS, D0, KMAX, NMSG = 40, 5, 12, 20
SLOT_WINDOW = (6.0, 20.0)

def make_worlds(season, R, seed=7):
    rng = np.random.default_rng(seed)
    pdr, prd = SEASONS[season]
    W = np.zeros((R, DAYS), int)
    W[:, 0] = rng.random(R) < pdr/(pdr+prd)
    for d in range(1, DAYS):
        u = rng.random(R)
        W[:, d] = np.where(W[:, d-1] == 1, u < (1-prd), u < pdr)
    pb = np.where(W == 1, P_BLOCK[1], P_BLOCK[0])
    B = rng.random((R, DAYS)) < pb
    U = rng.random((3, R, DAYS - D0, KMAX))
    TM = 24*rng.uniform(8, 25, (R, NMSG))
    return dict(W=W, B=B, U=U, TM=TM)

def evaluate(world, k, m, return_lat=False):
    W, B, U, TM = world["W"], world["B"], world["U"], world["TM"]
    R = W.shape[0]
    times = np.array([13.0]) if m == 1 else np.linspace(*SLOT_WINDOW, m)
    st = times[np.arange(k) % m]
    days = np.arange(D0, DAYS)
    lat = np.full((R, NMSG), np.inf)
    for r in range(R):
        w, b = W[r, D0:], B[r, D0:]
        ok = (~b)[:, None] & (U[0, r, :, :k] <= Q_PICK) & (U[1, r, :, :k] >= P_BRK)
        pf = P_NOM * (1 - BETA * w[:, None])
        tr = np.where(U[2, r, :, :k] < pf, T0, T0*SLOW)
        dep = (24*days)[:, None] + st[None, :]
        arr = dep + tr
        dep, arr = dep[ok], arr[ok]
        if dep.size == 0: continue
        o = np.argsort(dep); dep, arr = dep[o], arr[o]
        sm = np.minimum.accumulate(arr[::-1])[::-1]
        idx = np.searchsorted(dep, TM[r])
        good = idx < dep.size
        lat[r, good] = sm[idx[good]] - TM[r, good]
    if return_lat: return lat
    flat = lat.ravel()
    return float(np.mean(flat <= 24)), float(np.percentile(flat, 95)), float(np.median(flat))

def pareto(points):
    front = []
    for a in points:
        dom = any((b[0] >= a[0] and b[1] <= a[1] and b[2] <= a[2]) and (b[0] > a[0] or b[1] < a[1] or b[2] < a[2]) for b in points)
        if not dom: front.append(a)
    return front

def safe_save(fig, out_dir, filename, dpi=200):
    path = os.path.join(out_dir, filename)
    test_path = os.path.join(out_dir, "_test_write.txt")
    try:
        with open(test_path, "w") as f: f.write("ok")
        os.remove(test_path)
    except Exception as e:
        print(f"  ⚠ Écriture impossible dans {out_dir} : {e}")
        out_dir = os.path.abspath(".")
        path = os.path.join(out_dir, filename)
        print(f"  → Fallback vers : {path}")
    try:
        fig.savefig(path, dpi=dpi)
        print(f"  ✓ Sauvegardé : {path}")
    except Exception as e:
        print(f"  ✗ Échec savefig : {e}")
        alt = os.path.join(out_dir, f"alt_{datetime.datetime.now().strftime('%H%M%S')}_{filename}")
        fig.savefig(alt, dpi=dpi)
        print(f"  → Sauvegardé sous : {alt}")
    plt.close(fig)

# ==========================================
# 2. ANALYSE ET GÉNÉRATION DE LA FIGURE
# ==========================================
def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(f"figures_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Dossier de sortie : {out_dir}\n")

    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": .3,
        "font.family": "DejaVu Sans"
    })

    # --- ANALYSE (ex E2.py) ---
    print("Calcul de l'espace des décisions (k, m)...")
    R = 1500
    res_pluie = {}
    # On ne calcule que la saison des pluies pour la Fig E
    world = make_worlds("pluie", R)
    for k in range(1, KMAX + 1):
        for m in range(1, min(k, 6) + 1):
            res_pluie[f"{k},{m}"] = evaluate(world, k, m)

    # --- FIGURE (ex E3.py) ---
    print("Génération de la Fig E...")
    K, M = 12, 6
    H = np.full((M, K), np.nan)
    for kk, v in res_pluie.items():
        k, m = map(int, kk.split(",")); H[m-1, k-1] = v[0]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panneau gauche : Heatmap
    im = ax[0].imshow(H, origin="lower", cmap="viridis", aspect="auto",
                      extent=[.5, K+.5, .5, M+.5], vmin=.6, vmax=.96)
    ax[0].set_xlabel("Количество транспортных средств в день k")
    ax[0].set_ylabel("Количество временных окон отправления в день m")
    ax[0].set_title("P(задержка ≤ 24 ч), сезон дождей")
    plt.colorbar(im, ax=ax[0])
   
    # Annotations sur la heatmap
    for kk, v in res_pluie.items():
        k, m = map(int, kk.split(","))
        if m == 2 and k in (2, 4, 8):
            ax[0].text(k, m, f"{v[0]:.2f}", ha="center", va="center", color="w", fontsize=7)

    # Panneau droit : Pareto
    cs = 0.5
    pts = [(v[0], v[1], int(kk.split(",")[0]) + cs*int(kk.split(",")[1]), kk) for kk, v in res_pluie.items()]
    f = sorted(pareto(pts), key=lambda x: x[2])
   
    ax[1].scatter([p[2] for p in pts], [p[0] for p in pts], c="lightgray", s=18, label="Доминируемые решения")
    ax[1].plot([p[2] for p in f], [p[0] for p in f], "o-", c="C3", label="Точный фронт Парето")
   
    for p in f:
        if p[3] in ("1,1", "2,2", "4,2", "3,3"):
            ax[1].annotate(f"({p[3]})", (p[2], p[0]), textcoords="offset points", xytext=(6, -10), fontsize=8)
           
    ax[1].set_xlabel("Нормализованная стоимость k + 0,5·m")
    ax[1].set_ylabel("P(задержка ≤ 24 ч)")
    ax[1].set_title("Компромисс надежность–стоимость (задержка P95 — 3-я цель)")
    ax[1].legend(loc="lower right", fontsize=8)
    ax[1].grid(alpha=.3)

    # Titre général (supprimé car vous l'avez dans votre document)
    # fig.suptitle("Fig. E – ...")
   
    fig.tight_layout()
    safe_save(fig, out_dir, "fig_E_pareto_k_m.png")

    print(f"\n✓ Figure E générée dans : {out_dir}")

if __name__ == "__main__":
    main()