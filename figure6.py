import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================
# 1. MOTEUR DE SIMULATION (ex sim2.py)
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

    # --- 1. CALCULS (ex H1.py) ---
    print("Calcul des courbes de seuil...")
    rng = np.random.default_rng(7); R = 1500
    pdr, prd = SEASONS["pluie"]
    W = np.zeros((R, DAYS), int); W[:, 0] = rng.random(R) < pdr/(pdr+prd)
    for d in range(1, DAYS):
        u = rng.random(R); W[:, d] = np.where(W[:, d-1] == 1, u < (1-prd), u < pdr)
    UB = rng.random((R, DAYS)); U = rng.random((3, R, DAYS - D0, KMAX)); TM = 24*rng.uniform(8, 25, (R, NMSG))
    qs = np.round(np.arange(0.02, 0.62, 0.02), 2); ks = [2, 3, 4, 5, 6, 8]
    P = {}
    for k in ks:
        m = 1 if k == 1 else 2; rows = []
        for q in qs:
            pb = np.where(W == 1, q, q*0.02/0.15)
            world = dict(W=W, B=UB < pb, U=U, TM=TM)
            lat = evaluate(world, k, m, return_lat=True); rows.append((lat <= 24).mean(axis=1))
        P[k] = np.array(rows)

    def crossing(curve, level):
        below = np.where(curve < level)[0]
        if len(below) == 0: return np.nan
        j = below[0]
        if j == 0: return 0.0
        x0, x1, y0, y1 = qs[j-1], qs[j], curve[j-1], curve[j]
        return float(x0 + (level - y0)*(x1 - x0)/(y1 - y0))

    rb = np.random.default_rng(3); B = 500; res = {}
    for level in (0.90, 0.95):
        for k in ks:
            pt = crossing(P[k].mean(axis=1), level)
            bs = [crossing(P[k][:, rb.integers(0, R, R)].mean(axis=1), level) for _ in range(B)]
            bs = np.array(bs, float)
            res[f"{level}|{k}"] = dict(q=pt,
                                       lo=float(np.nanpercentile(bs, 2.5)) if np.isfinite(bs).any() else None,
                                       hi=float(np.nanpercentile(bs, 97.5)) if np.isfinite(bs).any() else None)

    # --- 2. FIGURE (ex H2.py) ---
    print("Génération de la Fig H...")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panneau gauche
    for k in ks:
        ax[0].plot(qs, P[k].mean(axis=1), label=f"k = {k}")
    for lv, st in ((0.90, "--"), (0.95, ":")):
        ax[0].axhline(lv, c="k", ls=st, lw=1)
    ax[0].axvline(0.15, c="gray", lw=.8);
    ax[0].text(0.152, 0.76, "базовая гипотеза\nq = 0,15", fontsize=8, color="gray")
    ax[0].set_xlabel("Вероятность перекрытия q (участок 120 км, день дождя)");
    ax[0].set_ylabel("P(задержка ≤ 24 ч)")
    ax[0].set_title("Доставка в зависимости от частоты перекрытий");
    ax[0].legend(fontsize=8)

    # Panneau droit
    for lv, c, lab in ((0.90, "C0", "≥ 90 %"), (0.95, "C3", "≥ 95 %")):
        q = [res[f"{lv}|{k}"]["q"] for k in ks];
        lo = [res[f"{lv}|{k}"]["lo"] for k in ks];
        hi = [res[f"{lv}|{k}"]["hi"] for k in ks]
        ax[1].errorbar(ks, q, yerr=[np.array(q) - np.array(lo), np.array(hi) - np.array(q)],
                       marker="o", capsize=3, c=c, label=f"Доставка {lab}")
    ax[1].set_xlabel("Количество транспортных средств в день k");
    ax[1].set_ylabel("Критический порог q*"); ax[1].set_ylim(0, .35)
    ax[1].set_title("Максимально допустимая частота перекрытий");
    ax[1].legend(fontsize=8)

    # Titre général (supprimé car vous l'avez dans votre document)
    # fig.suptitle("Fig. H – ...")
   
    fig.tight_layout()
    safe_save(fig, out_dir, "fig_H_seuil_critique.png")

    print(f"\n✓ Figure H générée dans : {out_dir}")

if __name__ == "__main__":
    main()