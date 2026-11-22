import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================
# 1. MOTEUR DE SIMULATION (ex F1.py)
# ==========================================
V_KMH, SLOW = 40.0, 2.0
P_NOM, BETA, Q_PICK, P_BRK = 0.7, 0.3, 0.95, 0.03
ROUTES = {
 "A": dict(d=120.0, block={0: 0.02, 1: 0.15}),
 "B": dict(d=200.0, block={0: 0.01, 1: 0.05}),
}
SEASONS = {"sec": (0.03, 0.5), "pluie": (0.25, 0.25)}
DAYS, D0, KMAX, MMAX, NMSG = 40, 5, 12, 4, 20
SLOT_WINDOW = (6.0, 20.0); C_SLOT = 0.5

def make_worlds(season, R, seed=11):
    rng = np.random.default_rng(seed); pdr, prd = SEASONS[season]
    W = np.zeros((R, DAYS), int); W[:, 0] = rng.random(R) < pdr/(pdr+prd)
    for d in range(1, DAYS):
        u = rng.random(R); W[:, d] = np.where(W[:, d-1] == 1, u < (1-prd), u < pdr)
    blk = {}
    for r, p in ROUTES.items():
        blk[r] = rng.random((R, DAYS)) < np.where(W == 1, p["block"][1], p["block"][0])
    U = rng.random((3, R, DAYS - D0, KMAX)); TM = 24*rng.uniform(8, 25, (R, NMSG))
    return dict(W=W, blk=blk, U=U, TM=TM)

def assign_routes(k, m, nB):
    slot = np.arange(k) % m; isB = np.zeros(k, bool); left = nB; s = 0
    while left > 0:
        cand = np.where((slot == s % m) & (~isB))[0]
        if cand.size: isB[cand[0]] = True; left -= 1
        s += 1
        if s > 10*k*m: break
    return isB

def cost(k, m, nB):
    return (k - nB)*1.0 + nB*ROUTES["B"]["d"]/ROUTES["A"]["d"] + C_SLOT*m

def evaluate(world, k, m, nB, return_lat=False):
    W, blk, U, TM = world["W"], world["blk"], world["U"], world["TM"]
    R = W.shape[0]
    times = np.array([13.0]) if m == 1 else np.linspace(*SLOT_WINDOW, m)
    st = times[np.arange(k) % m]; isB = assign_routes(k, m, nB)
    T0v = np.where(isB, ROUTES["B"]["d"], ROUTES["A"]["d"]) / V_KMH
    days = np.arange(D0, DAYS); lat = np.full((R, NMSG), np.inf)
    for r in range(R):
        w = W[r, D0:]
        cut = np.where(isB[None, :], blk["B"][r, D0:, None], blk["A"][r, D0:, None])
        ok = (~cut) & (U[0, r, :, :k] <= Q_PICK) & (U[1, r, :, :k] >= P_BRK)
        pf = P_NOM*(1 - BETA*w[:, None])
        tr = np.where(U[2, r, :, :k] < pf, T0v[None, :], T0v[None, :]*SLOW)
        dep = (24*days)[:, None] + st[None, :]; arr = dep + tr
        dep, arr = dep[ok], arr[ok]
        if dep.size == 0: continue
        o = np.argsort(dep); dep, arr = dep[o], arr[o]
        sm = np.minimum.accumulate(arr[::-1])[::-1]
        idx = np.searchsorted(dep, TM[r]); good = idx < dep.size
        lat[r, good] = sm[idx[good]] - TM[r, good]
    if return_lat: return lat
    f = lat.ravel()
    return float(np.mean(f <= 24)), float(np.percentile(f, 95)), cost(k, m, nB)

def space():
    return [(k, m, nB) for k in range(1, KMAX+1) for m in range(1, min(k, MMAX)+1) for nB in range(0, k+1)]

def pareto_idx(F):
    F = np.asarray(F); keep = []
    for i in range(len(F)):
        if not np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)): keep.append(i)
    return keep

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

    # --- 1. Espace des décisions (ex F2.py) ---
    print("Calcul de l'espace des décisions (k, m, nB)...")
    R = 1000
    world_pluie = make_worlds("pluie", R)
    tab = {}
    for (k, m, nB) in space():
        tab[f"{k},{m},{nB}"] = evaluate(world_pluie, k, m, nB)
   
    keys = list(tab); F = np.array([[-tab[c][0], tab[c][1], tab[c][2]] for c in keys])
    fr = set(keys[i] for i in pareto_idx(F))
    sub = [c for c in keys if c.endswith(",0")]; Fs = np.array([[-tab[c][0], tab[c][1], tab[c][2]] for c in sub])
    fr0 = set(sub[i] for i in pareto_idx(Fs))

    # --- 2. Robustesse : corrélation des coupures entre axes (ex F3.py) ---
    print("Calcul de la robustesse (corrélation ρ)...")
    R_rob = 1000
    world_rob = make_worlds("pluie", R_rob)
    rho_res = {}
    base_B = world_rob["blk"]["B"].copy()
    rr = np.random.default_rng(5).random(world_rob["blk"]["A"].shape)
   
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        world_rob["blk"]["B"] = base_B | (world_rob["blk"]["A"] & (rr < rho))
        rho_res[rho] = {c: evaluate(world_rob, *map(int, c.split(",")))[:2] for c in ("4,2,0", "4,2,2", "6,3,3")}
        print(f"  ρ = {rho:.2f}")

    # --- 3. FIGURE (ex F4.py) ---
    print("\nGénération de la Fig F...")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panneau gauche : Front de Pareto
    ax[0].scatter([tab[c][2] for c in keys], [tab[c][0] for c in keys],
                  c="lightgray", s=10, label="Пространство решений (344 точки)")
    for s, col, lab in ((fr0, "C0", "Фронт, только ось A (r = 0)"),
                        (fr, "C3", "Фронт, оси A + B (r свободно)")):
        p = sorted((tab[c][2], tab[c][0]) for c in s);
        ax[0].plot(*zip(*p), "o-", c=col, ms=3, label=lab)
   
    ax[0].annotate("(4, 2, 2)", (tab["4,2,2"][2], tab["4,2,2"][0]),
                   xytext=(10, -22), textcoords="offset points", fontsize=8,
                   arrowprops=dict(arrowstyle="-", lw=.5))
    ax[0].set_ylim(.6, 1.01);
    ax[0].set_xlabel("Нормализованная стоимость");
    ax[0].set_ylabel("P(задержка ≤ 24 ч)")
    ax[0].set_title("Сезон дождей");
    ax[0].legend(fontsize=8, loc="lower right")

    # Panneau droit : Robustesse
    rhos = sorted(rho_res.keys())
    for c, lab, st in (("4,2,0", "(4,2,0) : только ось A", "--"),
                       ("4,2,2", "(4,2,2) : две оси", "-"),
                       ("6,3,3", "(6,3,3) : две оси", "-")):
        ax[1].plot(rhos, [rho_res[r][c][0] for r in rhos], st, marker="o", label=lab)
   
    ax[1].set_xlabel("Корреляция ρ перекрытий между осями A и B");
    ax[1].set_ylabel("P(задержка ≤ 24 ч)")
    ax[1].set_title("Устойчивость выигрыша от диверсификации");
    ax[1].legend(fontsize=8)

    # Titre général (supprimé car vous l'avez dans votre document)
    # fig.suptitle("Fig. F – ...")
   
    fig.tight_layout()
    safe_save(fig, out_dir, "fig_F_route_r.png")

    print(f"\n✓ Figure F générée dans : {out_dir}")

if __name__ == "__main__":
    main()