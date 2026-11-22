import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================
# 1. MOTEUR DE SIMULATION (ex G1.py)
# ==========================================
V_KMH, SLOW = 40.0, 2.0
P_NOM, BETA, Q_PICK, P_BRK = 0.7, 0.3, 0.95, 0.03
P120A = {0: 0.02, 1: 0.15}; P120B = {0: 0.01, 1: 0.05}
PDR, PRD = 0.25, 0.25
DAYS, D0, NMSG, KMAX = 40, 5, 20, 8
SLOT_WINDOW = (6.0, 20.0); C_SLOT = 0.5; B_FACTOR = 1.5
SEG = {"S1": 155.0, "S2": 107.0, "S3": 150.0, "S4": 100.0, "S5": 130.0}
PATHS = {"Чибанга": ["S1"], "Маюмба": ["S1", "S2"], "Макоку": ["S3"], "Манджи": ["S4"], "Оконджа": ["S5"]}
VILLAGES = list(PATHS)

def make_base(R, seed=21):
    rng = np.random.default_rng(seed)
    W = np.zeros((R, DAYS), int); W[:, 0] = rng.random(R) < PDR/(PDR+PRD)
    for d in range(1, DAYS):
        u = rng.random(R); W[:, d] = np.where(W[:, d-1] == 1, u < (1-PRD), u < PDR)
    return dict(W=W, Useg={s: rng.random((R, DAYS)) for s in SEG}, UB={v: rng.random((R, DAYS)) for v in VILLAGES},
               U=rng.random((3, R, DAYS - D0, KMAX)), TM=24*rng.uniform(8, 25, (R, NMSG)))

def cut(U, L, W, p120):
    p = 1 - (1 - np.where(W == 1, p120[1], p120[0]))**(L/120.0)
    return U < p

def village_geometry(seg=None):
    seg = seg or SEG
    return {v: sum(seg[s] for s in PATHS[v]) for v in VILLAGES}

def eval_village(base, v, k, nB, seg=None, return_lat=False):
    seg = seg or SEG; W = base["W"]; R = W.shape[0]
    dA = sum(seg[s] for s in PATHS[v]); dB = B_FACTOR*dA
    m = 1 if k == 1 else 2
    cutA = np.zeros_like(W, bool)
    for s in PATHS[v]: cutA |= cut(base["Useg"][s], seg[s], W, P120A)
    cutB = cut(base["UB"][v], dB, W, P120B)
    cost = (dA/120.0)*(k - nB) + (dB/120.0)*nB + (C_SLOT*m if k > 0 else 0.0)
    if k == 0: return (np.full((R, NMSG), np.inf) if return_lat else (1.0, cost))
    times = np.array([13.0]) if m == 1 else np.linspace(*SLOT_WINDOW, m)
    st = times[np.arange(k) % m]
    slot = np.arange(k) % m; isB = np.zeros(k, bool); left, s_ = nB, 0
    while left > 0 and s_ < 10*k*m:
        c = np.where((slot == s_ % m) & (~isB))[0]
        if c.size: isB[c[0]] = True; left -= 1
        s_ += 1
    T0v = np.where(isB, dB, dA)/V_KMH; days = np.arange(D0, DAYS)
    U, TM = base["U"], base["TM"]; lat = np.full((R, NMSG), np.inf)
    for r in range(R):
        w = W[r, D0:]
        cutm = np.where(isB[None, :], cutB[r, D0:, None], cutA[r, D0:, None])
        ok = (~cutm) & (U[0, r, :, :k] <= Q_PICK) & (U[1, r, :, :k] >= P_BRK)
        tr = np.where(U[2, r, :, :k] < P_NOM*(1 - BETA*w[:, None]), T0v[None, :], T0v[None, :]*SLOW)
        dep = (24*days)[:, None] + st[None, :]; arr = dep + tr
        dep, arr = dep[ok], arr[ok]
        if dep.size == 0: continue
        o = np.argsort(dep); dep, arr = dep[o], arr[o]
        sm = np.minimum.accumulate(arr[::-1])[::-1]
        idx = np.searchsorted(dep, TM[r]); good = idx < dep.size
        lat[r, good] = sm[idx[good]] - TM[r, good]
    if return_lat: return lat
    return 1.0 - float(np.mean(lat <= 24)), cost

def table(base, with_B, seg=None):
    T = {}
    for v in VILLAGES:
        opts = []
        for k in range(0, KMAX + 1):
            for nB in (range(0, k + 1) if with_B else [0]):
                psi, c = eval_village(base, v, k, nB, seg); opts.append(((k, nB), psi, c))
        T[v] = opts
    return T

def prune(F):
    keep = []
    for i in range(len(F)):
        if not np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)): keep.append(i)
    return np.array(keep)

def exact_front(T):
    S = np.array([[0.0, 0.0, 0.0]]); allocs = [()]
    for v in VILLAGES:
        opts = T[v]; psi = np.array([o[1] for o in opts]); cs = np.array([o[2] for o in opts])
        nS = (S[:, None, 0] + psi[None, :]).ravel(); nM = np.maximum(S[:, None, 1], psi[None, :]).ravel()
        nC = (S[:, None, 2] + cs[None, :]).ravel()
        newal = [a + (opts[j][0],) for a in allocs for j in range(len(opts))]
        F = np.stack([nS, nM, nC], 1); keep = prune(F)
        S = F[keep]; allocs = [newal[i] for i in keep]
    S[:, 0] /= len(VILLAGES)
    return S, allocs

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

    # --- 1. Calculs (ex G2.py) ---
    print("Calcul des fronts exacts pour les 5 villes...")
    R = 800
    base = make_base(R)
    res = {}
    for label, wb in (("axe_unique", False), ("deux_axes", True)):
        print(f"  -> {label}")
        T = table(base, wb); F, al = exact_front(T)
        res[label] = dict(T={v: [(o[0], o[1], o[2]) for o in T[v]] for v in T}, F=F.tolist())

    # --- 2. FIGURE (ex G4.py) ---
    print("\nGénération de la Fig G...")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panneau gauche : Ψ en fonction de k pour chaque ville (axe unique)
    for v in VILLAGES:
        d = {tuple(o[0]): o[1] for o in res["axe_unique"]["T"][v]}
        ax[0].plot(range(1, 9), [d[(k, 0)] for k in range(1, 9)], marker="o", ms=3,
                   label=f"{v} ({village_geometry()[v]:.0f} км)")
   
    ax[0].set_xlabel("Количество транспортных средств в день k");
    ax[0].set_ylabel("Ψ = 1 − P(задержка ≤ 24 ч)"); ax[0].set_ylim(0, .3)
    ax[0].set_title("Единая ось, сезон дождей");
    ax[0].legend(fontsize=8)

    # Panneau droit : Compromis coût-équité
    for lab, c, name in (("axe_unique", "C0", "Единая ось"), ("deux_axes", "C3", "Две оси (гипотеза)")):
        F = np.array(res[lab]["F"]); o = np.argsort(F[:, 2]); F = F[o]
        best = np.minimum.accumulate(F[:, 1])
        ax[1].step(F[:, 2], best, where="post", c=c, label=name)
   
    ax[1].set_xlabel("Нормализованная общая стоимость");
    ax[1].set_ylabel("Максимальное Ψ среди пяти городов (справедливость)")
    ax[1].set_ylim(0, .35);
    ax[1].set_title("Компромисс стоимость–справедливость (точный фронт)");
    ax[1].legend()

    # Titre général (supprimé car vous l'avez dans votre document)
    # fig.suptitle("Fig. G – ...")
   
    fig.tight_layout()
    safe_save(fig, out_dir, "fig_G_cinq_villes.png")

    print(f"\n✓ Figure G générée dans : {out_dir}")

if __name__ == "__main__":
    main()