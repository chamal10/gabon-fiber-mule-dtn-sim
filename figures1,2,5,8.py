import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================
# 1. MOTEUR DE SIMULATION
# ==========================================
rng = np.random.default_rng(42)

D_KM, V_KMH = 120.0, 40.0
T0 = D_KM / V_KMH
SLOW = 2.0
P_NOM, BETA = 0.7, 0.3
Q_PICK, P_BRK = 0.95, 0.03
P_BLOCK = {0: 0.02, 1: 0.15}
DAYS, R, NMSG = 40, 1500, 20

# Clés internes (invisibles sur les figures)
SEASONS = {"saison seche": (0.03, 0.5), "saison des pluies": (0.25, 0.25)}
# Traduction pour l'affichage
SEASONS_RU = {"saison seche": "Сухой сезон", "saison des pluies": "Сезон дождей"}

def weather(pdr, prd):
    w = np.zeros(DAYS, int); w[0] = rng.random() < pdr/(pdr+prd)
    for d in range(1, DAYS):
        w[d] = (rng.random() < (1-prd)) if w[d-1] else (rng.random() < pdr)
    return w

def run(k, season, correlated=True):
    pdr, prd = SEASONS[season]
    frac_rain = pdr/(pdr+prd)
    lat = []
    for _ in range(R):
        w = weather(pdr, prd)
        blocked_day = np.array([rng.random() < P_BLOCK[x] for x in w])
        dep, arr = [], []
        for d in range(5, DAYS):
            n = rng.poisson(k)
            if n == 0: continue
            t = 24*d + rng.uniform(6, 20, n)
            for ti in t:
                if correlated:
                    wd, blk = w[d], blocked_day[d]
                else:
                    wd = rng.random() < frac_rain
                    blk = rng.random() < P_BLOCK[int(wd)]
                if blk or rng.random() > Q_PICK or rng.random() < P_BRK: continue
                p_fast = P_NOM*(1-BETA*wd)
                tr = T0 if rng.random() < p_fast else T0*SLOW
                dep.append(ti); arr.append(ti+tr)
        if not dep:
            lat += [np.inf]*NMSG; continue
        o = np.argsort(dep); dep = np.array(dep)[o]; arr = np.array(arr)[o]
        sufmin = np.minimum.accumulate(arr[::-1])[::-1]
        t0 = 24*rng.uniform(8, 25, NMSG) + rng.uniform(0, 24, NMSG)*0
        idx = np.searchsorted(dep, t0)
        l = np.where(idx < len(dep), sufmin[np.minimum(idx, len(dep)-1)] - t0, np.inf)
        lat += list(l)
    return np.array(lat)

def stats(l):
    out = {tau: float(np.mean(l <= tau)) for tau in (6, 12, 24, 48)}
    out["med"] = float(np.median(l)); out["p95"] = float(np.percentile(l, 95))
    return out

def safe_save(fig, out_dir, filename, dpi=200):
    """Sauvegarde robuste avec test d'écriture et fallback."""
    path = os.path.join(out_dir, filename)
    # Test d'écriture
    test_path = os.path.join(out_dir, "_test_write.txt")
    try:
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception as e:
        print(f"  ⚠ Écriture impossible dans {out_dir} : {e}")
        # Fallback vers le dossier utilisateur
        out_dir = os.path.abspath(".")
        path = os.path.join(out_dir, filename)
        print(f"  → Fallback vers : {path}")
    try:
        fig.savefig(path, dpi=dpi)
        print(f"  ✓ Sauvegardé : {path}")
    except Exception as e:
        print(f"  ✗ Échec savefig : {e}")
        # Dernier recours : nom unique
        alt = os.path.join(out_dir, f"alt_{datetime.datetime.now().strftime('%H%M%S')}_{filename}")
        fig.savefig(alt, dpi=dpi)
        print(f"  → Sauvegardé sous : {alt}")
    plt.close(fig)

# ==========================================
# 2. ANALYSES ET GÉNÉRATION DES FIGURES
# ==========================================
def main():
    # Dossier horodaté : aucun verrou possible
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

    # ---------------------------------------------------------
    # FIG A : CDF de latence
    # ---------------------------------------------------------
    print("Génération Fig A...")
    R = 1000
    cdf = {}
    for season in SEASONS:
        for k in (1, 3, 5, 8):
            l = run(k, season, True)
            taus = list(range(0, 97, 2))
            cdf[f"{season}|{k}"] = [float(np.mean(l <= t)) for t in taus]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, s in zip(ax, ["saison seche", "saison des pluies"]):
        for k in (1, 3, 5, 8):
            a.plot(taus, cdf[f"{s}|{k}"], label=f"k = {k}")
        a.axvline(24, ls=":", c="gray")
        a.set_xlabel("Задержка τ (часы)"); a.set_ylim(0, 1.02)
        a.set_title(SEASONS_RU[s])
    ax[0].set_ylabel("P(задержка ≤ τ)"); ax[0].legend(loc="lower right")
    fig.tight_layout(); safe_save(fig, out_dir, "fig_A_cdf_latence.png")

    # ---------------------------------------------------------
    # FIG B : Plateau en k
    # ---------------------------------------------------------
    print("Génération Fig B...")
    R = 500
    ks = list(range(1, 16))
    curves = {}
    for season in SEASONS:
        for corr in (True, False):
            P24, P95 = [], []
            for k in ks:
                st = stats(run(k, season, corr)); P24.append(st[24]); P95.append(st["p95"])
            curves[f"{season}|{'corr' if corr else 'indep'}"] = {"P24": P24, "P95": P95}

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for s, st in (("saison seche", "-"), ("saison des pluies", "--")):
        lab = "сухой сезон" if "seche" in s else "сезон дождей"
        ax[0].plot(ks, curves[f"{s}|corr"]["P24"], st, c="C3", label=f"коррелированная, {lab}")
        ax[0].plot(ks, curves[f"{s}|indep"]["P24"], st, c="C0", label=f"независимая, {lab}")
        ax[1].plot(ks, curves[f"{s}|corr"]["P95"], st, c="C3")
        ax[1].plot(ks, curves[f"{s}|indep"]["P95"], st, c="C0")
    ax[0].set_ylabel("P(задержка ≤ 24 ч)"); ax[1].set_ylabel("Задержка 95-го процентиля (ч)")
    for a in ax: a.set_xlabel("Количество транспортных средств в день k")
    ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); safe_save(fig, out_dir, "fig_B_plateau_k.png")

    # ---------------------------------------------------------
    # FIG C : Tornado
    # ---------------------------------------------------------
    print("Génération Fig C...")
    R = 600
    def base_run(): return stats(run(5, "saison des pluies", True))[24]
    base = base_run()
    def setD(v):
        global D_KM, T0; D_KM = v; T0 = D_KM / V_KMH
    def setV(v):
        global V_KMH, T0; V_KMH = v; T0 = D_KM / V_KMH
    def setattr_(n): return lambda v: globals().update({n: v})
    def setBr(v): P_BLOCK[1] = v

    params = [
     ("Расстояние до подключенного узла (км)", setD, 120, 60, 200),
     ("Номинальная скорость (км/ч)", setV, 40, 30, 50),
     ("Коэффициент замедления", setattr_("SLOW"), 2.0, 1.5, 3.0),
     ("p_номинальное", setattr_("P_NOM"), 0.7, 0.5, 0.9),
     ("Вероятность подхвата Q", setattr_("Q_PICK"), 0.95, 0.85, 1.0),
     ("Поломка транспортного средства", setattr_("P_BRK"), 0.03, 0.0, 0.10),
     ("Дорога перекрыта во время дождя", setBr, 0.15, 0.05, 0.30),
    ]
    tor = []
    for name, setter, nom, lo, hi in params:
        vals = []
        for v in (lo, hi):
            setter(v); vals.append(base_run())
        setter(nom)
        tor.append({"name": name, "lo": lo, "hi": hi, "P_lo": vals[0], "P_hi": vals[1]})

    t = sorted(tor, key=lambda p: abs(p["P_hi"] - p["P_lo"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, p in enumerate(t):
        ax.barh(i, p["P_lo"] - base, left=base, color="C2", alpha=.8)
        ax.barh(i, p["P_hi"] - base, left=base, color="C1", alpha=.8)
        ax.text(min(p["P_lo"], p["P_hi"]) - .002, i, f'{p["lo"]}', ha="right", va="center", fontsize=8)
        ax.text(max(p["P_lo"], p["P_hi"]) + .002, i, f'{p["hi"]}', ha="left", va="center", fontsize=8)
    ax.axvline(base, c="k", lw=1); ax.set_yticks(range(len(t))); ax.set_yticklabels([p["name"] for p in t])
    ax.set_xlabel("P(задержка ≤ 24 ч), k = 5, сезон дождей"); ax.set_xlim(.82, .97)
    fig.tight_layout(); safe_save(fig, out_dir, "fig_C_tornado.png")

    # ---------------------------------------------------------
    # FIG D : Panne de fibre
    # ---------------------------------------------------------
    print("Génération Fig D...")
    R = 1000
    L = {"sec": run(5, "saison seche", True), "pluie": run(5, "saison des pluies", True)}
    durs = np.array([2, 4, 8, 12, 18, 24, 36, 48, 72, 120, 168])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(durs, durs, "r", lw=2, label="Только оптоволокно: ожидание восстановления")
    for s, st, c in (("sec", "-", "b"), ("pluie", "--", "C9")):
        med = [np.median(np.minimum(d, L[s])) for d in durs]
        p95 = [np.percentile(np.minimum(d, L[s]), 95) for d in durs]
        lab_ru = "сухо" if s == "sec" else "дождь"
        ax[0].plot(durs, med, st, c=c, lw=2, label=f"Оптоволокно + мулы k = 5 ({lab_ru})")
        ax[1].plot(durs, durs, "r", lw=2)
        ax[1].plot(durs, p95, st, c=c, lw=2)
    for a, t in zip(ax, ("Медианная задержка (ч)", "Задержка 95-го процентиля (ч)")):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("Длительность отказа оптоволокна (ч)"); a.set_ylabel(t)
        for m in (18, 36): a.axvline(m, c="gray", ls=":", lw=1)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].text(18, 2.3, "MTTR (сухо)", fontsize=7, rotation=90, ha="right")
    ax[0].text(36, 2.3, "MTTR (дождь)", fontsize=7, rotation=90, ha="right")
    fig.tight_layout(); safe_save(fig, out_dir, "fig_D_panne_fibre.png")

    print(f"\n✓ Toutes les figures sont dans : {out_dir}")

if __name__ == "__main__":
    main()