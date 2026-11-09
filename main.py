# -*- coding: utf-8 -*-
"""
Лабораторная работа №2 по дисциплине «Методы оптимизации. Линейное
программирование». Вариант 1.

Задача: транспортная задача «склады → магазины», минимум стоимости
перевозок (4 склада, 6 магазинов).

Реализовано:
    * загрузка данных из CSV;
    * LP-модель транспортной задачи с ограничениями на маршруты;
    * решение HiGHS, независимая проверка балансов и границ;
    * визуализация сети и матрицы перевозок;
    * сценарии: дефицит склада, отказ маршрута, стресс с дефицитом спроса;
    * модель с потерянными продажами (штраф за неудовлетворённый спрос);
    * теневые цены складов и насыщенных маршрутов;
    * жадная эвристика и MILP-версия (дополнительное задание).

Запуск:
    python generate_data.py
    python main.py
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"
TOL = 1e-7


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def load_data():
    warehouses, stores, costs, caps = [], [], [], []
    with open(DATA / "warehouses.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            warehouses.append({"id": int(row["warehouse_id"]), "name": row["warehouse"],
                               "supply": float(row["supply"])})
    with open(DATA / "stores.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stores.append({"id": int(row["store_id"]), "name": row["store"],
                           "demand": float(row["demand"])})
    with open(DATA / "routes.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            costs.append([float(row["cost"])])
            caps.append([float(row["capacity"])])
    costs = np.array(costs).reshape(len(warehouses), len(stores))
    caps = np.array(caps).reshape(len(warehouses), len(stores))
    with open(DATA / "params.json", encoding="utf-8") as f:
        params = json.load(f)
    return warehouses, stores, costs, caps, params


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
def solve_transport(costs, supply, demand, caps, penalty=None, disp=False):
    """
    min sum c_ij x_ij.
    Если penalty is None — чистая транспортная модель.
    Иначе добавляются переменные недопоставки u_j с штрафом penalty.
    """
    n, m = costs.shape
    N = n * m
    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(N)
        row[i * m:(i + 1) * m] = 1.0
        A_ub.append(row)
        b_ub.append(supply[i])
    for i in range(n):
        for j in range(m):
            row = np.zeros(N)
            row[i * m + j] = 1.0
            A_ub.append(row)
            b_ub.append(caps[i, j])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    A_eq = []
    for j in range(m):
        row = np.zeros(N)
        row[j::m] = 1.0
        A_eq.append(row)
    A_eq = np.array(A_eq)
    b_eq = demand.copy()

    if penalty is None:
        c_obj = costs.flatten()
        bounds = [(0.0, None)] * N
    else:
        A_ub = np.hstack([A_ub, np.zeros((A_ub.shape[0], m))])
        A_eq = np.hstack([A_eq, np.eye(m)])
        c_obj = np.concatenate([costs.flatten(), np.full(m, float(penalty))])
        bounds = [(0.0, None)] * (N + m)

    args = dict(A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs")
    if not disp:
        return linprog(c_obj, **args), ""
    old_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    try:
        res = linprog(c_obj, options={"disp": True}, **args)
    finally:
        sys.stdout.flush()
        os.dup2(old_fd, 1)
        os.close(old_fd)
        tmp.seek(0)
        log = tmp.read().decode("utf-8", errors="replace")
        tmp.close()
    return res, log


def verify_transport(res, costs, supply, demand, caps, penalty=None, tol=1e-7):
    n, m = costs.shape
    N = n * m
    x = res.x[:N].reshape(n, m)
    unmet = res.x[N:] if penalty is not None else np.zeros(m)
    supply_used = x.sum(axis=1)
    delivered = x.sum(axis=0)
    cap_violation = np.maximum(0.0, x - caps)
    supply_violation = np.maximum(0.0, supply_used - supply)
    demand_violation = np.abs(delivered + unmet - demand)
    neg_violation = np.maximum(0.0, -x).max()
    transport_cost = float(np.sum(x * costs))
    return {
        "x": x, "unmet": unmet, "supply_used": supply_used,
        "delivered": delivered,
        "max_violation": float(max(cap_violation.max(), supply_violation.max(),
                                   demand_violation.max(), neg_violation)),
        "transport_cost": transport_cost,
        "penalty_cost": float(penalty * unmet.sum()) if penalty is not None else 0.0,
        "objective": float(np.sum(x * costs) + (penalty * unmet.sum()
                                                if penalty is not None else 0.0)),
        "feasible": bool(max(cap_violation.max(), supply_violation.max(),
                             demand_violation.max(), neg_violation) <= tol),
    }


def shadow_prices(res, costs, supply, demand, caps):
    n, m = costs.shape
    marg_ub = np.asarray(res.ineqlin.marginals)
    # Для min c^T x и ограничения <=: ценность дополнительной единицы = -marginal.
    supply_sp = [-float(marg_ub[i]) for i in range(n)]
    route_sp = [-float(marg_ub[n + i * m + j]) for i in range(n) for j in range(m)]
    route_sp = np.array(route_sp).reshape(n, m)

    fd_supply = []
    for i in range(n):
        s2 = supply.copy()
        s2[i] += 1.0
        r, _ = solve_transport(costs, s2, demand, caps)
        fd_supply.append(float(res.fun - r.fun) if r.success else np.nan)

    fd_route = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            caps2 = caps.copy()
            caps2[i, j] += 1.0
            r, _ = solve_transport(costs, supply, demand, caps2)
            fd_route[i, j] = float(res.fun - r.fun) if r.success else np.nan
    return {"supply": supply_sp, "supply_fd": fd_supply,
            "route": route_sp, "route_fd": fd_route}


def greedy_transport(costs, supply, demand, caps):
    n, m = costs.shape
    s = supply.copy()
    d = demand.copy()
    x = np.zeros((n, m))
    routes = [(costs[i, j], i, j) for i in range(n) for j in range(m)]
    routes.sort()
    for _, i, j in routes:
        amount = min(s[i], d[j], caps[i, j])
        if amount <= 1e-9:
            continue
        x[i, j] += amount
        s[i] -= amount
        d[j] -= amount
    return {"x": x, "cost": float(np.sum(x * costs)),
            "unmet": float(d.sum()), "feasible": bool(d.sum() <= 1e-7)}


def solve_milp(costs, supply, demand, caps):
    n, m = costs.shape
    N = n * m
    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(N)
        row[i * m:(i + 1) * m] = 1.0
        A_ub.append(row)
        b_ub.append(supply[i])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = []
    for j in range(m):
        row = np.zeros(N)
        row[j::m] = 1.0
        A_eq.append(row)
    A_eq = np.array(A_eq)
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub),
                   LinearConstraint(A_eq, demand, demand)]
    res = milp(costs.flatten(), constraints=constraints,
               integrality=np.ones(N), bounds=Bounds(np.zeros(N), caps.flatten()))
    return res


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def draw_network(warehouses, stores, x, caps, path, title):
    G = nx.DiGraph()
    for w in warehouses:
        G.add_node(("w", w["id"]))
    for s in stores:
        G.add_node(("s", s["id"]))
    pos = {}
    for i, w in enumerate(warehouses):
        pos[("w", w["id"])] = (-2.0, 2.4 - i * 1.4)
    for j, s in enumerate(stores):
        pos[("s", s["id"])] = (2.2, 2.8 - j * 1.05)

    fig, ax = plt.subplots(figsize=(15, 8.5))
    nx.draw_networkx_nodes(G, pos, nodelist=[("w", w["id"]) for w in warehouses],
                           node_size=2200, node_color="#AED6F1",
                           edgecolors="#2C3E50", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[("s", s["id"]) for s in stores],
                           node_size=2000, node_color="#A9DFBF",
                           edgecolors="#2C3E50", ax=ax)
    labels = {("w", w["id"]): w["name"] for w in warehouses}
    labels.update({("s", s["id"]): s["name"] for s in stores})
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    n, m = x.shape
    max_cost = float(caps.max()) if caps.size else 1.0
    for i in range(n):
        for j in range(m):
            if x[i, j] <= 1e-6:
                continue
            util = x[i, j] / caps[i, j] if caps[i, j] > 0 else 0
            color = plt.cm.RdYlGn_r(min(1.0, util))
            width = 0.8 + 4.5 * util
            nx.draw_networkx_edges(G, pos, edgelist=[(("w", i + 1), ("s", j + 1))],
                                   ax=ax, edge_color=[color], width=width,
                                   arrows=True, arrowsize=14, alpha=0.85)
            x1, y1 = pos[("w", i + 1)]
            x2, y2 = pos[("s", j + 1)]
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06,
                    f"{x[i, j]:.0f}/{caps[i, j]:.0f}", fontsize=7.5,
                    ha="center", va="bottom", color="#1B4F72",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_matrix(costs, x, caps, store_names, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    im0 = axes[0].imshow(costs, cmap="YlOrRd", aspect="auto")
    axes[0].set_title("Стоимости перевозок")
    for i in range(costs.shape[0]):
        for j in range(costs.shape[1]):
            axes[0].text(j, i, f"{costs[i, j]:.1f}", ha="center", va="center", fontsize=8)
    axes[0].set_xticks(range(len(store_names)))
    axes[0].set_xticklabels(store_names, rotation=30, ha="right", fontsize=8)
    axes[0].set_yticks(range(costs.shape[0]))
    axes[0].set_yticklabels([f"W{i + 1}" for i in range(costs.shape[0])])
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(x, cmap="Blues", aspect="auto")
    axes[1].set_title("Потоки x_ij (в скобках — capacity)")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            axes[1].text(j, i, f"{x[i, j]:.0f}\n({caps[i, j]:.0f})",
                         ha="center", va="center", fontsize=7.5)
    axes[1].set_xticks(range(len(store_names)))
    axes[1].set_xticklabels(store_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_yticks(range(x.shape[0]))
    axes[1].set_yticklabels([f"W{i + 1}" for i in range(x.shape[0])])
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_balance(warehouses, stores, supply_used, delivered, demand, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    wnames = [w["name"] for w in warehouses]
    snames = [s["name"] for s in stores]
    axes[0].bar(wnames, [w["supply"] for w in warehouses], color="#AED6F1",
                label="доступно")
    axes[0].bar(wnames, supply_used, color="#2F5D8A", alpha=0.8, label="использовано")
    axes[0].set_title("Использование складов")
    axes[0].set_ylabel("Ед. груза")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(snames, demand, color="#A9DFBF", label="спрос")
    axes[1].bar(snames, delivered, color="#27AE60", alpha=0.8, label="доставлено")
    axes[1].set_title("Удовлетворение спроса")
    axes[1].set_ylabel("Ед. груза")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_scenarios(records, path):
    keys = ["base", "deficit", "route_failure", "stress"]
    titles = ["базовый", "дефицит W2", "отказ маршрута", "стресс"]
    costs = [records[k]["cost"] if records[k]["cost"] is not None else 0 for k in keys]
    unmet = [records[k]["unmet"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(titles, costs, color=["#2F5D8A", "#F39C12", "#C0392B", "#8E44AD"],
                       alpha=0.85)
    top = max([v for v in costs if v > 0] or [1])
    for bar, k in zip(bars, keys):
        rec = records[k]
        if rec["cost"] is None:
            bar.set_height(0.04 * top)
            axes[0].text(bar.get_x() + bar.get_width() / 2, 0.05 * top, "нет",
                         rotation=90, ha="center", va="bottom", color="#C0392B",
                         fontsize=9)
        else:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                         f"{rec['cost']:.0f}", ha="center", va="center",
                         color="white", fontsize=9)
    axes[0].set_ylabel("Стоимость, у.е.")
    axes[0].set_title("Стоимость перевозок по сценариям")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(titles, unmet, color=["#2F5D8A", "#F39C12", "#C0392B", "#8E44AD"],
                alpha=0.85)
    for i, v in enumerate(unmet):
        axes[1].text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    axes[1].set_ylabel("Недопоставка, ед.")
    axes[1].set_title("Неудовлетворённый спрос")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_greedy(lp_cost, greedy, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    vals = [lp_cost, greedy["cost"]]
    bars = ax.bar(["LP-оптимум", "жадная эвристика"], vals,
                  color=["#2F5D8A", "#F39C12"], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 15, f"{v:.0f}",
                ha="center", fontsize=10)
    ax.set_ylabel("Стоимость, у.е.")
    ax.set_title("LP против жадной эвристики")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_dashboard(warehouses, stores, x, caps, records, output, path):
    from matplotlib.gridspec import GridSpec
    import matplotlib.image as mpimg

    tmp = OUT / "_net_tmp.png"
    draw_network(warehouses, stores, x, caps, tmp, "Потоки в оптимальном плане")
    img = mpimg.imread(tmp)

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.25)
    fig.suptitle("ЛР №2 · Транспортная задача «склады → магазины» (вариант 1)",
                 fontsize=16, fontweight="bold", color="#12263A", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Оптимальные перевозки")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    lines = ["Сводка", "",
             f"Стоимость: {output['cost']:.2f} у.е.",
             f"Складов: {len(warehouses)}, магазинов: {len(stores)}", "",
             "Насыщенные маршруты:"]
    if output["bottlenecks"]:
        for b in output["bottlenecks"]:
            lines.append(f"  {b['w']} → {b['s']}: {b['flow']:.0f}/{b['cap']:.0f}, "
                         f"цена {b['shadow']:.2f}")
    else:
        lines.append("  нет")
    ax2.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EAF2FB", edgecolor="#2F5D8A"))

    ax3 = fig.add_subplot(gs[1, 0])
    wnames = [w["name"] for w in warehouses]
    ax3.bar(wnames, [w["supply"] for w in warehouses], color="#AED6F1",
            label="доступно")
    ax3.bar(wnames, output["supply_used"], color="#2F5D8A", alpha=0.85,
            label="использовано")
    ax3.set_title("Использование складов")
    ax3.tick_params(axis="x", rotation=15)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 1])
    keys = ["base", "deficit", "route_failure", "stress"]
    vals = [records[k]["cost"] if records[k]["cost"] is not None else 0 for k in keys]
    bars = ax4.bar(["base", "def", "fail", "stress"], vals,
                   color=["#2F5D8A", "#F39C12", "#C0392B", "#8E44AD"], alpha=0.85)
    top = max(vals + [1])
    for bar, k in zip(bars, keys):
        rec = records[k]
        label = f"{rec['cost']:.0f}" if rec["cost"] is not None else "нет"
        if rec["unmet"] > 0:
            label += f"\n−{rec['unmet']:.0f}"
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * top,
                 label, ha="center", fontsize=8.5)
    ax4.set_title("Сценарии (ниже — недопоставка)")
    ax4.grid(alpha=0.3, axis="y")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(["LP", "greedy"], [output["cost"], output["greedy_cost"]],
            color=["#2F5D8A", "#F39C12"], alpha=0.85)
    ax5.set_title("LP против жадной эвристики")
    ax5.grid(alpha=0.3, axis="y")

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def save_verification(path: str, warehouses, stores, ver, solver_obj, caps):
    rows = []
    for i, wh in enumerate(warehouses):
        rows.append(["supply", wh["name"],
                     round(float(ver["supply_used"][i]), 6),
                     round(float(wh["supply"]), 6),
                     round(float(max(0.0, ver["supply_used"][i] - wh["supply"])), 9),
                     "OK" if ver["supply_used"][i] <= wh["supply"] + TOL else "FAIL"])
    for j, st in enumerate(stores):
        delivered = float(ver["delivered"][j] + ver["unmet"][j])
        rows.append(["demand", st["name"], round(delivered, 6),
                     round(float(st["demand"]), 6),
                     round(abs(delivered - st["demand"]), 9),
                     "OK" if abs(delivered - st["demand"]) <= TOL else "FAIL"])
    for i, wh in enumerate(warehouses):
        for j, st in enumerate(stores):
            flow = float(ver["x"][i, j])
            rows.append(["route_capacity", f"{wh['name']} -> {st['name']}",
                         round(flow, 6), round(float(caps[i, j]), 6),
                         round(float(max(0.0, flow - caps[i, j])), 9),
                         "OK" if flow <= caps[i, j] + TOL else "FAIL"])
    for i, wh in enumerate(warehouses):
        for j, st in enumerate(stores):
            flow = float(ver["x"][i, j])
            rows.append(["nonnegativity", f"{wh['name']} -> {st['name']}",
                         round(flow, 6), 0.0,
                         round(float(max(0.0, -flow)), 9),
                         "OK" if flow >= -TOL else "FAIL"])
    delta = abs(float(ver["objective"]) - float(solver_obj))
    rows.append(["objective", "recomputed_vs_solver",
                 round(float(ver["objective"]), 6), round(float(solver_obj), 6),
                 round(delta, 9), "OK" if delta <= 1e-6 else "FAIL"])
    rows.append(["overall", "max_violation",
                 round(float(ver["max_violation"]), 9), TOL,
                 round(float(ver["max_violation"]), 9),
                 "OK" if ver["feasible"] else "FAIL"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "entity", "value", "limit", "violation", "status"])
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    warehouses, stores, costs, caps, params = load_data()
    supply = np.array([w["supply"] for w in warehouses])
    demand = np.array([s["demand"] for s in stores])
    n, m = costs.shape

    res, log = solve_transport(costs, supply, demand, caps, penalty=None, disp=True)
    if not res.success:
        raise SystemExit(f"Базовая задача не решена: {res.message}")
    ver = verify_transport(res, costs, supply, demand, caps)
    save_verification(OUT / "verification.csv", warehouses, stores, ver,
                      float(res.fun), caps)
    sp = shadow_prices(res, costs, supply, demand, caps)

    # Узкие места: насыщенные маршруты.
    bottlenecks = []
    for i in range(n):
        for j in range(m):
            if caps[i, j] > 0 and ver["x"][i, j] >= caps[i, j] - 1e-6 and ver["x"][i, j] > 1e-6:
                bottlenecks.append({
                    "w": warehouses[i]["name"], "s": stores[j]["name"],
                    "flow": float(ver["x"][i, j]), "cap": float(caps[i, j]),
                    "shadow": float(sp["route"][i, j]),
                    "fd": float(sp["route_fd"][i, j]),
                })

    # Сценарии
    records = {}
    records["base"] = {"status": int(res.status), "cost": float(res.fun),
                       "unmet": 0.0, "x": ver["x"], "supply_used": ver["supply_used"]}

    sup_def = supply.copy()
    sup_def[1] *= 0.80
    res_def, _ = solve_transport(costs, sup_def, demand, caps)
    if res_def.success:
        vd = verify_transport(res_def, costs, sup_def, demand, caps)
        records["deficit"] = {"status": int(res_def.status), "cost": float(res_def.fun),
                              "unmet": 0.0, "x": vd["x"],
                              "supply_used": vd["supply_used"]}
    else:
        records["deficit"] = {"status": int(res_def.status), "cost": None,
                              "unmet": float(demand.sum()), "x": None,
                              "supply_used": None}

    # Отказ самого загруженного маршрута.
    i_fail, j_fail = np.unravel_index(np.argmax(ver["x"]), ver["x"].shape)
    caps_fail = caps.copy()
    caps_fail[i_fail, j_fail] = 0.0
    res_fail, _ = solve_transport(costs, supply, demand, caps_fail)
    if res_fail.success:
        vf = verify_transport(res_fail, costs, supply, demand, caps_fail)
        records["route_failure"] = {"status": int(res_fail.status),
                                    "cost": float(res_fail.fun), "unmet": 0.0,
                                    "x": vf["x"], "supply_used": vf["supply_used"]}
    else:
        records["route_failure"] = {"status": int(res_fail.status), "cost": None,
                                    "unmet": float(demand.sum()), "x": None,
                                    "supply_used": None}

    # Стресс: два склада -20%, спрос +10%, разрешены потерянные продажи.
    sup_str = supply.copy()
    sup_str[0] *= 0.80
    sup_str[1] *= 0.80
    dem_str = demand * 1.10
    res_str, _ = solve_transport(costs, sup_str, dem_str, caps, penalty=1000.0)
    if res_str.success:
        vs = verify_transport(res_str, costs, sup_str, dem_str, caps, penalty=1000.0)
        records["stress"] = {"status": int(res_str.status),
                             "cost": vs["transport_cost"],
                             "unmet": float(vs["unmet"].sum()),
                             "penalty_cost": vs["penalty_cost"],
                             "x": vs["x"], "supply_used": vs["supply_used"]}
    else:
        records["stress"] = {"status": int(res_str.status), "cost": None,
                             "unmet": float(dem_str.sum()), "x": None,
                             "supply_used": None}

    greedy = greedy_transport(costs, supply, demand, caps)
    milp_res = solve_milp(costs, supply, demand, caps)
    milp_cost = float(milp_res.fun) if milp_res.success else None
    milp_gap = (abs(milp_cost - float(res.fun)) / max(1.0, abs(float(res.fun)))
                if milp_cost is not None else None)

    # ---------------- CSV ----------------
    with open(OUT / "lab2_flows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warehouse", "store", "flow", "capacity", "cost",
                    "transport_cost", "utilization"])
        for i, wh in enumerate(warehouses):
            for j, st in enumerate(stores):
                w.writerow([wh["name"], st["name"], round(float(ver["x"][i, j]), 4),
                            caps[i, j], costs[i, j],
                            round(float(ver["x"][i, j] * costs[i, j]), 3),
                            round(float(ver["x"][i, j] / caps[i, j]), 4)])

    with open(OUT / "lab2_balance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warehouse", "supply", "used", "remaining"])
        for i, wh in enumerate(warehouses):
            w.writerow([wh["name"], wh["supply"], round(float(ver["supply_used"][i]), 3),
                        round(float(wh["supply"] - ver["supply_used"][i]), 3)])
        w.writerow([])
        w.writerow(["store", "demand", "delivered", "shortage"])
        for j, st in enumerate(stores):
            w.writerow([st["name"], st["demand"], round(float(ver["delivered"][j]), 3),
                        round(float(st["demand"] - ver["delivered"][j]), 3)])

    with open(OUT / "lab2_scenarios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "status", "transport_cost", "unmet_demand", "feasible"])
        for key, rec in records.items():
            w.writerow([key, rec["status"], rec["cost"], round(rec["unmet"], 3),
                        rec["cost"] is not None])

    with open(OUT / "lab2_bottlenecks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warehouse", "store", "flow", "capacity", "shadow_price",
                    "finite_difference"])
        for b in bottlenecks:
            w.writerow([b["w"], b["s"], round(b["flow"], 3), b["cap"],
                        round(b["shadow"], 4), round(b["fd"], 4)])

    with open(OUT / "lab2_greedy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "cost", "unmet_demand", "feasible", "gap_percent"])
        w.writerow(["LP", round(float(res.fun), 3), 0.0, True, 0.0])
        w.writerow(["greedy", round(greedy["cost"], 3), round(greedy["unmet"], 3),
                    greedy["feasible"],
                    round((greedy["cost"] - float(res.fun)) / float(res.fun) * 100, 3)])

    with open(OUT / "lab2_milp.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "objective", "status", "gap_percent"])
        w.writerow(["LP", round(float(res.fun), 3), int(res.status), 0.0])
        w.writerow(["MILP", round(milp_cost, 3) if milp_cost is not None else "",
                    int(milp_res.status), round(milp_gap * 100, 6)
                    if milp_gap is not None else ""])

    summary = {
        "variant": 1,
        "student": "Якимов Максим Андреевич, Б24-517, № 31",
        "n_warehouses": n, "n_stores": m,
        "total_supply": float(supply.sum()), "total_demand": float(demand.sum()),
        "base_cost": float(res.fun),
        "status": res.message,
        "max_violation": ver["max_violation"],
        "feasible": ver["feasible"],
        "supply_shadow": sp["supply"],
        "supply_shadow_fd": sp["supply_fd"],
        "failed_route": [warehouses[i_fail]["name"], stores[j_fail]["name"]],
        "failure_cost": records["route_failure"]["cost"],
        "stress_transport_cost": records["stress"]["cost"],
        "stress_unmet": records["stress"]["unmet"],
        "greedy_cost": greedy["cost"],
        "greedy_gap_percent": (greedy["cost"] - float(res.fun)) / float(res.fun) * 100,
        "milp_cost": milp_cost,
        "milp_gap_percent": milp_gap * 100 if milp_gap is not None else None,
        "bottlenecks": bottlenecks,
        "seed": params["seed"],
    }
    with open(OUT / "lab2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    with open(OUT / "solver.log", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\nБАЗОВЫЙ СЦЕНАРИЙ\n" + "=" * 70 + "\n")
        f.write(log)
        f.write("\nДвойственные оценки ограничений:\n")
        f.write(np.array2string(res.ineqlin.marginals, precision=6) + "\n")
        f.write("\nСценарий дефицита W2:\n")
        f.write("(решение получено; см. lab2_scenarios.csv)\n")
        f.write("\nСтрессовый сценарий с потерянными продажами:\n")
        f.write("(решение получено; см. lab2_scenarios.csv)\n")

    # ---------------- Графики ----------------
    draw_network(warehouses, stores, ver["x"], caps, OUT / "lab2_network.png",
                 "Оптимальные перевозки склады → магазины")
    plot_matrix(costs, ver["x"], caps, [s["name"] for s in stores],
                OUT / "lab2_matrix.png")
    plot_balance(warehouses, stores, ver["supply_used"], ver["delivered"], demand,
                 OUT / "lab2_balance.png")
    plot_scenarios(records, OUT / "lab2_scenarios.png")
    plot_greedy(float(res.fun), greedy, OUT / "lab2_greedy.png")
    output = {
        "cost": float(res.fun),
        "supply_used": ver["supply_used"],
        "bottlenecks": bottlenecks,
        "greedy_cost": greedy["cost"],
    }
    plot_dashboard(warehouses, stores, ver["x"], caps, records, output,
                   OUT / "lab2_dashboard.png")

    # ---------------- Консоль ----------------
    print("=" * 65)
    print("ЛР №2. Вариант 1: транспортная задача склады -> магазины")
    print("=" * 65)
    print(f"Статус: {res.message}")
    print(f"Стоимость: {res.fun:.3f} у.е.")
    print(f"Проверка: max нарушение = {ver['max_violation']:.2e}, "
          f"допустимость = {ver['feasible']}")
    print("\nПотоки (ненулевые):")
    for i, wh in enumerate(warehouses):
        for j, st in enumerate(stores):
            if ver["x"][i, j] > 1e-6:
                print(f"  {wh['name']} -> {st['name']}: {ver['x'][i, j]:.1f} / "
                      f"{caps[i, j]:.0f}")
    print("\nТеневые цены складов:")
    for wh, v, fd in zip(warehouses, sp["supply"], sp["supply_fd"]):
        print(f"  {wh['name']}: {v:.3f} (конечная разность {fd:.3f})")
    print("\nУзкие маршруты:")
    for b in bottlenecks:
        print(f"  {b['w']} -> {b['s']}: {b['flow']:.1f}/{b['cap']:.0f}, "
              f"теневая цена {b['shadow']:.3f}")
    print("\nСценарии:")
    for key, rec in records.items():
        cost = f"{rec['cost']:.2f}" if rec["cost"] is not None else "—"
        print(f"  {key:<15} cost={cost}, unmet={rec['unmet']:.1f}")
    print(f"\nЖадная эвристика: {greedy['cost']:.2f} "
          f"(зазор {(greedy['cost'] - res.fun) / res.fun * 100:.2f}%)")
    print(f"MILP: {milp_cost}, зазор {milp_gap * 100:.4f}%")
    print(f"\nФайлы сохранены в {OUT}")


if __name__ == "__main__":
    main()
