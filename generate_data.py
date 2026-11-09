# -*- coding: utf-8 -*-
"""
Генерация данных варианта 1 для ЛР №2 по методам оптимизации.
Задача: транспортная задача «склады → магазины», минимум стоимости.
Минимальный масштаб варианта: 4 склада, 6 магазинов.

Seed: 10000 + 100*Q + N = 10131.
"""

from pathlib import Path
import csv
import json
import numpy as np

SEED = 10131
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

WAREHOUSES = ["Склад W1", "Склад W2", "Склад W3", "Склад W4"]
STORES = ["Магазин S1", "Магазин S2", "Магазин S3",
          "Магазин S4", "Магазин S5", "Магазин S6"]
SUPPLY = np.array([130.0, 110.0, 100.0, 90.0])
DEMAND = np.array([60.0, 70.0, 50.0, 80.0, 60.0, 70.0])
BASE_COST = np.array([
    [3.0, 5.0, 7.0, 6.0, 8.0, 9.0],
    [6.0, 4.0, 5.0, 7.0, 6.0, 8.0],
    [8.0, 7.0, 4.0, 5.0, 7.0, 6.0],
    [9.0, 8.0, 6.0, 4.0, 5.0, 7.0],
])


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    costs = np.round(BASE_COST * rng.normal(1.0, 0.08, BASE_COST.shape), 2)
    caps = rng.integers(45, 86, size=BASE_COST.shape).astype(float)
    # Договорное ограничение на маршрут W3 → S6: не более 60 ед.
    caps[2, 5] = 60.0

    with open(DATA / "warehouses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warehouse_id", "warehouse", "supply"])
        for i, name in enumerate(WAREHOUSES):
            w.writerow([i + 1, name, float(SUPPLY[i])])

    with open(DATA / "stores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["store_id", "store", "demand"])
        for j, name in enumerate(STORES):
            w.writerow([j + 1, name, float(DEMAND[j])])

    with open(DATA / "routes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["warehouse", "store", "cost", "capacity"])
        for i in range(len(WAREHOUSES)):
            for j in range(len(STORES)):
                w.writerow([WAREHOUSES[i], STORES[j], float(costs[i, j]),
                            float(caps[i, j])])

    with open(DATA / "params.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "n_warehouses": len(WAREHOUSES),
            "n_stores": len(STORES),
            "total_supply": float(SUPPLY.sum()),
            "total_demand": float(DEMAND.sum()),
            "units": {"flow": "ед. груза", "cost": "у.е. за ед.",
                      "capacity": "ед.", "supply_demand": "ед."},
            "note": "Синтетическая транспортная сеть; seed фиксирован.",
        }, f, ensure_ascii=False, indent=2)

    print("Данные сохранены в", DATA)
    print("Стоимости:\n", costs)
    print("Пропускные способности:\n", caps.astype(int))


if __name__ == "__main__":
    main()
