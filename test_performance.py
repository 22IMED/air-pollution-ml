"""
Tests de charge et de stress — Pipeline Air Pollution
======================================================
• Test de charge  : montée progressive du volume de données (20 % → 100 % du dataset)
• Test de stress  : répétition intensive sur le volume maximal + mesure de stabilité

Dépendances : pytest, tracemalloc, psutil (optionnel pour la RAM process complète)
"""

import time
import tracemalloc
import pytest
import pandas as pd
import numpy as np

from pipeline import load_artifacts, run_pipeline

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def artifacts():
    """Charge les artefacts une seule fois pour tous les tests du module."""
    return load_artifacts()


@pytest.fixture(scope="module")
def full_df():
    return pd.read_csv("global_air_pollution_dataset.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _write_tmp_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEST DE CHARGE : 5 paliers de volume (20 % → 100 %)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.perf
@pytest.mark.parametrize("fraction", [0.2, 0.4, 0.6, 0.8, 1.0])
def test_load_time_by_volume(fraction, full_df, artifacts, tmp_path):
    """
    La prédiction doit rester sous MAX_SECONDS secondes,
    quelle que soit la taille du sous-ensemble.
    """
    MAX_SECONDS = 10.0

    model, scaler, imputer, iso_forest = artifacts
    subset = full_df.sample(frac=fraction, random_state=42)
    csv_file = str(tmp_path / f"subset_{int(fraction*100)}.csv")
    _write_tmp_csv(subset, csv_file)

    tracemalloc.start()
    t0 = time.perf_counter()

    results = run_pipeline(csv_file, model, scaler, imputer, iso_forest)

    elapsed          = time.perf_counter() - t0
    _, peak_ram      = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_ram / (1024 * 1024)
    n_rows      = len(subset)

    print(f"\n[CHARGE] fraction={fraction:.0%} | n={n_rows:>6} | "
          f"time={elapsed:.4f}s | peak RAM={peak_ram_mb:.2f} MB")

    assert elapsed < MAX_SECONDS, (
        f"Pipeline trop lent sur {n_rows} lignes : {elapsed:.2f}s > {MAX_SECONDS}s"
    )
    assert len(results) == n_rows, "Le nombre de lignes en sortie ne correspond pas"


# ─────────────────────────────────────────────────────────────────────────────
# TEST DE CHARGE RAM : le pic mémoire ne doit pas exploser
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.perf
def test_memory_ceiling(full_df, artifacts, tmp_path):
    """Le pic mémoire sur le dataset complet doit rester sous MAX_RAM_MB."""
    MAX_RAM_MB = 500.0

    model, scaler, imputer, iso_forest = artifacts
    csv_file = str(tmp_path / "full.csv")
    _write_tmp_csv(full_df, csv_file)

    tracemalloc.start()
    run_pipeline(csv_file, model, scaler, imputer, iso_forest)
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_ram_mb = peak_ram / (1024 * 1024)
    print(f"\n[RAM] Peak = {peak_ram_mb:.2f} MB (limite : {MAX_RAM_MB} MB)")

    assert peak_ram_mb < MAX_RAM_MB, (
        f"Consommation mémoire trop élevée : {peak_ram_mb:.1f} MB > {MAX_RAM_MB} MB"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST DE STRESS : N_RUNS répétitions consécutives sur le dataset complet
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.perf
def test_stress_repeated_calls(full_df, artifacts, tmp_path):
    """
    Exécute le pipeline N_RUNS fois de suite.
    Vérifie que le temps moyen reste stable (pas de dégradation progressive).
    """
    N_RUNS         = 10
    MAX_AVG_SECONDS = 5.0
    MAX_STD_FACTOR  = 0.5   # l'écart-type ne doit pas dépasser 50 % de la moyenne

    model, scaler, imputer, iso_forest = artifacts
    csv_file = str(tmp_path / "full_stress.csv")
    _write_tmp_csv(full_df, csv_file)

    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        run_pipeline(csv_file, model, scaler, imputer, iso_forest)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times)
    std = np.std(times)

    print(f"\n[STRESS] {N_RUNS} runs | "
          f"avg={avg:.4f}s | std={std:.4f}s | "
          f"min={min(times):.4f}s | max={max(times):.4f}s")

    assert avg < MAX_AVG_SECONDS, (
        f"Temps moyen trop élevé : {avg:.2f}s > {MAX_AVG_SECONDS}s"
    )
    assert std < avg * MAX_STD_FACTOR, (
        f"Instabilité détectée : std={std:.4f}s représente "
        f"{100*std/avg:.0f}% du temps moyen"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST DE STRESS CONCURRENTIELS (threads simulés séquentiellement)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.perf
def test_stress_concurrent_batches(full_df, artifacts, tmp_path):
    """
    Simule N_BATCHES requêtes de tailles aléatoires envoyées rapidement.
    Toutes doivent se terminer sans erreur et en moins de BATCH_MAX_SECONDS.
    """
    N_BATCHES        = 20
    BATCH_MAX_SECONDS = 3.0
    rng = np.random.default_rng(0)

    model, scaler, imputer, iso_forest = artifacts
    n_total = len(full_df)

    failures = []
    for i in range(N_BATCHES):
        size   = rng.integers(50, n_total // 4)
        subset = full_df.sample(n=size, random_state=int(rng.integers(1e6)))
        csv_file = str(tmp_path / f"batch_{i}.csv")
        _write_tmp_csv(subset, csv_file)

        t0 = time.perf_counter()
        try:
            result  = run_pipeline(csv_file, model, scaler, imputer, iso_forest)
            elapsed = time.perf_counter() - t0
            if elapsed > BATCH_MAX_SECONDS:
                failures.append(
                    f"Batch {i} ({size} lignes) : {elapsed:.2f}s > {BATCH_MAX_SECONDS}s"
                )
            if len(result) != size:
                failures.append(f"Batch {i} : sortie {len(result)} ≠ entrée {size}")
        except Exception as e:
            failures.append(f"Batch {i} a levé une exception : {e}")

    assert not failures, "Échecs lors du stress test:\n" + "\n".join(failures)
