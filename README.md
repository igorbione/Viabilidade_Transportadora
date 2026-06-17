# Logistics Feasibility Analysis — E-commerce Client Expansion

**Route Clustering & Fleet Dimensioning for a São Paulo Carrier**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-yellow)
![Folium](https://img.shields.io/badge/Maps-Folium-green)

---

## Overview

A small carrier operating in the São Paulo metropolitan area was evaluating whether to absorb a new e-commerce client's pickup operation as a significant business expansion. The client had ~270 active sellers spread across São Paulo and 4 surrounding municipalities, dispatching an estimated 5,000+ packages per day.

This project delivers a data-driven answer to the carrier's key question: **can we serve this client, and how many vehicles would we need?**

---

## Problem Statement

- **Input:** A spreadsheet of 270 seller pickup addresses — no geographic coordinates, no route assignments.
- **Challenge 1 — Geocoding:** Brazilian street addresses provided only zip codes (CEPs); coordinates had to be resolved from scratch.
- **Challenge 2 — Route design:** 270 unstructured pickup points need to be grouped into operationally viable zones before fleet sizing can begin.
- **Challenge 3 — Uncertainty:** Package volume varies day to day, and São Paulo traffic is unpredictable. A single deterministic fleet estimate would be misleading.

---

## Methodology

The analysis follows a three-stage pipeline (one notebook per stage):

### Stage 1 — Data Preparation (`01_data_preparation.ipynb`)
- Load raw prospect spreadsheet (270 rows)
- Two-pass geocoding: CEP → street string via [ViaCEP API](https://viacep.com.br), then coordinates via [Nominatim/OpenStreetMap](https://nominatim.org)
- Exponential-backoff retry for failed addresses
- Output: `enderecos_com_coordenadas.csv`

### Stage 2 — Cluster Analysis (`02_cluster_analysis.ipynb`)
- Exploratory analysis: volume distribution, geographic spread, top areas by demand
- Feature engineering: `[lat_scaled, lon_scaled, volume_normalized × 0.3]`
- Optimal k selection: elbow method + silhouette score (averaged when methods disagree)
- K-means clustering (`n_init=20` to avoid local minima)
- Interactive Folium map with cluster coverage radii

### Stage 3 — Fleet Dimensioning (`03_fleet_recommendations.ipynb`)
- Monte Carlo simulation: **5,000 runs** per (cluster × capacity × trips/day) combination
- Uncertainty modeled: ±10% volume, ±10% service time, 1.0–1.3× traffic factor
- P5/P50/P95 vehicle count distributions per scenario
- Sensitivity analysis: full scope vs. phased onboarding (excluding Brás high-density zone)

---

## Key Results

### Geographic Clustering

3 optimal clusters identified across the São Paulo metro area:

| Cluster | Pickup Points | Total Volume | Max Radius |
|---------|--------------|-------------|------------|
| 0 | ~45 | ~240 pkg | ~12 km |
| 1 | ~46 | ~290 pkg | ~26 km |
| 2 | ~179 | ~4,550 pkg | ~30 km |

Cluster 2 dominates: 66% of pickup points and ~89% of volume, centered on São Paulo proper and extending to Cajamar.

### Fleet Recommendations (Full Scope — Moderate Scenario)

| Scenario | Capacity/Vehicle | Trips/Day | Fleet Range (P5–P95) |
|---|---|---|---|
| Conservative | 100 packages | 1 | larger fleet |
| **Moderate (recommended)** | **150 packages** | **1** | **~25–31 vehicles** |
| Optimistic | 300 packages | 2 | smaller fleet |

### Phased Onboarding Option

Excluding the **Brás zone** (45 high-density pickup points, ~17% of volume) from the initial scope reduces the recommended fleet to **~13–16 vehicles** — a viable entry point for a carrier that wants to validate the operation before committing to full coverage.

### Notable Findings

- Volume distribution is heavily right-skewed: median ~3 packages/point, max 796 packages at a single location. This outlier drives fleet sizing for its cluster disproportionately.
- ~98% of pickup points fall within São Paulo city limits; 5 points span Cotia, Cajamar, Santo André, Jandira.
- Geographic coverage spans ~55 km × 55 km.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data processing | `pandas`, `numpy` |
| Machine learning | `scikit-learn` (KMeans, StandardScaler, silhouette_score) |
| Geospatial | `geopy` (Nominatim, geodesic), `folium` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Geocoding APIs | Nominatim (OpenStreetMap), ViaCEP |
| Export | `openpyxl` |

---

## Repository Structure

```
Viabilidade_Transportadora/
├── data/
│   ├── raw/                        # Original prospect spreadsheet (gitignored)
│   └── processed/                  # Geocoded logistics data (gitignored)
├── notebooks/
│   ├── 01_data_preparation.ipynb   # Geocoding pipeline
│   ├── 02_cluster_analysis.ipynb   # K-means clustering & visualization
│   └── 03_fleet_recommendations.ipynb  # Monte Carlo simulation & results
├── src/
│   └── logistics_optimizer.py      # LogisticsOptimizer class
├── outputs/                        # Generated maps and Excel reports (gitignored)
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn geopy folium plotly openpyxl pgeocode requests

# Run notebooks in order
jupyter notebook notebooks/01_data_preparation.ipynb  # requires internet (geocoding APIs)
jupyter notebook notebooks/02_cluster_analysis.ipynb  # runs offline
jupyter notebook notebooks/03_fleet_recommendations.ipynb  # runs offline
```

> Notebooks 02 and 03 are self-contained and can run using the pre-geocoded CSV in `data/processed/` without re-running stage 01.

---

## Data Privacy

This project was delivered as a freelance engagement. Real seller addresses and client identifiers have been removed from version control (see `.gitignore`). Column names and client references have been generalized. The `LogisticsOptimizer` class includes a `generate_sample_data()` method to demonstrate the full pipeline without real data.

---

---

## Versão em Português

### Contexto

Este projeto analisa a viabilidade de absorção de uma nova operação de coleta para um cliente de e-commerce por parte de uma transportadora na região metropolitana de São Paulo. A análise responde à pergunta central: **vale a pena expandir? E se sim, quantos veículos são necessários?**

### Problema

- **Entrada:** Planilha com 270 endereços de vendedores sem coordenadas geográficas.
- **Desafio 1 — Geocodificação:** Apenas CEPs disponíveis; coordenadas precisavam ser resolvidas via API.
- **Desafio 2 — Design de rotas:** 270 pontos de coleta sem agrupamento prévio.
- **Desafio 3 — Incerteza:** Volume diário e trânsito paulistano são variáveis — uma estimativa determinística seria imprecisa.

### Metodologia

1. **Preparação dos dados (`01_data_preparation.ipynb`):** Geocodificação de 270 endereços via CEP (ViaCEP API) + coordenadas (Nominatim/OpenStreetMap), com retry exponencial para falhas.

2. **Análise de clusters (`02_cluster_analysis.ipynb`):** K-means ponderado por volume (`peso = 0.3`). Número ótimo de clusters determinado pelo método do cotovelo + silhueta (k=3). Mapa interativo com Folium.

3. **Dimensionamento de frota (`03_fleet_recommendations.ipynb`):** Simulação de Monte Carlo com 5.000 iterações por cenário. Variáveis de incerteza: volume (±10%), tempo de serviço (±10%), fator de trânsito (1,0×–1,3×). Distribuições P5/P50/P95.

### Resultados Principais

**Cenário recomendado (moderado): 150 pacotes/veículo, 1 viagem/dia**
- Escopo completo: **~25–31 veículos** (P5–P95)
- Excluindo Brás (entrada gradual): **~13–16 veículos**

**Fatores de atenção:**
- Cluster 2 concentra ~89% do volume total — é o dimensionador crítico da frota.
- Um vendedor com volume máximo de 796 pacotes é um outlier relevante; pode justificar veículo dedicado.
- Região do Brás (45 pontos de coleta, alta densidade) representa uma decisão estratégica de onboarding.

### Como Executar

```bash
pip install pandas numpy scikit-learn geopy folium plotly openpyxl pgeocode requests
jupyter notebook notebooks/01_data_preparation.ipynb
jupyter notebook notebooks/02_cluster_analysis.ipynb
jupyter notebook notebooks/03_fleet_recommendations.ipynb
```
