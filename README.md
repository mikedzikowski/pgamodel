# PGA One and Done — Pick Optimizer

Season-long PGA Tour "One and Done" pick optimizer with a Streamlit dashboard.

Blends **DataGolf** skill-model predictions with **Kalshi** prediction-market signals
to surface the best weekly pick given which players you've already used.

---

## Features

- **Full field rankings** — DataGolf baseline + course-history-fit model side by side
- **Kalshi market integration** — normalized implied win probabilities blended with the model
- **5-year course history** — per-player finish history at the current venue
- **DG vs Kalshi divergences** — visual bar charts highlighting model/market disagreements
- **OAD pick card** — week's recommended pick excluding players you've already used
- **ILP optimizer** — season-long optimal schedule using PuLP

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mikedzikowski/pga-oad.git
cd pga-oad
python -m venv .venv          # or: uv venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"       # or: uv pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your DataGolf API key
```

**.env**
```
DATAGOLF_API_KEY=your_api_key_here
```

A DataGolf API key is required. Get one at [datagolf.com](https://datagolf.com).

Kalshi markets are fetched via the public REST API — no Kalshi account needed.

### 3. Run the Streamlit dashboard

```bash
.venv/bin/streamlit run app.py
# Opens at http://localhost:8501
```

---

## Command-line scripts

| Script | Description |
|--------|-------------|
| `scripts/fetch_field.py` | Print current field with win probabilities |
| `scripts/blended_picks.py` | DG + Kalshi blended rankings and divergences |
| `scripts/course_history.py` | 5-year course history for current event |
| `scripts/analysis.py` | Full analysis: field, specialists, model/market outliers |
| `scripts/fetch_schedule.py` | Upcoming PGA Tour schedule |

```bash
# Examples
python scripts/blended_picks.py --top 30
python scripts/blended_picks.py --edge-only
python scripts/analysis.py
python scripts/course_history.py
```

---

## Project structure

```
pga-oad/
├── app.py                  # Streamlit dashboard
├── pyproject.toml
├── .env.example            # Copy to .env and fill in your key
├── scripts/
│   ├── fetch_field.py
│   ├── blended_picks.py
│   ├── course_history.py
│   ├── analysis.py
│   └── fetch_schedule.py
├── src/pga_oad/
│   ├── client.py           # DataGolf REST API client
│   ├── kalshi.py           # Kalshi prediction market client
│   ├── blend.py            # Signal blender (DG + Kalshi)
│   ├── optimizer.py        # ILP One-and-Done optimizer (PuLP)
│   ├── models.py           # Pydantic models
│   └── cache.py            # File-based JSON cache
└── tests/
    └── test_client.py
```

---

## Data sources

| Source | Data | Auth |
|--------|------|------|
| [DataGolf API](https://datagolf.com/api-overview) | Pre-tournament predictions, course history archive, schedule | API key (query param) |
| [Kalshi](https://kalshi.com) | PGA Tour prediction markets (win, top-5, top-10, top-20, make cut, H2H) | None (public GET) |

---

## License

MIT
