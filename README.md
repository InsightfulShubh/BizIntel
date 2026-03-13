## BizIntel - Data Cleaning Pipeline

This project prepares YC and Crunchbase startup data for downstream RAG workflows.

### What it does

- Cleans and filters the YC + Crunchbase CSVs in `data-source/`.
- Exports cleaned outputs to `processing/data/`.
- Flags suspicious rows with `is_suspicious` for review.

### Run the pipeline (uv)

```powershell
Set-Location "c:\Users\ShubhankDubey\Shubhank_All\EPAM-2026\learning-projects\BizIntel"
uv sync
uv run python -m processing.main
```

### Outputs

- `processing/data/yc_cleaned.csv`
- `processing/data/crunchbase_cleaned.csv`

Each output includes a consistent schema with `is_suspicious`.
