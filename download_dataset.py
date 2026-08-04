from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ibm-esa-geospatial/TerraMesh",
    repo_type="dataset",
    allow_patterns="val/*",
    local_dir="data/TerraMesh",
)
