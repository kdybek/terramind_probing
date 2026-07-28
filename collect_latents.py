import os
import numpy as np
from torch.utils.data import DataLoader
from terratorch import BACKBONE_REGISTRY
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.terramesh import build_terramesh_dataset, Transpose, MultimodalTransforms, MultimodalNormalize, statistics
import pandas as pd
import zarr
import pickle
import subprocess


MODALITIES = ["S2L2A", "S2L1C", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_METADATA_URL = "https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/val_metadata.parquet"
DATA_DIR = "data"
VAL_METADATA_PATH = os.path.join(DATA_DIR, "val_metadata.parquet")
ZARR_PATH = os.path.join(DATA_DIR, "latents.zarr.zip")
META_PATH = os.path.join(DATA_DIR, "metadata.pkl")


def get_metadata(metadata_df, key):
    key_zarr = f"{key}.zarr.zip"
    row = metadata_df[metadata_df["zarr"] == key_zarr]

    if row.empty:
        print(f"Warning: No metadata found for key {key}.")
        return None
    else:
        return row.iloc[0]


def load_models():
    model_names = ["terramind_v1_tiny", "terramind_v1_small", "terramind_v1_base", "terramind_v1_large"]
    latent_dims = [192, 384, 768, 1024]
    models = {}
    latent_dim_dict = dict(zip(model_names, latent_dims))
    for name in model_names:
        model = BACKBONE_REGISTRY.build(
            name,
            pretrained=True,
            modalities=MODALITIES,
            merge_method='mean'
        )
        model.to(DEVICE)
        model.eval()
        models[name] = model

    return models, latent_dim_dict


def main():
    val_transform = MultimodalTransforms(
        transforms=A.Compose([  # We use albumentations because of the shared transform between image modalities
            # Convert data to channel last (expected shape from albumentations)
            Transpose([1, 2, 0]),
            MultimodalNormalize(mean=statistics["mean"], std=statistics["std"]),
            A.CenterCrop(224, 224),  # Use center crop in val split
            # A.RandomCrop(224, 224),  # Use random crop in train split
            # A.D4(),  # Optionally, use random flipping and rotation for the train split
            ToTensorV2(),  # Convert to tensor and back to channel first
        ],
            is_check_shapes=False,  # Not needed because of aligned data in TerraMesh
            additional_targets={m: "image" for m in MODALITIES}
        ),
        non_image_modalities=["__key__", "__url__"],  # Additional non-image keys
    )

    dataset = build_terramesh_dataset(
        path="https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh/resolve/main/",
        modalities=MODALITIES,
        split="val",
        transform=val_transform,
        shuffle=False,
        batch_size=256,
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=64,
                            persistent_workers=True, prefetch_factor=2)

    os.makedirs(DATA_DIR, exist_ok=True)

    subprocess.run(["wget", "-O", VAL_METADATA_PATH, VAL_METADATA_URL], check=True)
    metadata_df = pd.read_parquet(VAL_METADATA_PATH)

    models, latent_dim_dict = load_models()

    root = zarr.open(ZARR_PATH, mode="w")

    datasets = {}
    for model_name, model in models.items():
        latent_dim = latent_dim_dict[model_name]
        datasets[model_name] = root.create_dataset(
            model_name,
            shape=(0, latent_dim),
            chunks=(256, latent_dim),
            dtype=np.float32,
        )

    metadatas = []
    with torch.no_grad():
        for batch in dataloader:
            keys = batch['__key__']

            for key in keys:
                metadata = get_metadata(metadata_df, key)
                metadatas.append(metadata)

            input = {m: batch[m].to(DEVICE).float() for m in MODALITIES if m in batch}

            for model_name, model in models.items():
                dataset = datasets[model_name]
                latent = model(input)[-1].mean(axis=1).cpu().numpy()
                dataset.append(latent)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadatas, f)


if __name__ == "__main__":
    main()
