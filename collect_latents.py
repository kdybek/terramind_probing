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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "data/TerraMesh"
DATA_DIR = "data"
ZARR_PATH = os.path.join(DATA_DIR, "latents.zarr")
META_PATH = os.path.join(DATA_DIR, "metadata.pkl")

METADATA_KEYS = ["__key__", "__url__", "center_lon", "center_lat", "time_S2L2A", "time_DEM"]
MODELS = ["terramind_v1_tiny", "terramind_v1_small", "terramind_v1_base", "terramind_v1_large"]
LATENT_DIMS_DICT = {
    "terramind_v1_tiny": 192,
    "terramind_v1_small": 384,
    "terramind_v1_base": 768,
    "terramind_v1_large": 1024
}

TERRAMIND_V1_MEANS = [
    1390.458,
    1503.317,
    1718.197,
    1853.91,
    2199.1,
    2779.975,
    2987.011,
    3083.234,
    3132.22,
    3162.988,
    2424.884,
    1857.648,
]
TERRAMIND_V1_STDS = [
    2106.761,
    2141.107,
    2038.973,
    2134.138,
    2085.321,
    1889.926,
    1820.257,
    1871.918,
    1753.829,
    1797.379,
    1434.261,
    1334.311,
]


def load_model(model_name):
    model = BACKBONE_REGISTRY.build(
        model_name,
        pretrained=True,
        modalities=["S2L2A"],
        merge_method='mean'
    )
    model.to(DEVICE)
    model.eval()

    return model


def prepare_input(input, model_name):
    if model_name.startswith("terramind_v1_"):
        means = torch.tensor(TERRAMIND_V1_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(TERRAMIND_V1_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
        return input


def main():
    val_transform = MultimodalTransforms(
        transforms=A.Compose([  # We use albumentations because of the shared transform between image modalities
            # Convert data to channel last (expected shape from albumentations)
            Transpose([1, 2, 0]),
            A.CenterCrop(224, 224),  # Use center crop in val split
            # A.RandomCrop(224, 224),  # Use random crop in train split
            # A.D4(),  # Optionally, use random flipping and rotation for the train split
            ToTensorV2(),  # Convert to tensor and back to channel first
        ],
            is_check_shapes=False,  # Not needed because of aligned data in TerraMesh
            additional_targets={m: "image" for m in ["S2L2A", "DEM", "cloud_mask"]}  # Specify additional modalities to transform
        ),
        non_image_modalities=["__key__", "__url__", "center_lon", "center_lat", "time_S2L2A", "time_DEM"],  # Additional non-image keys
    )

    dataset = build_terramesh_dataset(
        path=DATASET_PATH,
        modalities=["S2L2A", "DEM"],
        split="val",
        transform=val_transform,
        shuffle=False,
        return_metadata=True,
        batch_size=128,
    )

    dataloader = DataLoader(dataset, batch_size=None, num_workers=16,
                            persistent_workers=True, prefetch_factor=2)

    os.makedirs(DATA_DIR, exist_ok=True)

    root = zarr.open(ZARR_PATH, mode="w")

    for model_name in MODELS:
        latent_dim = LATENT_DIMS_DICT[model_name]

        root.create_dataset(
            model_name,
            shape=(0, latent_dim),
            chunks=(256, latent_dim),
            dtype=np.float32,
        )

    per_model_metadata = {}
    for model_name in MODELS:
        model = load_model(model_name)

        metadatas = []
        with torch.no_grad():
            for batch in dataloader:
                dems = batch['DEM']
                meta = [{key: batch[key][i] for key in METADATA_KEYS} for i in range(len(dems))]

                for dem in dems:
                    C, H, W = dem.shape
                    assert C == 1, f"Expected DEM to have 1 channel, but got {C} channels."
                    elevation = dem[0, H // 2, W // 2].item()  # Get the elevation at the center pixel

                    metadata = meta.pop(0)
                    metadata['elevation'] = elevation
                    metadatas.append(metadata)

                input = batch['S2L2A'].float()
                input = prepare_input(input, model_name)
                input = input.to(DEVICE)
                latents = model(input)[-1].mean(axis=1).cpu().numpy()  # Average over spatial dimensions
                root[model_name].append(latents)

        per_model_metadata[model_name] = metadatas

    with open(META_PATH, "wb") as f:
        pickle.dump(per_model_metadata, f)

    per_model_counts = {model_name: len(per_model_metadata[model_name]) for model_name in MODELS}

    print(f"Latents saved to {ZARR_PATH} and metadata saved to {META_PATH}")
    print(f"Processed samples per model: {per_model_counts}.")


if __name__ == "__main__":
    main()
