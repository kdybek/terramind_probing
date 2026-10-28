import os
import numpy as np
from torch.utils.data import DataLoader
from terratorch import BACKBONE_REGISTRY
import torch
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from utils.terramesh import build_terramesh_dataset, Transpose, MultimodalTransforms
import zarr
import pickle
import thor_terratorch_ext


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "data/TerraMesh"
DATA_DIR = "data"
ZARR_PATH = os.path.join(DATA_DIR, "latents.zarr")
META_PATH = os.path.join(DATA_DIR, "metadata.pkl")

METADATA_KEYS = ["__key__", "__url__", "center_lon", "center_lat", "time_S2L2A", "time_DEM"]
MODELS = [
    "terramind_v1_tiny", 
    "terramind_v1_small", 
    "terramind_v1_base", 
    "terramind_v1_large", 
    "thor_v1_large", 
    "prithvi_eo_v2_600"
]
LATENT_DIMS_DICT = {
    "terramind_v1_tiny": 192,
    "terramind_v1_small": 384,
    "terramind_v1_base": 768,
    "terramind_v1_large": 1024,
    "prithvi_eo_v2_600": 1280,
}

S2L2A_BANDS = [
    "COASTAL_AEROSOL",  # B1
    "BLUE",             # B2
    "GREEN",            # B3
    "RED",              # B4
    "RED_EDGE_1",       # B5
    "RED_EDGE_2",       # B6
    "RED_EDGE_3",       # B7
    "NIR_BROAD",        # B8
    "NIR_NARROW",       # B8A
    "WATER_VAPOR",      # B9
    "SWIR_1",           # B11
    "SWIR_2",           # B12
]

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

THOR_MEANS = [
    0.182569,  # COASTAL_AEROSOL (B1)
    0.176620,  # BLUE (B2)
    0.195923,  # GREEN (B3)
    0.213948,  # RED (B4)
    0.263378,  # RED_EDGE_1 (B5)
    0.300818,  # RED_EDGE_2 (B6)
    0.313144,  # RED_EDGE_3 (B7)
    0.308133,  # NIR_BROAD (B8)
    0.320993,  # NIR_NARROW (B8A)
    0.322589,  # WATER_VAPOR (B9)
    0.221550,  # SWIR_1 (B11)
    0.175772,  # SWIR_2 (B12)
]
THOR_STDS = [
    0.282463,  # COASTAL_AEROSOL (B1)
    0.264520,  # BLUE (B2)
    0.252949,  # GREEN (B3)
    0.259180,  # RED (B4)
    0.272771,  # RED_EDGE_1 (B5)
    0.248175,  # RED_EDGE_2 (B6)
    0.235432,  # RED_EDGE_3 (B7)
    0.226434,  # NIR_BROAD (B8)
    0.223274,  # NIR_NARROW (B8A)
    0.235857,  # WATER_VAPOR (B9)
    0.171606,  # SWIR_1 (B11)
    0.156223,  # SWIR_2 (B12)
]

PRITHVI_V2_MEANS = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_V2_STDS = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]


def load_model(model_name):
    if model_name.startswith("terramind_v1_"):
        model = BACKBONE_REGISTRY.build(
            model_name,
            pretrained=True,
            modalities=["S2L2A"],
            merge_method='mean'
        )
    elif model_name.startswith("thor_v1_"):
        model = BACKBONE_REGISTRY.build(
            model_name,
            pretrained=True,
            model_bands=S2L2A_BANDS,
            input_params=dict(  # Optional input parameters to customize
                ground_covers=[
                    10 * 224 * 224
                ],  # Ground cover in meters (typically input image size [px] * input image resolution)
                flexivit_patch_size_seqs=[16],  # Patch size in pixels
            ),
            merge_method='mean'
        )
    elif model_name.startswith("prithvi_v2_"):
        model = BACKBONE_REGISTRY.build(
            model_name,
            pretrained=True,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(DEVICE)
    model.eval()

    return model


def prepare_input(input, model_name):
    if model_name.startswith("terramind_v1_"):
        means = torch.tensor(TERRAMIND_V1_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(TERRAMIND_V1_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
    elif model_name.startswith("prithvi_v2_"):
        input = input[:, [1, 2, 3, 8, 10, 11], :, :]  # Select bands: B2, B3, B4, B8A, B11, B12
        # Scale from 10m to 30m resolution
        input = F.interpolate(input, size=(224, 224), mode='area')
        means = torch.tensor(PRITHVI_V2_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(PRITHVI_V2_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
    elif model_name.startswith("thor_v1_"):
        input = input / 10000.0
        means = torch.tensor(THOR_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(THOR_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return input


def get_dataset(model_name):
    if model_name.startswith("prithvi_v2_"):
        crop_size = 224 * 3
    else:
        crop_size = 224

    val_transform = MultimodalTransforms(
        transforms=A.Compose([  # We use albumentations because of the shared transform between image modalities
            # Convert data to channel last (expected shape from albumentations)
            Transpose([1, 2, 0]),
            A.CenterCrop(crop_size, crop_size),  # Use center crop in val split
            # A.RandomCrop(224, 224),  # Use random crop in train split
            # A.D4(),  # Optionally, use random flipping and rotation for the train split
            ToTensorV2(),  # Convert to tensor and back to channel first
        ],
            is_check_shapes=False,  # Not needed because of aligned data in TerraMesh
            additional_targets={m: "image" for m in ["S2L2A", "DEM", "cloud_mask"]}  # Specify additional modalities to transform
        ),
        non_image_modalities=["__key__", "__url__", "center_lon", "center_lat", "time_S2L2A", "time_DEM"],  # Additional non-image keys
    )

    return build_terramesh_dataset(
        path=DATASET_PATH,
        modalities=["S2L2A", "DEM"],
        split="val",
        transform=val_transform,
        shuffle=False,
        return_metadata=True,
        batch_size=128,
    )

    
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    root = zarr.open(ZARR_PATH, mode="w")

    for model_name in MODELS:
        latent_dim = LATENT_DIMS_DICT[model_name]

        root.create_dataset(
            model_name,
            shape=(0, latent_dim),
            chunks=(1024, latent_dim),
            dtype=np.float32,
        )

    per_model_metadata = {}
    for model_name in MODELS:
        dataset = get_dataset(model_name)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=16,
                                persistent_workers=True, prefetch_factor=2)
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

    values = per_model_metadata.values()
    assert all(values[0] == v for v in values[1:]), "Metadata mismatch between models."

    metadata = values[0]

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Latents saved to {ZARR_PATH} and metadata saved to {META_PATH}")
    print(f"Processed {len(metadata)} samples.")


if __name__ == "__main__":
    main()
