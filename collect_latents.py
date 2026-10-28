import os
import numpy as np
from torch.utils.data import DataLoader
from terratorch import BACKBONE_REGISTRY
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from vendor.terramesh import build_terramesh_dataset, Transpose, MultimodalTransforms
from vendor.use_croma import PretrainedCROMA
import zarr
import pickle
import subprocess
import thor_terratorch_ext
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "data/TerraMesh"
DATA_DIR = "data"
ZARR_PATH = os.path.join(DATA_DIR, "latents.zarr")
META_PATH = os.path.join(DATA_DIR, "metadata.pkl")

METADATA_KEYS = ["__key__", "__url__", "center_lon", "center_lat", "time_S2L2A"]
MODELS = [
    "terramind_v1_tiny", 
    "terramind_v1_small", 
    "terramind_v1_base", 
    "terramind_v1_large", 
    "thor_v1_large", 
    "prithvi_eo_v2_600",
    "dofa_large_patch16_224",
    "croma_large",
    "pixel_stats"  # Added pixel_stats as a model for comparison
]
LATENT_DIMS_DICT = {
    "terramind_v1_tiny": 192,
    "terramind_v1_small": 384,
    "terramind_v1_base": 768,
    "terramind_v1_large": 1024,
    "thor_v1_large": 1024,
    "prithvi_eo_v2_600": 1280,
    "dofa_large_patch16_224": 1024,
    "croma_large": 1024,
    "pixel_stats": 24,  # 12 means + 12 stds
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
PRITHVI_V2_BANDS = [
    "BLUE",             # B2
    "GREEN",            # B3
    "RED",              # B4
    "NIR_NARROW",       # B8A
    "SWIR_1",           # B11
    "SWIR_2",           # B12
]

DOFA_MEANS = [114.1099739 , 114.81779093, 126.63977424,  84.33539309,
              97.84789168, 103.94461911, 101.435633, 72.32804172,
              56.66528851]
DOFA_STDS = [77.84352553, 69.96844919, 67.42465279, 64.57022983, 61.72545487,
             61.34187099, 60.29744676, 47.88519516, 42.55886798]
DOFA_BANDS = ["RED", "GREEN", "BLUE", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_BROAD", "SWIR_1", "SWIR_2"]

CROMA_WEIGHTS_LINK = "https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_large.pt"
CROMA_WEIGHTS_PATH = "data/croma_large.pt"


def pixel_stats(x):
    mean = x.mean(dim=[2, 3])
    std = x.std(dim=[2, 3])

    return torch.cat([mean, std], dim=1)


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
                    2400
                ],  # Ground cover in meters (typically input image size [px] * input image resolution)
                flexivit_patch_size_seqs=[8],  # Patch size in pixels
            ),
            merge_method='mean'
        )
    elif model_name.startswith("prithvi_eo_v2_"):
        model = BACKBONE_REGISTRY.build(
            model_name,
            pretrained=True,
        )
    elif model_name.startswith("dofa_"):
        model = BACKBONE_REGISTRY.build(
            model_name,
            pretrained=True,
            model_bands=DOFA_BANDS,
        )
    elif model_name == "croma_large":
        if not os.path.exists(CROMA_WEIGHTS_PATH):
            print(f"Downloading CROMA weights from {CROMA_WEIGHTS_LINK}...")
            subprocess.run(["wget", "-O", CROMA_WEIGHTS_PATH, CROMA_WEIGHTS_LINK], check=True)
        model = PretrainedCROMA(pretrained_path=CROMA_WEIGHTS_PATH, size='large', modality='optical', image_resolution=120)
    elif model_name == "pixel_stats":
        return None
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(DEVICE)
    model.eval()

    return model


def satmae_normalize(x):
    # taken from SatMAE and SeCo
    x = x.float()

    imgs = []
    for channel in range(x.shape[1]):
        min_value = x[:, channel, :, :].mean() - 2 * x[:, channel, :, :].std()
        max_value = x[:, channel, :, :].mean() + 2 * x[:, channel, :, :].std()

        img = (x[:, channel, :, :] - min_value) / (max_value - min_value)
        img = torch.clip(img, 0, 1).unsqueeze(dim=1)
        imgs.append(img)

    return torch.cat(imgs, dim=1)


def prepare_input(input, model_name):
    if model_name.startswith("terramind_v1_"):
        means = torch.tensor(TERRAMIND_V1_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(TERRAMIND_V1_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
    elif model_name.startswith("prithvi_eo_v2_"):
        indices = [S2L2A_BANDS.index(band) for band in PRITHVI_V2_BANDS]
        input = input[:, indices, :, :]
        means = torch.tensor(PRITHVI_V2_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(PRITHVI_V2_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
    elif model_name.startswith("thor_v1_"):
        input = input / 10000.0
        means = torch.tensor(THOR_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(THOR_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
    elif model_name.startswith("dofa_"):
        indices = [S2L2A_BANDS.index(band) for band in DOFA_BANDS]
        input = input[:, indices, :, :]
        input = (input / 10000.0) * 255.0
        means = torch.tensor(DOFA_MEANS, device=input.device).view(1, -1, 1, 1)
        stds = torch.tensor(DOFA_STDS, device=input.device).view(1, -1, 1, 1)
        input = (input - means) / stds
    elif model_name == "croma_large":
        input = satmae_normalize(input)
    elif model_name == "pixel_stats":
        pass
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    input = input.to(DEVICE)
    return input

@torch.no_grad()
def get_latents(model, input, model_name):
    if model_name.startswith("croma_"):
        latents = model(optical_images=input)["optical_GAP"].cpu().numpy()
    elif model_name.startswith("thor_v1_"):
        latents = model(input)[-1].mean(dim=[2, 3]).cpu().numpy()
    elif model_name == "pixel_stats":
        latents = pixel_stats(input).cpu().numpy()
    else:
        latents = model(input)[-1].mean(dim=1).cpu().numpy()

    return latents


def get_dataloader(model_name):
    if model_name.startswith("thor_v1_"):
        crop_size = 240
        batch_size = 128
    elif model_name.startswith("croma_"):
        crop_size = 120
        batch_size = 512
    else:
        crop_size = 224
        batch_size = 512

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
        non_image_modalities=["__key__", "__url__", "center_lon", "center_lat", "time_S2L2A"],  # Additional non-image keys
    )

    dataset = build_terramesh_dataset(
        path=DATASET_PATH,
        modalities=["S2L2A", "DEM"],
        split="val",
        transform=val_transform,
        shuffle=False,
        return_metadata=True,
        batch_size=batch_size,
    )

    return DataLoader(dataset, batch_size=None, num_workers=16, persistent_workers=True, prefetch_factor=2)


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
        dataloader = get_dataloader(model_name)
        model = load_model(model_name)

        metadatas = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
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
                latents = get_latents(model, input, model_name)
                root[model_name].append(latents)

        per_model_metadata[model_name] = metadatas

    with open(META_PATH, "wb") as f:
        pickle.dump(per_model_metadata, f)

    counts = {model_name: len(per_model_metadata[model_name]) for model_name in MODELS}
    print("Sample counts per model:")
    for model_name, count in counts.items():
        print(f"{model_name}: {count}")

    print(f"Latents saved to {ZARR_PATH} and metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
