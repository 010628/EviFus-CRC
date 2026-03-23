#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CT foundation-model feature extraction for EviFus-CRC.

This script is intentionally independent from training:
it produces patient-level CT embeddings that can later be referenced from the CSV.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from lighter_zoo import SegResEncoder
from monai.transforms import Compose, LoadImage, EnsureType, Orientation, ScaleIntensityRange, CropForeground, Spacing

def build_preprocess():
    """Preprocessing consistent with the manuscript CT pipeline."""
    return Compose([
        LoadImage(ensure_channel_first=True),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.0,1.0,1.0), mode="trilinear"),
        ScaleIntensityRange(a_min=-1024, a_max=3071, b_min=0.0, b_max=1.0, clip=True),
        CropForeground(method="otsu"),
    ])

def load_model(model_dir: str, device: torch.device):
    model = SegResEncoder.from_pretrained(model_dir, local_files_only=True)
    model.eval().to(device)
    return model

@torch.inference_mode()
def extract_single_feature(image_path: str, model, preprocess, device: torch.device) -> Dict[str, np.ndarray]:
    image_tensor = preprocess(image_path)
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.as_tensor(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    outputs = model(image_tensor)
    final_feature_map = outputs[-1]
    pooled_feature = torch.nn.functional.adaptive_avg_pool3d(final_feature_map, 1).flatten(1)
    return {
        "pooled_feature": pooled_feature.squeeze(0).cpu().numpy().astype(np.float32),
        "feature_map": final_feature_map.squeeze(0).cpu().numpy().astype(np.float32),
    }

def read_manifest(input_path: str) -> List[Dict[str, str]]:
    path = Path(input_path)
    if path.suffix.lower() in {".nii", ".gz"}:
        patient_id = path.name.replace(".nii.gz", "").replace(".nii", "")
        return [{"patient_id": patient_id, "image_path": str(path)}]
    if path.suffix.lower()==".csv":
        df = pd.read_csv(path)
        return df[["patient_id","image_path"]].to_dict(orient="records")
    raise ValueError("input-path must be a .nii/.nii.gz file or a CSV manifest.")

def parse_args():
    parser=argparse.ArgumentParser(description="Extract CT-FM features from tumour-centred 3D CT volumes.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    parser.add_argument("--save-feature-map", action="store_true")
    return parser.parse_args()

def main():
    args=parse_args()
    output_dir=Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save pooled patient-level embeddings; these are the default inputs for dataset.py.
    pooled_dir=output_dir/'pooled_features'; pooled_dir.mkdir(exist_ok=True)
    fmap_dir=output_dir/'feature_maps'
    if args.save_feature_map: fmap_dir.mkdir(exist_ok=True)
    device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    preprocess=build_preprocess(); model=load_model(args.model_dir, device)
    records=read_manifest(args.input_path)
    failed=[]
    for record in tqdm(records, desc="Extracting CT features"):
        try:
            feats=extract_single_feature(record["image_path"], model, preprocess, device)
            np.save(pooled_dir/f'{record["patient_id"]}.npy', feats["pooled_feature"])
            if args.save_feature_map:
                np.save(fmap_dir/f'{record["patient_id"]}.npy', feats["feature_map"])
        except Exception as e:
            failed.append({"patient_id":record["patient_id"],"image_path":record["image_path"],"error":str(e)})
    with open(output_dir/'metadata.json','w',encoding='utf-8') as f:
        json.dump({"model_dir":args.model_dir,"input_path":args.input_path,"num_cases":len(records),"save_feature_map":args.save_feature_map}, f, indent=2, ensure_ascii=False)
    if failed:
        pd.DataFrame(failed).to_csv(output_dir/'failed_cases.csv', index=False)

if __name__=="__main__":
    main()
