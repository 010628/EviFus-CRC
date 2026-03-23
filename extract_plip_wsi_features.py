#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pathology foundation-model feature extraction for EviFus-CRC.

Workflow:
1. identify tissue-rich regions on the WSI,
2. extract image patches,
3. encode patches with PLIP,
4. mean-pool patch embeddings into a patient-/slide-level feature vector.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
import openslide
from transformers import AutoModel, AutoProcessor

VALID_WSI_SUFFIXES={".svs",".tif",".tiff",".ndpi",".mrxs"}

def read_manifest(input_path: str)->List[Dict[str,str]]:
    path=Path(input_path)
    if path.suffix.lower() in VALID_WSI_SUFFIXES:
        return [{"patient_id":path.stem,"slide_path":str(path)}]
    if path.suffix.lower()==".csv":
        df=pd.read_csv(path)
        return df[["patient_id","slide_path"]].to_dict(orient="records")
    raise ValueError("input-path must be a WSI file or a CSV manifest.")

def load_plip(model_name: str, device: torch.device):
    processor=AutoProcessor.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    return processor, model

def build_thumbnail_tissue_mask(slide: openslide.OpenSlide, max_side: int = 2048, background_intensity_threshold: int = 220, min_rgb_std: float = 8.0):
    width,height=slide.dimensions
    scale=max(width,height)/max_side if max(width,height)>max_side else 1.0
    thumb_w=max(1,int(width/scale)); thumb_h=max(1,int(height/scale))
    thumbnail=slide.get_thumbnail((thumb_w,thumb_h)).convert("RGB")
    thumb_np=np.asarray(thumbnail,dtype=np.uint8)
    intensity=thumb_np.mean(axis=2); rgb_std=thumb_np.std(axis=2)
    tissue_mask=((intensity<background_intensity_threshold)&(rgb_std>min_rgb_std)).astype(np.uint8)
    return tissue_mask,(thumb_w,thumb_h)

def generate_candidate_coords(slide, tissue_mask, thumb_size, patch_size, stride, read_level, min_tissue_ratio: float = 0.40):
    """Generate candidate coordinates from a thumbnail tissue mask."""
    width_lvl0,height_lvl0=slide.dimensions; thumb_w,thumb_h=thumb_size
    scale_x=width_lvl0/thumb_w; scale_y=height_lvl0/thumb_h
    downsample=slide.level_downsamples[read_level]
    patch_size_lvl0=int(patch_size*downsample); stride_lvl0=int(stride*downsample)
    coords=[]
    for y in range(0,max(height_lvl0-patch_size_lvl0+1,1), stride_lvl0):
        for x in range(0,max(width_lvl0-patch_size_lvl0+1,1), stride_lvl0):
            tx0=int(x/scale_x); ty0=int(y/scale_y); tx1=int((x+patch_size_lvl0)/scale_x); ty1=int((y+patch_size_lvl0)/scale_y)
            tx1=min(tx1,thumb_w); ty1=min(ty1,thumb_h)
            if tx1<=tx0 or ty1<=ty0: continue
            region_mask=tissue_mask[ty0:ty1,tx0:tx1]
            if region_mask.size==0: continue
            if float(region_mask.mean())>=min_tissue_ratio:
                coords.append((x,y))
    return coords

def patch_quality_filter(patch: Image.Image, white_threshold: int = 220, min_tissue_ratio: float = 0.50)->bool:
    arr=np.asarray(patch.convert("RGB"),dtype=np.uint8)
    intensity=arr.mean(axis=2); tissue_ratio=float((intensity<white_threshold).mean())
    return tissue_ratio>=min_tissue_ratio

@torch.inference_mode()
def extract_patch_features(patches: Sequence[Image.Image], processor, model, device):
    inputs=processor(images=list(patches), return_tensors="pt")
    inputs={k:v.to(device) for k,v in inputs.items()}
    outputs=model(**inputs)
    if hasattr(outputs,"image_embeds") and outputs.image_embeds is not None:
        feats=outputs.image_embeds
    elif hasattr(model,"get_image_features"):
        feats=model.get_image_features(**inputs)
    else:
        raise RuntimeError("Cannot find image embedding output from PLIP model.")
    return feats.detach().cpu().numpy().astype(np.float32)

def parse_args():
    parser=argparse.ArgumentParser(description="Extract PLIP features from H&E whole-slide images.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="vinid/plip")
    parser.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    parser.add_argument("--read-level", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--thumbnail-max-side", type=int, default=2048)
    parser.add_argument("--max-patches-per-slide", type=int, default=2000)
    parser.add_argument("--save-patch-features", action="store_true")
    parser.add_argument("--save-patch-coords", action="store_true")
    return parser.parse_args()

def main():
    args=parse_args()
    output_dir=Path(args.output_dir).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    slide_feature_dir=output_dir/'slide_features'; slide_feature_dir.mkdir(exist_ok=True)
    patch_feature_dir=output_dir/'patch_features'; patch_coord_dir=output_dir/'patch_coords'
    if args.save_patch_features: patch_feature_dir.mkdir(exist_ok=True)
    if args.save_patch_coords: patch_coord_dir.mkdir(exist_ok=True)
    device=torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    processor,model=load_plip(args.model_name, device)
    records=read_manifest(args.input_path)
    failed=[]
    for record in tqdm(records, desc="Extracting PLIP features"):
        patient_id=str(record["patient_id"]); slide_path=str(record["slide_path"])
        try:
            slide=openslide.OpenSlide(slide_path)
            tissue_mask,thumb_size=build_thumbnail_tissue_mask(slide,max_side=args.thumbnail_max_side)
            coords=generate_candidate_coords(slide,tissue_mask,thumb_size,args.patch_size,args.stride,args.read_level)
            if len(coords)==0: raise RuntimeError("No valid tissue-rich patches were identified.")
            if len(coords)>args.max_patches_per_slide: coords=coords[:args.max_patches_per_slide]
            all_patch_features=[]; kept_coords=[]; patch_buffer=[]; coord_buffer=[]
            for x,y in coords:
                patch=slide.read_region((x,y), args.read_level, (args.patch_size,args.patch_size)).convert("RGB")
                if not patch_quality_filter(patch): continue
                patch_buffer.append(patch); coord_buffer.append((x,y))
                if len(patch_buffer)==args.batch_size:
                    feats=extract_patch_features(patch_buffer,processor,model,device)
                    all_patch_features.append(feats); kept_coords.extend(coord_buffer); patch_buffer=[]; coord_buffer=[]
            if patch_buffer:
                feats=extract_patch_features(patch_buffer,processor,model,device)
                all_patch_features.append(feats); kept_coords.extend(coord_buffer)
            if len(all_patch_features)==0: raise RuntimeError("All candidate patches were filtered out after quality control.")
            patch_features=np.concatenate(all_patch_features,axis=0).astype(np.float32)
            # The public release uses the mean pooled slide-level embedding as
            # the default pathology representation for downstream fusion.
            slide_feature=patch_features.mean(axis=0).astype(np.float32)
            np.save(slide_feature_dir/f"{patient_id}.npy", slide_feature)
            if args.save_patch_features: np.save(patch_feature_dir/f"{patient_id}.npy", patch_features)
            if args.save_patch_coords: pd.DataFrame(kept_coords, columns=["x_level0","y_level0"]).to_csv(patch_coord_dir/f"{patient_id}.csv", index=False)
            slide.close()
        except Exception as e:
            failed.append({"patient_id":patient_id,"slide_path":slide_path,"error":str(e)})
    with open(output_dir/'metadata.json','w',encoding='utf-8') as f:
        json.dump({"model_name":args.model_name,"input_path":args.input_path,"num_cases":len(records)}, f, indent=2, ensure_ascii=False)
    if failed:
        pd.DataFrame(failed).to_csv(output_dir/'failed_cases.csv', index=False)

if __name__=="__main__":
    main()
