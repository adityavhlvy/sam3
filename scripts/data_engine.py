# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import os
import json
import torch
import cv2
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_util

# Add project root to path to import sam3
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # geospatial-deliniation
sam3_root = os.path.join(project_root, "sam3")
sys.path.append(sam3_root)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def setup_model(checkpoint_path, device="cuda"):
    print(f"Loading model from {checkpoint_path}...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        enable_inst_interactivity=True
    )
    processor = Sam3Processor(model)
    return processor

def get_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Draws semitransparent mask on image.
    image: numpy array (H, W, 3) BGR
    mask: numpy array (H, W) bool
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def main():
    parser = argparse.ArgumentParser(description="SAM3 Data Engine: Phase 1 (Lite)")
    parser.add_argument("--image_dir", required=True, help="Folder containing input images (jpg, png, tif)")
    parser.add_argument("--output_dir", required=True, help="Folder to save JSON and verification overlays")
    parser.add_argument("--prompt", required=True, help="Text prompt (Concept) to search for")
    parser.add_argument("--checkpoint", default="../sam3-inference/sam3.pt", help="Path to SAM3 checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    overlay_dir = os.path.join(args.output_dir, "verification_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # 1. Load Model
    processor = setup_model(args.checkpoint, args.device)
    processor.set_confidence_threshold(args.conf)

    # 2. Find Images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True))
    
    print(f"Found {len(image_files)} images in {args.image_dir}")

    # 3. Process
    images_json = []
    annotations_json = []
    ann_id_counter = 1

    for i, img_path in enumerate(tqdm(image_files)):
        # Load Image
        try:
            # Handle TIFF -> RGB for model
            cv_img = cv2.imread(img_path)
            if cv_img is None: 
                print(f"Skipping corrupt image: {img_path}")
                continue
            
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            h, w = cv_img.shape[:2]
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        # Run Inference
        state = processor.set_image(pil_img)
        output = processor.set_text_prompt(args.prompt, state)
        
        masks = output.get("masks", [])
        scores = output.get("scores", [])
        
        # Convert pytorch tensors to numpy
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        # If has detections
        has_detection = len(masks) > 0
        
        # Overlay Generation
        vis_img = cv_img.copy()
        for mask in masks:
            vis_img = get_overlay(vis_img, mask)
        
        # Save overlay for user verification
        base_name = os.path.basename(img_path)
        overlay_path = os.path.join(overlay_dir, f"overlay_{base_name}.jpg") # Save as jpg for viewability
        cv2.imwrite(overlay_path, vis_img)

        # Append to JSON
        # SA-Co Image Entry
        image_id = i + 1
        images_json.append({
            "id": image_id,
            "file_name": os.path.abspath(img_path), # Absolute path for safety
            "text_input": args.prompt,
            "width": w,
            "height": h,
            "queried_category": 1,
            "is_instance_exhaustive": 1, 
            "is_pixel_exhaustive": 1
        })

        if has_detection:
            for mask in masks:
                # Convert mask to RLE
                # binary mask (H, W) -> uint8 -> fortran order
                mask_uint8 = np.asfortranarray(mask.astype(np.uint8))
                rle = mask_util.encode(mask_uint8)
                rle['counts'] = rle['counts'].decode('utf-8') # JSON serializable

                # Bbox
                bbox = mask_util.toBbox(rle).tolist() # [x, y, w, h]
                # Normalize bbox
                norm_bbox = [
                    bbox[0] / w,
                    bbox[1] / h,
                    bbox[2] / w,
                    bbox[3] / h
                ]
                area = float(mask_util.area(rle))

                annotations_json.append({
                    "id": ann_id_counter,
                    "image_id": image_id,
                    "segmentation": rle,
                    "bbox": norm_bbox,
                    "area": area,
                    "category_id": 1,
                    "iscrowd": 0,
                    "source": "sam3_auto"
                })
                ann_id_counter += 1

    # 4. Save Final JSON
    dataset = {
        "images": images_json,
        "annotations": annotations_json,
        "categories": [{"id": 1, "name": args.prompt}]
    }

    out_json_path = os.path.join(args.output_dir, "_annotations.coco.json")
    with open(out_json_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDone! Processed {len(images_json)} images.")
    print(f"Generated {len(annotations_json)} masks.")
    print(f"1. Check overlays in: {overlay_dir}")
    print(f"2. Validated dataset JSON: {out_json_path}")

if __name__ == "__main__":
    main()
