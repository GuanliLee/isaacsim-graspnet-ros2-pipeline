#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import SAM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument(
        "--weights",
        default="weights/sam_b.pt",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(args.image).convert("RGB"))
    model = SAM(args.weights)
    result = model.predict(
        args.image,
        bboxes=[args.bbox],
        verbose=False,
        retina_masks=True,
    )[0]
    if result.masks is None or len(result.masks.data) == 0:
        raise RuntimeError("SAM returned no mask for the supplied bbox")

    masks = result.masks.data.cpu().numpy().astype(np.float32)
    mask = cv2.resize(
        masks[0], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    ) > 0.5
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    Image.fromarray(mask_u8).save(out_dir / "sam_mask_selected_raw.png")
    dilated = cv2.dilate(mask_u8, np.ones((5, 5), dtype=np.uint8), iterations=1)
    Image.fromarray(dilated).save(
        out_dir / "sam_mask_selected_dilated_for_graspnet.png"
    )

    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay = np.where(
        mask_u8[..., None] > 0,
        (0.58 * overlay + 0.42 * red).astype(np.uint8),
        overlay,
    )
    annotated = Image.fromarray(overlay)
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = args.bbox
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
    draw.text((x1, max(0, y1 - 15)), "AD Calcium Milk | SAM", fill=(0, 255, 0))
    annotated.save(out_dir / "sam_selected_overlay.png")

    ys, xs = np.nonzero(mask_u8)
    summary = {
        "model": "SAM",
        "weights": args.weights,
        "bbox_xyxy": [float(value) for value in args.bbox],
        "mask_pixels": int(mask.sum()),
        "mask_bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ],
    }
    with (out_dir / "sam_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
