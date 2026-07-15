#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOE
from ultralytics.utils import SETTINGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="work/capture/color.png")
    parser.add_argument("--weights", default="weights/yoloe-26l-seg.pt")
    parser.add_argument("--out-dir", default="outputs/latest")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--target-mode", choices=["yellow", "ad_milk", "white_ad_milk"], default="yellow")
    parser.add_argument("--draw-candidates", action="store_true")
    parser.add_argument("--depth", default="work/capture/depth.png")
    parser.add_argument("--depth-tolerance-m", type=float, default=0.08)
    parser.add_argument("--disable-depth-filter", action="store_true")
    parser.add_argument(
        "--select-strategy",
        choices=["nearest", "score", "leftmost", "first_row_arm"],
        default="nearest",
        help=(
            "nearest selects the closest candidate in the wrist depth image; "
            "score uses the detector ranking; leftmost selects the outer-left "
            "shelf candidate; first_row_arm keeps complete white first-row "
            "bottles and selects the one nearest the gripper image axis."
        ),
    )
    return parser.parse_args()


def yellow_ratio(rgb_crop, mask_crop):
    if rgb_crop.size == 0 or mask_crop.size == 0 or not np.any(mask_crop):
        return 0.0
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 45) & (hsv[..., 1] > 35) & (hsv[..., 2] > 60)
    selected = yellow & mask_crop
    return float(selected.sum() / max(1, mask_crop.sum()))


def white_ratio(rgb_crop, mask_crop):
    if rgb_crop.size == 0 or mask_crop.size == 0 or not np.any(mask_crop):
        return 0.0
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    white = (hsv[..., 1] < 80) & (hsv[..., 2] > 120)
    selected = white & mask_crop
    return float(selected.sum() / max(1, mask_crop.sum()))


def clamp_bbox(xyxy, width, height):
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def keep_best_connected_component(mask, xyxy):
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(xyxy, w, h)
    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[y1 : y2 + 1, x1 : x2 + 1] = mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    if num <= 1:
        return clipped.astype(bool)

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bbox_area = max(1.0, float((x2 - x1 + 1) * (y2 - y1 + 1)))
    best_label = None
    best_score = -1e18
    for label in range(1, num):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        ccx, ccy = centroids[label]
        dist = ((ccx - cx) / max(1.0, x2 - x1 + 1)) ** 2 + ((ccy - cy) / max(1.0, y2 - y1 + 1)) ** 2
        area_ratio = min(1.0, area / bbox_area)
        score = area_ratio - 0.6 * dist
        if score > best_score:
            best_label = label
            best_score = score
    if best_label is None:
        return clipped.astype(bool)
    return labels == best_label


def filter_mask_by_depth(mask, depth_path, tolerance_m):
    path = Path(depth_path)
    if not path.exists() or not np.any(mask):
        return mask
    depth = np.array(Image.open(path))
    if depth.shape[:2] != mask.shape[:2]:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth_m = depth.astype(np.float32)
    if np.nanmax(depth_m) > 20.0:
        depth_m /= 1000.0
    values = depth_m[(mask > 0) & np.isfinite(depth_m) & (depth_m > 0)]
    if values.size < 50:
        return mask
    target_depth = float(np.median(values))
    filtered = mask & (np.abs(depth_m - target_depth) <= tolerance_m)
    if int(filtered.sum()) < max(80, int(mask.sum() * 0.35)):
        return mask
    return filtered


def load_depth_m(depth_path, shape):
    path = Path(depth_path)
    if not path.exists():
        return None
    depth = np.array(Image.open(path))
    if depth.shape[:2] != shape[:2]:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    depth_m = depth.astype(np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if np.any(valid) and np.nanmax(depth_m[valid]) > 20.0:
        depth_m /= 1000.0
    return depth_m


def candidate_depth_stats(mask, xyxy, depth_m):
    if depth_m is None:
        return None, None
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(xyxy, w, h)
    local_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    local_depth = depth_m[y1 : y2 + 1, x1 : x2 + 1]
    values = local_depth[(local_mask > 0) & np.isfinite(local_depth) & (local_depth > 0)]
    if values.size < 20:
        values = local_depth[np.isfinite(local_depth) & (local_depth > 0)]
    if values.size == 0:
        return None, 0
    return float(np.median(values)), int(values.size)


def clean_selected_mask(mask, xyxy, depth_path=None, depth_tolerance_m=0.08, use_depth=True):
    cleaned = keep_best_connected_component(mask, xyxy)
    if use_depth and depth_path:
        cleaned = filter_mask_by_depth(cleaned, depth_path, depth_tolerance_m)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_u8 = cleaned.astype(np.uint8) * 255
    cleaned_u8 = cv2.morphologyEx(cleaned_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned_u8 = cv2.morphologyEx(cleaned_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned_u8 > 0


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(args.image).convert("RGB"))
    h, w = image.shape[:2]
    depth_m = load_depth_m(args.depth, image.shape)

    weights_dir = Path(args.weights).resolve().parent
    text_encoder = weights_dir / "mobileclip2_b.ts"
    if text_encoder.exists():
        SETTINGS.update({"weights_dir": str(weights_dir)})

    model = YOLOE(args.weights)
    if args.target_mode == "white_ad_milk":
        classes = ["AD calcium milk bottle"]
    elif args.target_mode == "ad_milk":
        classes = ["AD calcium milk bottle"]
    else:
        classes = ["bottle", "coffee bottle", "yellow bottle", "drink bottle", "milk tea bottle"]
    model.set_classes(classes)
    results = model.predict(args.image, conf=args.conf, verbose=False)
    result = results[0]

    boxes = result.boxes
    masks = result.masks
    detections = []
    if boxes is not None and masks is not None:
        mask_data = masks.data.cpu().numpy()
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(float)
            conf = float(box.conf[0].cpu().item())
            cls_idx = int(box.cls[0].cpu().item())
            mask = cv2.resize(mask_data[idx], (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            crop = image[y1 : y2 + 1, x1 : x2 + 1]
            mask_crop = mask[y1 : y2 + 1, x1 : x2 + 1]
            yr = yellow_ratio(crop, mask_crop)
            wr = white_ratio(crop, mask_crop)
            median_depth_m, depth_pixels = candidate_depth_stats(mask, xyxy, depth_m)
            cy = (y1 + y2) * 0.5 / h
            cx = (x1 + x2) * 0.5 / w
            area = float((x2 - x1 + 1) * (y2 - y1 + 1))
            if args.target_mode == "white_ad_milk":
                # Close left-wrist view: AD calcium milk is a mostly white
                # bottle on the shelf. Keep only plausible full/near-full
                # shelf bottles so generic bottle-shaped false positives do
                # not enter the downstream SAM/GraspNet mask.
                on_shelf = 0.18 <= cy <= 0.95 and 0.02 <= cx <= 0.95
                size_ok = 900.0 <= area <= 30000.0
                tool_region = cy >= 0.62 and 0.35 <= cx <= 0.65
                if not on_shelf or tool_region or not size_ok or wr < 0.10:
                    continue
                score = conf + 3.0 * wr + 1.0
            elif args.target_mode == "ad_milk":
                # Left wrist view sees AD calcium milk as small bottles on the
                # left/middle shelf. Avoid the shopping cart false positive.
                in_ad_roi = 0.28 <= cx <= 0.47 and 0.24 <= cy <= 0.42
                size_ok = 90.0 <= area <= 380.0
                lower_shelf_bonus = 0.9 if 0.33 <= cy <= 0.42 else 0.0
                score = conf + (3.0 if in_ad_roi else -2.0) + (1.0 if size_ok else -1.0) + lower_shelf_bonus
            else:
                shelf_bonus = 0.9 if 0.10 <= cy <= 0.42 else 0.0
                score = conf + 3.0 * yr + shelf_bonus
            detections.append(
                {
                    "index": idx,
                    "class": classes[cls_idx] if cls_idx < len(classes) else str(cls_idx),
                    "confidence": conf,
                    "bbox_xyxy": [float(v) for v in xyxy],
                    "yellow_ratio": yr,
                    "white_ratio": wr,
                    "center_norm": [float(cx), float(cy)],
                    "area_px": area,
                    "median_depth_m": median_depth_m,
                    "depth_pixels": depth_pixels,
                    "selection_score": score,
                    "mask": mask,
                }
            )

    if not detections:
        raise RuntimeError("YOLOE found no segmented object")

    if args.select_strategy == "first_row_arm":
        plausible = [
            det
            for det in detections
            if det.get("median_depth_m") is not None
            and det["white_ratio"] >= 0.40
            and 0.15 <= det["center_norm"][0] <= 0.60
            and 0.30 <= det["center_norm"][1] <= 0.65
            and det["bbox_xyxy"][0] > 2
            and det["bbox_xyxy"][2] < w - 2
            and det["bbox_xyxy"][3] < h - 2
        ]
        if not plausible:
            raise RuntimeError("no complete white AD milk candidate in the first-row ROI")
        nearest_depth = min(det["median_depth_m"] for det in plausible)
        front_row = [
            det
            for det in plausible
            if det["median_depth_m"] <= nearest_depth + 0.06
        ]
        front_row.sort(
            key=lambda det: (
                abs(det["center_norm"][0] - 0.50),
                det["median_depth_m"],
                -det["selection_score"],
            )
        )
        selected_ids = {id(det) for det in front_row}
        detections.sort(
            key=lambda det: (
                0 if id(det) in selected_ids else 1,
                abs(det["center_norm"][0] - 0.50),
                float("inf") if det.get("median_depth_m") is None else det["median_depth_m"],
            )
        )
    elif args.select_strategy == "nearest":
        detections.sort(
            key=lambda d: (
                float("inf") if d.get("median_depth_m") is None else d["median_depth_m"],
                -d["selection_score"],
            )
        )
    elif args.select_strategy == "leftmost":
        detections.sort(
            key=lambda d: (
                d["center_norm"][0],
                float("inf") if d.get("median_depth_m") is None else d["median_depth_m"],
                -d["selection_score"],
            )
        )
    else:
        detections.sort(key=lambda d: d["selection_score"], reverse=True)
    selected = detections[0]
    selected["raw_mask_pixels_before_cleanup"] = int(selected["mask"].sum())
    cleaned_mask = clean_selected_mask(
        selected["mask"],
        selected["bbox_xyxy"],
        depth_path=args.depth,
        depth_tolerance_m=args.depth_tolerance_m,
        use_depth=not args.disable_depth_filter,
    )
    selected["raw_mask_pixels_after_cleanup"] = int(cleaned_mask.sum())
    raw_mask = cleaned_mask.astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(raw_mask, kernel, iterations=1)

    Image.fromarray(raw_mask).save(out_dir / "sam_mask_selected_raw.png")
    Image.fromarray(dilated).save(out_dir / "sam_mask_selected_dilated_for_graspnet.png")
    Image.fromarray(raw_mask).save(out_dir / "sam_mask_from_yoloe.png")

    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay = np.where(raw_mask[..., None] > 0, (0.55 * overlay + 0.45 * red).astype(np.uint8), overlay)
    Image.fromarray(overlay).save(out_dir / "sam_selected_overlay.png")

    draw_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(draw_img)
    for rank, det in enumerate(detections):
        if not args.draw_candidates and det is not selected:
            continue
        x1, y1, x2, y2 = det["bbox_xyxy"]
        color = (0, 255, 0) if det is selected else (255, 80, 40)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2 if det is selected else 1)
        depth_label = "nan" if det.get("median_depth_m") is None else f"{det['median_depth_m']:.3f}m"
        label = f"{rank}:{det['class']} {det['confidence']:.2f} d={depth_label} w={det['white_ratio']:.2f}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    draw_img.save(out_dir / "yoloe_detection_selected.png")
    draw_img.save(out_dir / "yoloe_detection.png")

    summary = {
        "classes": classes,
        "selection_strategy": args.select_strategy,
        "detections": [
            {k: v for k, v in det.items() if k != "mask"} | {"rank": rank}
            for rank, det in enumerate(detections)
        ],
        "selected": {k: v for k, v in selected.items() if k != "mask"} | {"rank": 0},
        "selected_raw_mask_pixels": int((raw_mask > 0).sum()),
        "selected_graspnet_mask_pixels": int((dilated > 0).sum()),
    }
    with open(out_dir / "yoloe_sam_selected_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["selected"], indent=2))


if __name__ == "__main__":
    main()
