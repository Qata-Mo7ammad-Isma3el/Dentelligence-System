# save as convert_to_coco.py
import os, json, argparse
from collections import defaultdict
from PIL import Image

def corners_to_xyxy(corners):
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2, y2

def xyxy_to_xywh(x1, y1, x2, y2):
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]

def read_custom(path):
    with open(path, "r") as f:
        data = json.load(f)
    out = []
    for b in data.get("boxes", []):
        name = str(b.get("name", "")).strip()
        corners = b.get("corners", [])
        if len(corners) != 4:
            continue
        x1, y1, x2, y2 = corners_to_xyxy(corners)
        # image_id is stored in the 3rd value of each corner triplet
        img_ids = [int(c[2]) for c in corners if len(c) > 2]
        image_id = img_ids[0] if img_ids else 0
        score = float(b.get("probability", 1.0))
        out.append({
            "image_id": image_id,
            "label": name,
            "bbox_xyxy": [x1, y1, x2, y2],
            "score": score
        })
    return out

def collect_images_info(all_items, image_dir=None, filename_pattern="{id}.png"):
    """
    Build COCO 'images' from available image_ids.
    If image_dir is provided, we try to read width/height. Otherwise we write 0/0.
    """
    image_ids = sorted(set([it["image_id"] for it in all_items]))
    images = []
    for iid in image_ids:
        file_name = filename_pattern.format(id=iid)
        width = height = 0
        if image_dir:
            # Try to find the file; if filename_pattern is used, look there first
            candidate = os.path.join(image_dir, file_name)
            if not os.path.exists(candidate):
                # fallback: search by id prefix across common extensions
                exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
                found = None
                for root, _, files in os.walk(image_dir):
                    for fn in files:
                        if fn.startswith(str(iid)) and os.path.splitext(fn)[1].lower() in exts:
                            found = os.path.join(root, fn); break
                    if found: break
                if found:
                    candidate = found
                else:
                    candidate = None
            if candidate and os.path.exists(candidate):
                try:
                    with Image.open(candidate) as im:
                        width, height = im.size
                    file_name = os.path.relpath(candidate, image_dir)
                except Exception:
                    pass
        images.append({"id": iid, "file_name": file_name, "width": width, "height": height})
    return images

def build_categories(gt_items, pred_items, use_full_label=True, reduce_label=None):
    """
    Categories are made from the union of labels in GT and predictions.
    - use_full_label=True keeps the full "3 - 7 - 0" as a class.
    - If you want to collapse to just the last token (e.g., disease id), set reduce_label="last".
    """
    labels = set()
    for it in gt_items + pred_items:
        lab = it["label"]
        if reduce_label == "last":
            lab = lab.split("-")[-1].strip()
        labels.add(lab)
    labels = sorted(labels)
    cat_id_of = {lab: i + 1 for i, lab in enumerate(labels)}
    categories = [{"id": cid, "name": lab} for lab, cid in cat_id_of.items()]
    return categories, cat_id_of

def convert_gt(gt_items, categories_map, reduce_label=None):
    annotations = []
    ann_id = 1
    for it in gt_items:
        x1, y1, x2, y2 = it["bbox_xyxy"]
        bbox = xyxy_to_xywh(x1, y1, x2, y2)
        area = bbox[2] * bbox[3]
        lab = it["label"]
        if reduce_label == "last":
            lab = lab.split("-")[-1].strip()
        cat_id = categories_map[lab]
        annotations.append({
            "id": ann_id,
            "image_id": it["image_id"],
            "category_id": cat_id,
            "bbox": [float(v) for v in bbox],
            "area": float(area),
            "iscrowd": 0
        })
        ann_id += 1
    return annotations

def convert_preds(pred_items, categories_map, reduce_label=None):
    """
    COCO detections format: list[ {image_id, category_id, bbox[x,y,w,h], score} ]
    """
    dets = []
    for it in pred_items:
        x1, y1, x2, y2 = it["bbox_xyxy"]
        bbox = xyxy_to_xywh(x1, y1, x2, y2)
        lab = it["label"]
        if reduce_label == "last":
            lab = lab.split("-")[-1].strip()
        # Some preds might have labels not in GT; keep them anyway so loadRes works.
        if lab not in categories_map:
            # create a dummy new id AFTER the fact (not ideal for strict COCO, but ok for eval)
            categories_map[lab] = max(categories_map.values(), default=0) + 1
        dets.append({
            "image_id": it["image_id"],
            "category_id": categories_map[lab],
            "bbox": [float(v) for v in bbox],
            "score": float(it["score"])
        })
    return dets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Path to GT.json (custom format)")
    ap.add_argument("--pred", required=True, help="Path to predictions.json (custom format)")
    ap.add_argument("--out_dir", required=True, help="Where to save COCO files")
    ap.add_argument("--image_dir", default=None, help="Optional: directory of images to fill width/height")
    ap.add_argument("--filename_pattern", default="{id}.png", help="Filename pattern if using image_dir")
    ap.add_argument("--label_mode", choices=["full","last"], default="full",
                    help="Use full 'name' as class or only last token (e.g., disease id)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt_items   = read_custom(args.gt)
    pred_items = read_custom(args.pred)

    # Build categories
    reduce_label = "last" if args.label_mode == "last" else None
    categories, cat_map = build_categories(gt_items, pred_items, reduce_label=reduce_label)

    # Images (from union of image_ids)
    images = collect_images_info(gt_items + pred_items, image_dir=args.image_dir,
                                 filename_pattern=args.filename_pattern)

    # Annotations & Detections
    annotations = convert_gt(gt_items, cat_map, reduce_label=reduce_label)
    detections  = convert_preds(pred_items, cat_map, reduce_label=reduce_label)

    # COCO GT
    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    gt_out = os.path.join(args.out_dir, "instances_gt.json")
    with open(gt_out, "w") as f:
        json.dump(coco_gt, f, indent=2)

    # COCO detections
    det_out = os.path.join(args.out_dir, "detections_pred.json")
    with open(det_out, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"Saved:\n  {gt_out}\n  {det_out}")
    print("\nExample evaluation (pycocotools):")
    print(f"""\
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt = COCO(r"{gt_out}")
dt = gt.loadRes(r"{det_out}")
e = COCOeval(gt, dt, iouType='bbox')
e.evaluate(); e.accumulate(); e.summarize()
""")

if __name__ == "__main__":
    main()
