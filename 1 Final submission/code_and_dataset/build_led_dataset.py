import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def split_image_to_tiles(img_path, out_dir, tile_size=None, prefix=None):
    """
    Split an image into 8x8 tiles and save them.
    Tile indexing follows the user's ordering:
      - For columns: RIGHT to LEFT (col = 7 .. 0)
      - For rows inside a column: BOTTOM to TOP (row = 7 .. 0)
    So the flattened index order is: for c in 7..0: for r in 7..0: yield tile(r,c)
    This function returns a list of (tile_filename, (row, col), flat_index).
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # compute tile size if not provided (use integer division)
    tile_w = tile_h = tile_size if tile_size else (w // 8, h // 8)

    tiles_info = []
    base = Path(img_path).stem if prefix is None else prefix

    for col in range(7, -1, -1):           # right -> left
        for row in range(7, -1, -1):       # bottom -> top
            left = int(round(col * (w / 8.0)))
            upper = int(round(row * (h / 8.0)))
            # compute right/bottom using floats to cover remainder pixels in last tiles
            right = int(round((col + 1) * (w / 8.0)))
            lower = int(round((row + 1) * (h / 8.0)))

            tile = img.crop((left, upper, right, lower))

            flat_index = (7 - col) * 8 + (7 - row)  # same mapping formula
            filename = f"{base}_led_{flat_index}.png"
            out_path = Path(out_dir) / filename
            tile.save(out_path)

            tiles_info.append( (str(out_path), (row, col), flat_index) )

    return tiles_info

def remap_labels_for_image(original_label_list):
    """
    original_label_list: list of 64 ints in the user's format:
      order = columns scanned RIGHT->LEFT, each column BOTTOM->TOP
    This function will simply validate length and return as numpy array.
    If your labels were in another order and you need to convert FROM row-major
    to this order, you'd implement the inverse mapping here.
    """
    arr = np.array(original_label_list, dtype=int)
    if arr.size != 64:
        raise ValueError("Label list must contain 64 elements")
    return arr  # already in the desired ordering

def build_dataset(images_dir, labels_file, out_tiles_dir="led_dataset/photos", tile_size=None, write_tile_labels=True):
    """
    images_dir: folder with cropped images (file names match keys in labels_file)
    labels_file: JSON mapping image relative path -> list[64] (in user's ordering)
    out_tiles_dir: where tile images will be saved
    """
    ensure_dir(out_tiles_dir)

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    tile_labels = {}  # map tile filename -> label int
    image_to_tiles = {}  # map image_name -> list of tile filenames in order

    for img_key in sorted(labels.keys()):
        img_path = Path(images_dir) / img_key
        if not img_path.exists():
            print(f"Warning: image not found, skipping: {img_path}")
            continue

        orig_labels = labels[img_key]
        arr = remap_labels_for_image(orig_labels)  # verifies length

        tiles_info = split_image_to_tiles(str(img_path), out_tiles_dir, tile_size=tile_size, prefix=Path(img_key).stem)

        # tiles_info is in the SAME order as arr (by construction)
        if len(tiles_info) != 64:
            raise RuntimeError(f"Expected 64 tiles for {img_key}, got {len(tiles_info)}")

        image_tile_list = []
        for (_, _, flat_index), label in zip( ((t[0], t[1], t[2]) for t in tiles_info), arr ):
            tile_filename = Path(tiles_info[flat_index][0]).name if False else None
            # above line unnecessary; simpler:
            tile_path, _, idx = tiles_info[flat_index]
            tile_name = Path(tile_path).name
            tile_labels[tile_name] = int(label)
            image_tile_list.append(tile_name)

        image_to_tiles[img_key] = image_tile_list

    if write_tile_labels:
        tile_labels_path = Path(out_tiles_dir) / "tile_labels.json"
        with open(tile_labels_path, 'w') as f:
            json.dump(tile_labels, f, indent=2)
        print(f"Wrote tile labels: {tile_labels_path}")

    print("Done building tiles.")
    return image_to_tiles, tile_labels


# Comparison utilities
def load_flat_json_map(path):
    """Load mapping filename->list and flatten to arrays preserving key order"""
    with open(path, 'r') as f:
        data = json.load(f)
    keys = sorted(data.keys())
    flat = []
    for k in keys:
        flat.extend(data[k])
    return np.array(flat, dtype=int), keys

def compare_predictions(labels_json, predictions_json, print_mismatch=True):
    """
    Compare two JSONs of the form {filename: [64 ints,...], ...}
    Prints overall and per-class accuracy and optionally per-image mismatches.
    Also prints per-tile true/pred pairs for visual verification.
    """
    with open(labels_json, 'r') as f:
        labels = json.load(f)
    with open(predictions_json, 'r') as f:
        preds = json.load(f)

    # Ensure same keys
    common = sorted(set(labels.keys()) & set(preds.keys()))
    if len(common) == 0:
        raise ValueError("No overlapping keys between labels and predictions")

    total = 0
    correct = 0
    per_class_counts = {0:0, 1:0, 2:0}
    per_class_correct = {0:0, 1:0, 2:0}

    for key in common:
        true = np.array(labels[key], dtype=int)
        pred = np.array(preds[key], dtype=int)
        if true.shape != (64,) or pred.shape != (64,):
            print(f"Skipping {key}: expected 64-length lists (got {true.shape}, {pred.shape})")
            continue

        print(f"\nImage: {key}")
        for idx in range(64):
            t = int(true[idx]); p = int(pred[idx])
            total += 1
            per_class_counts[t] += 1
            if t == p:
                correct += 1
                per_class_correct[t] += 1
            # print per-tile to confirm visually
            print(f"  Tile {idx:02d}: true={t} pred={p} {'OK' if t==p else 'ERR'}")

    overall_acc = correct / total if total>0 else 0.0
    print(f"\nOverall accuracy: {overall_acc*100:.2f}% ({correct}/{total})")
    for c in sorted(per_class_counts.keys()):
        cnt = per_class_counts[c]
        corr = per_class_correct[c]
        acc = (corr / cnt * 100) if cnt>0 else 0.0
        name = {0:'Empty',1:'Red',2:'Blue'}.get(c,str(c))
        print(f"  {name}: {acc:.2f}% ({corr}/{cnt})")

# Use
if __name__ == "__main__":
    images_dir = "cropped_images"    # folder with cropped 8x8 boards
    labels_file = "labels.json"      # labels in column-major bottom->top right->left
    out_tiles_dir = "led_dataset/photos"

    ensure_dir(out_tiles_dir)
    image_to_tiles, tile_labels = build_dataset(images_dir, labels_file, out_tiles_dir, tile_size=None, write_tile_labels=True)

