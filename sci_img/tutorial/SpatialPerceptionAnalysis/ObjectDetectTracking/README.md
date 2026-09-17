# 02 — Object Detection & Tracking

## Overview

Biological experiments increasingly involve imaging cells, particles, or organelles over
time to study dynamic processes: how a cell migrates in response to a chemical gradient,
how a virus particle is transported along a microtubule, how sister chromatids separate
during mitosis. Answering these questions requires two capabilities that this module builds
from the ground up: **detecting objects reliably in each frame**, and **linking those
detections across frames into coherent tracks**.

By the end you will have trained a YOLOv5 detector on a custom microscopy dataset,
evaluated it quantitatively, and built a full 3-D tracking pipeline using the LAP
algorithm on volumetric time-lapse data.

---

## What you will learn

### The object detection problem

Object detection asks: given an image, output a list of bounding boxes, each paired with a
class label and a confidence score. This is harder than classification (one label for the
whole image) because the model must simultaneously decide *what* objects are present and
*where* they are.

The two main architectural families:

**Two-stage detectors (R-CNN family):** first propose candidate regions (region proposal
network), then classify each region. High accuracy, slower inference. Example: Mask R-CNN.

**Single-stage detectors (YOLO family):** divide the image into a grid and predict boxes +
classes simultaneously from each cell. Faster, slightly lower accuracy on small objects.
YOLOv5 is the single-stage detector used in this module.

### YOLOv5 architecture

YOLOv5 has three major components:

**Backbone (CSPDarknet53):** a convolutional feature extractor that produces feature maps
at three scales (P3, P4, P5 — corresponding to stride 8, 16, 32 pixels). Deeper levels
capture semantic content (what); shallower levels retain spatial detail (where). The Cross
Stage Partial (CSP) connections split the feature map at each block to reduce computation
while preserving gradient flow.

**Neck (PANet):** fuses features across the three scales bidirectionally. Bottom-up path:
fine spatial detail propagates upward. Top-down path: semantic context propagates downward.
The output is three feature pyramids, each enriched with both local and global information.

**Head:** at each scale, predicts bounding boxes and class scores for a set of predefined
anchor shapes. Three anchors per scale = nine total anchor shapes, automatically clustered
to match the aspect-ratio distribution of your training data.

```
Input image
     │
     ▼
CSPDarknet53 backbone
  P3 ─────────────────────────────────────────────► Head (small objects)
  P4 ────────────────────────────────────────────►  Head (medium objects)
  P5 ──────────────────────────► PANet ──────────►  Head (large objects)
```

**YOLOv5 size variants:**

| Model | Parameters | Speed | mAP |
|-------|-----------|-------|-----|
| YOLOv5n | ~1.9 M | Fastest | Lower |
| YOLOv5s | ~7.2 M | Fast | Good |
| YOLOv5m | ~21 M | Moderate | Better |
| YOLOv5l | ~47 M | Slower | Best |

For most microscopy detection tasks with a T4 GPU, `yolov5s` gives the best speed/accuracy
trade-off. Start there and scale up only if mAP is insufficient.

### Preparing a custom dataset

YOLOv5 expects a specific directory structure and annotation format:

```
dataset/
├── images/
│   ├── train/  (images, .jpg or .png)
│   └── val/
└── labels/
    ├── train/  (one .txt per image)
    └── val/
```

Each `.txt` annotation file has one line per object:
```
<class_id> <cx> <cy> <w> <h>
```
where all coordinates are **normalised to [0, 1]** relative to image width/height. This
normalisation makes annotations resolution-independent.

**Using Roboflow:** Roboflow is a web platform for managing, augmenting, and exporting
annotation datasets. It can convert from LabelImg, CVAT, and other formats directly to
YOLOv5-PyTorch format, export a `data.yaml` config file, and apply online augmentation.
The notebook shows how to use the Roboflow API to download directly to Colab.

**The `data.yaml` config file:**
```yaml
path: /content/dataset     # root directory
train: images/train        # relative train path
val: images/val            # relative val path
nc: 2                      # number of classes
names: ['cell', 'nucleus'] # class names
```

### Key training hyperparameters

| Hyperparameter | What it controls | Typical starting value |
|---------------|-----------------|----------------------|
| `--img` | Input image size (resized to square) | 640 |
| `--batch-size` | Samples per gradient update | 16 (T4) |
| `--epochs` | Number of full passes over training data | 50–100 |
| `--weights` | Pre-trained checkpoint | `yolov5s.pt` (transfer learning) |
| `--hyp` | Hyperparameter YAML (lr, augmentation) | `hyp.scratch-low.yaml` |

**Transfer learning from COCO:** YOLOv5 pre-trained on COCO has already learned to detect
edges, textures, and shapes from 80 diverse object categories. Even though microscopy
objects look nothing like COCO categories, the low-level feature representations transfer
well. Starting from `yolov5s.pt` instead of random weights typically reduces the number
of epochs needed to reach good mAP by 2–3×.

### Evaluation: precision, recall, and mAP

A predicted box is a **true positive** if its IoU with any ground-truth box of the same
class exceeds a threshold (typically 0.5), and no other predicted box has already matched
that ground-truth box.

**Precision:** of all predicted boxes, what fraction are correct?
$P = \text{TP} / (\text{TP} + \text{FP})$

**Recall:** of all ground-truth objects, what fraction did the model find?
$R = \text{TP} / (\text{TP} + \text{FN})$

**Precision-recall curve:** as the confidence threshold decreases, more boxes are predicted
— recall increases but precision decreases. The area under this curve is the **Average
Precision (AP)** for one class.

**mAP@0.5** — the standard YOLOv5 metric — averages AP across all classes at IoU = 0.5.
**mAP@0.5:0.95** averages across IoU thresholds from 0.5 to 0.95 in steps of 0.05,
penalising imprecise localisation more heavily.

### Centroid-based 3-D object tracking

After detecting or segmenting objects in each frame, tracking links those detections into
continuous trajectories. The simplest approach: represent each object by its centroid and
link centroids across consecutive frames by minimising total displacement.

This is the **Linear Assignment Problem (LAP):** given a set of detected objects in frame
$t$ and a set in frame $t+1$, find the assignment of detections to tracks that minimises
total Euclidean distance. Formulated as a cost matrix $C$ where $C_{ij}$ is the distance
between track $i$ in frame $t$ and detection $j$ in frame $t+1$, solved by the Hungarian
algorithm in $O(n^3)$.

**Handling birth and death:** the basic LAP assignment cannot handle objects appearing or
disappearing. The **extended LAP** (Jaqaman et al., 2008) adds dummy rows and columns to
the cost matrix to allow each track to be "born" (no predecessor), "die" (no successor),
or "link" (normal assignment). The cost of birth/death is set by an estimated particle
density; the solver finds the globally optimal assignment simultaneously.

**LapTrack** implements this algorithm with a clean Python API and supports:
- 2-D and 3-D data
- Custom cost functions (Euclidean distance, Mahalanobis distance, feature similarity)
- Gap closing: linking a track in frame $t$ to a detection in frame $t+k$ when the
  object was missed in intermediate frames

**Napari** is used for visualisation: it displays the 3-D time-lapse volume and overlays
the computed tracks as line segments, allowing you to verify tracking quality visually.

### Why tracking matters in biology

Manual tracking is impractical at scale. A single 4-D microscopy experiment (3-D + time)
can contain thousands of cells across hundreds of time points. Automated LAP tracking
enables:

- **Cell division detection:** a track that splits into two at frame $t$ is a mitotic event
- **Migration speed quantification:** the arc length of a track divided by duration gives
  mean speed
- **Lineage tracing:** which daughter cells descended from which progenitor
- **Intracellular transport:** linking vesicle or organelle detections to measure velocity
  and directionality along cytoskeletal tracks

---

## Notebooks

| File | Description |
|------|-------------|
| `object_detection.ipynb` | YOLOv5 installation and Roboflow dataset setup; `data.yaml` configuration; training from COCO pre-trained weights; training curve visualisation (box loss, objectness loss, classification loss, mAP); validation on held-out images; confidence threshold sweep; inference on new images |
| `lap_laptrack.ipynb` | Loading a 3-D time-lapse volume; instance label centroid extraction; LapTrack configuration and LAP solving; track visualisation in napari; division detection; per-track speed and displacement statistics |

---

## Setup

```bash
# YOLOv5 (clones its own repository with all requirements)
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt

# Tracking
pip install laptrack napari
```

All installs are handled automatically in the first cell of each notebook.

---

## Hardware

| Task | GPU needed | VRAM usage | Notes |
|------|-----------|-----------|-------|
| YOLOv5 training (yolov5s, img=640) | Yes | ~4 GB | T4 recommended; CPU fallback is very slow |
| YOLOv5 inference | Optional | ~2 GB | Runs on CPU for evaluation |
| LapTrack | No | CPU only | LAP solving scales as O(n³) |

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `AssertionError: train: No labels found` | Wrong path in `data.yaml` or missing `.txt` annotation files | Verify `labels/train/` contains `.txt` files with same basename as images |
| Training mAP stays at 0 | Anchor shapes poorly matched to object sizes | Run `python utils/autoanchor.py` to re-cluster anchors on your dataset |
| CUDA out of memory | Batch size too large for model + image size | Reduce `--batch-size` to 8 or 4; or use `--img 320` |
| LapTrack returns too few tracks | Search radius too small for frame-to-frame displacement | Increase `max_distance` parameter to match typical object speed |
| Napari does not open on Colab | Colab has no display | Run LapTrack results locally; export tracks as CSV and visualise with matplotlib |

---

## Tips for beginners

- Check the YOLOv5 training log's `labels.jpg` output — it visualises the distribution of
  bounding box sizes and aspect ratios across your training set. If your objects are small
  relative to the image, consider tiling (splitting each image into overlapping crops before
  training).
- Confidence and NMS thresholds are tunable at inference time without retraining. Sweep
  both on a validation set to find the operating point that matches your recall/precision
  preference.
- Before running LapTrack on real data, test it on a synthetic dataset where you know the
  ground-truth tracks. This lets you tune `max_distance` and `gap_closing_max_frame_count`
  without ambiguity.
- In volumetric data, centroids are in (z, y, x) voxel coordinates. If voxel spacing is
  anisotropic (e.g. 0.3 µm in XY, 1.0 µm in Z), scale the coordinates by voxel size in
  µm before computing distances, or LapTrack will over-penalise Z-axis motion.
