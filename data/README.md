# Data

MAR20 images are not redistributed here; obtain them from the official MAR20 release.

## `splits.json`

MAR20 image ids for the three splits used in the paper, plus the scene-family id of every image
(images of one airfield scene share a family and always fall in the same split).

| Split | Images | Use |
|---|---:|---|
| `train` | 1,805 | detector, appearance references, part references, region experts |
| `development` | 663 | cross-fitted scores; VERA-z scales and the rejection threshold are fitted here |
| `test` | 1,374 | evaluated once, with the development threshold frozen |

## `mar20_parts/`: MAR20-Parts annotations

Polygon annotations of aircraft parts on 511 MAR20 aircraft, used to train the part detector.

| File | Aircraft | Polygons | MAR20 split of the source images |
|---|---:|---:|---|
| `train.json` | 400 | 3,369 | train |
| `holdout.json` | 22 | 187 | train |
| `val.json` | 53 | 436 | development |
| `val_add.json` | 36 | 313 | train |

**What is (and is not) annotated.** MAR20-Parts covers the ten known types (A2, A3, A4, A6, A7, A8, A9,
A10, A14, A20) and the five fighter types (A1, A5, A13, A16, A19). It contains **no** aircraft of the
evaluated unknown types (E-8 / A11, Tu-22 / A12, F-22 / A15, Tu-95 / A17, KC-10 / A18) and **no image
from the test split**. The paper also reports a part detector trained on the ten known types only
(fighter annotations removed); that variant gives equal or higher unknown rejection.

**Format.** COCO-style JSON with polygons in **MAR20 source-image pixel coordinates** (x, y).

- `categories`: nacelle, propeller_disk, exhaust, intake, main_wing, horizontal_tail, vertical_tail,
  nose_cockpit, canard, rotodome.
- `images`: one record per annotated aircraft:
  - `mar20_image_id`, `mar20_object_index` (0-based object order in the MAR20 annotation XML),
    `aircraft_type_code`, `mar20_split`, `scene_family_id`, `subset`;
  - `source_to_crop`: a 2×3 affine (row-major) mapping MAR20 pixels to the 448×448 heading-aligned
    crop on which the part was annotated and on which the part detector runs. Apply it to the
    polygons to recover the training crop geometry;
  - `transform`: `exact` (recorded when the crop was cut) or `registered` (recovered by SIFT + RANSAC
    affine registration of the crop against the MAR20 image; `registration_inliers` and
    `registration_median_residual_px` in crop pixels are given). Every transform maps onto a MAR20
    object of the recorded type.
- `annotations`: `polygon` (list of [x, y]), `bbox` ([x, y, w, h]), `category_id` and `attributes`
  (`side`, `visibility`, `geometry_quality`, `planform`, `propulsion_type`, `mount_region`,
  `evidence_kind`, `unit_count`; `cant` where applicable).

To rebuild the training crops: read the MAR20 image, warp it with `source_to_crop` to 448×448
(`cv2.warpAffine(image, M, (448, 448))`) and map the polygons with the same matrix.
