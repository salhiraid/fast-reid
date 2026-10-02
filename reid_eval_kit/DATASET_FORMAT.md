# Dataset format the loader reads (`reid_data/loader.py`)

```
DATASET_ROOT/videos/<video_id>/meta.json
DATASET_ROOT/videos/<video_id>/crops/<track>/<frame>.jpg        (224x224 BGR JPEG; paths in meta.json are relative to the video folder)
```
Only `tracks[*].crops` of `meta.json` are used (never `rejected_crops` or `rejected/`). Fields read (missing optional ones are treated as unknown):

| level | required | optional |
|---|---|---|
| video | `video_id` (= folder name), `n_crops`, `tracks` | `site`, `pov`, `country`, `orient`, `time`, `calibration.available`, `crop_params.effective_fps`, `quality_params` |
| track | `track_id`, `crops` | `tracklet_id` (default `<video_id>_<track_id>`, **the object id**), `identity_id`, `label` |
| crop | `crop_file`, `frame` | `timestamp_s`, `crop_transform.pad_x/pad_y` (letterbox padding), `bbox_wh`, `position_road_m` + `position_reliable` + `distance_m`, `view_azimuth_deg`, `view_bin`, `occluded_annot`, `overlap_by_closer_box`, `keypoints` (`visible` or `points`), `quality` |

Backslashes in `crop_file` are normalised. `n_crops` must equal the number of kept crops (checked). Without calibration / keypoints, the
position, azimuth and keypoint bins are simply "unknown" and everything else works (the `plain`, per-site and site-matching evaluations
do not use them at all). Objects exist **inside a video only** (`tracklet_id` is per video).
