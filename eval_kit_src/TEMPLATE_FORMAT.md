# Template format (what the evaluation reads)

A **template** is the saved embedding of every kept crop of the validation and test videos, for one model and one preprocessing mode.
Put one folder per model/mode inside a `templates/` folder:

```
templates/
  eval_split_v1.json                 <- optional: the split used to extract (else give --split)
  mymodel__unpad_stretch/
    manifest.json
    validation/<video_id>.npz
    test/<video_id>.npz
  mymodel__letterbox/...             <- a second mode of the same model: the evaluation picks one on the validation videos
  othermodel__letterbox/...
```

## `<video_id>.npz`: arrays aligned row by row, in the **canonical record order**

| array | dtype / shape | content |
|---|---|---|
| `emb` | float32 `(N, D)` | raw embedding, **before** L2 normalisation (the evaluation normalises) |
| `crop_uid` | str `(N,)` | `"<video_id>/<crop_file>"`, forward slashes, exactly as `reid_data.load_dataset` produces it |
| `tracklet_id` | str `(N,)` | object id |
| `frame` | int32 `(N,)` | frame number |

Canonical order = `reid_data.load_dataset(root, video_ids=[...])` order: sorted by `(video_id, tracklet_id, frame)`. The evaluation joins on `crop_uid`
and **fails loudly** if any crop is missing, extra or reordered. Only kept crops (`tracks[*].crops` of `meta.json`) are encoded.

## `manifest.json`

Required: `model_name`, `preproc_mode` (`letterbox` or `unpad_stretch`), `split_sha256` (sha256 of the split json used to extract).
Recommended: `split_file`, `checkpoint`, `checkpoint_sha256`, `embedding_dim`, `input_size`, `normalisation`, `flip_tta`, `device`, `torch`, `created`, `notes`.

## Two ways to create templates

1. **Any encoder, one function** (no npz code):
   ```bash
   python export_templates.py --data DATASET --split splits/eval_split_v1.json --out templates \
       --model-name mymodel --preproc letterbox unpad_stretch --encoder mypackage.mymodule:encode
   ```
   where `encode(images)` takes a list of BGR uint8 images (cut according to the preprocessing mode, not resized), resizes / normalises them and
   returns a `(len(images), D)` array.
2. **Write the npz yourself** (any language): follow the table above. The split file must exist before: `python make_split.py --data DATASET --out splits/eval_split_v1.json`.

Preprocessing modes: `letterbox` = the stored 224x224 image as is; `unpad_stretch` = the black letterbox padding cut away (`img[pad_y:224-pad_y, pad_x:224-pad_x]`),
the vehicle area then stretched by your encoder to its input size. Record in the manifest what your encoder really did.
