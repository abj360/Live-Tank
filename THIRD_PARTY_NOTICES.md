# Third-party notices

## DeepSORT tracker core — `live_tank/tracking/`

The Kalman filter, matching cascade and track lifecycle in `live_tank/tracking/`
are taken from **DeepSORT** by Nicolai Wojke:

- Upstream: https://github.com/nwojke/deep_sort
- Reached here through https://github.com/theAIGuysCode/yolov3_deepsort and a
  fork adapted for fish (`LICENSE-upstream`, Apache License 2.0).

Cite the original work:

```
@inproceedings{Wojke2017simple,
  title     = {Simple Online and Realtime Tracking with a Deep Association Metric},
  author    = {Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle = {2017 IEEE International Conference on Image Processing (ICIP)},
  year      = {2017},
  pages     = {3645--3649},
  doi       = {10.1109/ICIP.2017.8296962}
}
```

### Changes made here

- Kept only the tracker core: Kalman filter, IoU and appearance matching,
  linear assignment, track and tracker. The YOLOv3 detector, the TensorFlow
  appearance encoder (`mars-small128`) and the non-max-suppression helper are
  not used and were left out.
- `detection.py` and the matching code were updated for NumPy 2 (`np.float`
  was removed from NumPy).
- Appearance features come from a colour histogram computed in
  `live_tank/detector.py` instead of the CNN embedding.

### Licence note

DeepSORT is distributed by its author under the GPL-3.0 licence, while the
repository this copy came through carries Apache 2.0 (`LICENSE-upstream`).
Before publishing this repository under any other licence, or using it
commercially, check which terms apply to the vendored tracker code.
