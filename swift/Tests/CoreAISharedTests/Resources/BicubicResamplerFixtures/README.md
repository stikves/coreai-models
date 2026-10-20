# Bicubic resampler fixtures

Inputs are crops of scikit-image sample images (`chelsea`, `astronaut`, `camera`; public domain)
plus seeded synthetic images. `manifest.json` lists each case; a case `name` has files `name.png`,
`name.decoded.rgb` and `name.resized.rgb`.

- `*.decoded.rgb`: `PIL.Image.open(input).convert("RGB")` as raw HWC uint8 bytes.
- `*.resized.rgb`: `torchvision.transforms.v2.functional.resize(chw_uint8, [target_height, target_width],
  BICUBIC, antialias=True)` on CPU, as raw HWC uint8 bytes.

To regenerate a case, run those two calls on its input and write the bytes next to it.
