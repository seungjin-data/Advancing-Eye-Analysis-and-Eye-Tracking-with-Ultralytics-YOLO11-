# Advancing-Eye-Analysis-and-Eye-Tracking-with-Ultralytics-YOLO11-
Advancing Eye Analysis and Eye-Tracking with Ultralytics YOLO11 and EMME: Applications in Archaeology, Ophthalmology, Biometric Security, and Human-Computer Interaction
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17120643.svg)](https://doi.org/10.5281/zenodo.17120643)

## Reproducibility

Results reproduce with commit `5cafd160384b4cdf84d58c8b71ddd8657c701b99` (tags `v1.0.0`, `v1.0.1`);
archived at Zenodo (version DOI: `10.5281/zenodo.17120643`; concept DOI: `10.5281/zenodo.17120642`).

### Quick start
```bash
pip install -r requirements.txt
python scripts/eval.py --cfg configs/eye_yolo11.yaml --weights <zenodo_or_path> --seed 42
