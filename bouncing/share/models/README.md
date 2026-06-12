# Models Overview

This repository contains the latest versions of the detection and classification models used in the project.

---

# `best_7_class.pt`

Latest version of the multi-class detector.

## Detected Classes

| ID | Class Name |
|----|------------|
| 0 | Hexagon |
| 1 | Star |
| 2 | Triangle |
| 3 | Number 3 |
| 4 | Number 4 |
| 5 | Number 5 |
| 6 | ArUco |
| 7 | Start Base |

---

# `best_classifier.pt`

Latest version of the classifier for numbers and ArUco markers.

## Classification Classes

| ID | Class Name |
|----|------------|
| 3 | Number 3 |
| 4 | Number 4 |
| 5 | Number 5 |
| 6 | ArUco |

---

# `best_detector.pt`

Latest version of the detector for shapes and central symbols.

## Detected Classes

| ID | Class Name |
|----|------------|
| 0 | Hexagon |
| 1 | Star |
| 2 | Triangle |
| 3 | Number / ArUco |

---

# Notes

- `best_7_class.pt` detects all project classes directly, but makes a lot of mistakes.
- `best_detector.pt` is a simplified detector focused on shapes and central symbols.
- `best_classifier.pt` should be used after detecting class `3` from `best_detector.pt` to distinguish between:
  - Number 3
  - Number 4
  - Number 5
  - ArUco
