#!/usr/bin/env bash
# Shared path setup for raw-data reproduction scripts.
#
# The Python research scripts expect:
#   data/cv2024/Dataset/{training,validation}/...
#   data/kvasir_capsule/labelled_images/...
# Reviewers usually provide these via CV2024_ROOT and KVASIR_ROOT.  This helper
# creates non-destructive symlinks when the expected local paths are absent.

set -euo pipefail

setup_capsule_data_links() {
  : "${CV2024_ROOT:?Set CV2024_ROOT to the CV2024 Dataset directory or its parent}"
  : "${KVASIR_ROOT:?Set KVASIR_ROOT to Kvasir-Capsule labelled_images/ or its parent}"

  export CAPSULE_ROOT="${CAPSULE_ROOT:-$PWD}"
  export CAPSULE_ARTIFACT_ROOT="${CAPSULE_ARTIFACT_ROOT:-$PWD}"

  local cv_dataset="$CV2024_ROOT"
  if [[ -d "$CV2024_ROOT/Dataset" ]]; then
    cv_dataset="$CV2024_ROOT/Dataset"
  fi
  if [[ ! -d "$cv_dataset/training" || ! -d "$cv_dataset/validation" ]]; then
    echo "ERROR: CV2024_ROOT must point to a Dataset directory with training/ and validation/." >&2
    echo "       Got: $CV2024_ROOT" >&2
    return 2
  fi
  export CV2024_DATASET_ROOT="$cv_dataset"

  local kvasir_images="$KVASIR_ROOT"
  if [[ -d "$KVASIR_ROOT/labelled_images" ]]; then
    kvasir_images="$KVASIR_ROOT/labelled_images"
  fi
  if [[ ! -d "$kvasir_images" ]]; then
    echo "ERROR: KVASIR_ROOT must point to labelled_images/ or its parent." >&2
    echo "       Got: $KVASIR_ROOT" >&2
    return 2
  fi
  export KVASIR_IMAGES_ROOT="$kvasir_images"

  mkdir -p data/cv2024 data/kvasir_capsule results

  _link_if_needed "data/cv2024/Dataset" "$cv_dataset"
  _link_if_needed "data/kvasir_capsule/labelled_images" "$kvasir_images"
}

setup_cv2024_data_link() {
  : "${CV2024_ROOT:?Set CV2024_ROOT to the CV2024 Dataset directory or its parent}"

  export CAPSULE_ROOT="${CAPSULE_ROOT:-$PWD}"
  export CAPSULE_ARTIFACT_ROOT="${CAPSULE_ARTIFACT_ROOT:-$PWD}"

  local cv_dataset="$CV2024_ROOT"
  if [[ -d "$CV2024_ROOT/Dataset" ]]; then
    cv_dataset="$CV2024_ROOT/Dataset"
  fi
  if [[ ! -d "$cv_dataset/training" || ! -d "$cv_dataset/validation" ]]; then
    echo "ERROR: CV2024_ROOT must point to a Dataset directory with training/ and validation/." >&2
    echo "       Got: $CV2024_ROOT" >&2
    return 2
  fi
  export CV2024_DATASET_ROOT="$cv_dataset"

  mkdir -p data/cv2024 results
  _link_if_needed "data/cv2024/Dataset" "$cv_dataset"
}

_link_if_needed() {
  local dest="$1"
  local target="$2"

  if [[ "$dest" == "$target" ]]; then
    return 0
  fi
  if [[ -e "$dest" && ! -L "$dest" ]]; then
    return 0
  fi
  if [[ -L "$dest" ]]; then
    rm -f "$dest"
  fi
  ln -s "$target" "$dest"
}
