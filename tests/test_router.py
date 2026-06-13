from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PIPELINE_ROUTES = {
    "clean_labels": PROJECT_ROOT / "src" / "preprocess" / "clean_labels.py",
    "select_pseudo": PROJECT_ROOT / "src" / "preprocess" / "select_pseudo.py",
    "move_pseudo": PROJECT_ROOT / "src" / "preprocess" / "move_pseudo.py",
    "ensemble": PROJECT_ROOT / "src" / "inference" / "ensemble.py",
    "post_process": PROJECT_ROOT / "src" / "postprocess" / "post_process.py",
    "pack_submission": PROJECT_ROOT / "src" / "postprocess" / "pack_submission.py",
    "fix_dataset_json": PROJECT_ROOT / "src" / "utils" / "fix_dataset_json.py",
}


def label_name_for_image(image_name: str) -> str:
    """Map nnU-Net image filenames to their expected label filenames."""
    return image_name.replace("_0000.nii.gz", ".nii.gz")


def test_pipeline_routes_exist():
    missing_routes = [
        route_name
        for route_name, route_path in PIPELINE_ROUTES.items()
        if not route_path.is_file()
    ]

    assert missing_routes == []


def test_training_image_names_route_to_label_names():
    test_images = [
        "patient0001_0000.nii.gz",
        "patient0055_0000.nii.gz",
        "case_abc_0000.nii.gz",
    ]

    assert [label_name_for_image(name) for name in test_images] == [
        "patient0001.nii.gz",
        "patient0055.nii.gz",
        "case_abc.nii.gz",
    ]
