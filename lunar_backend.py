import os

# ================================================================
# SERVER MEMORY / CPU SETTINGS
# ================================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR


# ================================================================
# CONFIGURATION
# ================================================================

TARGET_WIDTH = 320
TARGET_HEIGHT = 192

CONFIDENCE_THRESHOLD = 0.30
MIN_MATCHES = 8
RANSAC_THRESHOLD = 5.0

GRID_ROWS = 4
GRID_COLS = 4

MAX_VISUAL_MATCHES = 100


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device("cpu")

try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ================================================================
# LAZY LoFTR
# ================================================================

matcher = None


def get_matcher():

    global matcher

    if matcher is None:

        print("Loading LoFTR...")

        matcher = LoFTR(
            pretrained="outdoor"
        )

        matcher = matcher.to(DEVICE)
        matcher.eval()

        print("LoFTR loaded successfully")

    return matcher


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def preprocess_image(image):

    if image is None:
        raise ValueError(
            "Could not load input image."
        )

    # Remove alpha channel
    if (
        len(image.shape) == 3
        and image.shape[2] == 4
    ):
        image = image[:, :, :3]

    # Convert to grayscale
    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()

    # Convert to uint8
    if gray.dtype != np.uint8:

        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    # Local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Mild noise reduction
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Final normalization
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gray.astype(np.uint8)


# ================================================================
# RESIZE
# ================================================================

def resize_image(image):

    return cv2.resize(
        image,
        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )


# ================================================================
# IMAGE TO TENSOR
# ================================================================

def image_to_tensor(image):

    tensor = (
        torch.from_numpy(image)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return tensor.to(DEVICE)


# ================================================================
# LoFTR
# ================================================================

def run_loftr(
    image_A,
    image_B
):

    model = get_matcher()

    image_A = resize_image(image_A)
    image_B = resize_image(image_B)

    tensor_A = image_to_tensor(image_A)
    tensor_B = image_to_tensor(image_B)

    try:

        with torch.inference_mode():

            output = model(
                {
                    "image0": tensor_A,
                    "image1": tensor_B
                }
            )

        points_A = (
            output["keypoints0"]
            .detach()
            .cpu()
            .numpy()
        )

        points_B = (
            output["keypoints1"]
            .detach()
            .cpu()
            .numpy()
        )

        confidence = (
            output["confidence"]
            .detach()
            .cpu()
            .numpy()
        )

    finally:

        del tensor_A
        del tensor_B

        del output

    return (
        points_A,
        points_B,
        confidence
    )


# ================================================================
# CONFIDENCE FILTER
# ================================================================

def filter_matches(
    points_A,
    points_B,
    confidence
):

    if len(points_A) == 0:

        return (
            points_A,
            points_B,
            confidence
        )

    mask = (
        confidence
        >= CONFIDENCE_THRESHOLD
    )

    return (
        points_A[mask],
        points_B[mask],
        confidence[mask]
    )


# ================================================================
# DUPLICATE REMOVAL
# ================================================================

def remove_duplicate_matches(
    points_A,
    points_B,
    confidence
):

    if len(points_A) == 0:

        return (
            points_A,
            points_B,
            confidence
        )

    rounded = np.round(
        points_A
    ).astype(np.int32)

    _, indices = np.unique(
        rounded,
        axis=0,
        return_index=True
    )

    indices = np.sort(indices)

    return (
        points_A[indices],
        points_B[indices],
        confidence[indices]
    )


# ================================================================
# RANSAC
# ================================================================

def run_ransac(
    points_A,
    points_B,
    confidence
):

    if len(points_A) < MIN_MATCHES:
        return None

    H, mask = cv2.findHomography(
        points_A.astype(np.float32),
        points_B.astype(np.float32),
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:
        return None

    mask = (
        mask
        .ravel()
        .astype(bool)
    )

    inliers_A = points_A[mask]
    inliers_B = points_B[mask]
    inlier_confidence = confidence[mask]

    total = len(points_A)
    inliers = len(inliers_A)

    return {
        "H": H,
        "mask": mask,
        "inliers_A": inliers_A,
        "inliers_B": inliers_B,
        "inlier_confidence": inlier_confidence,
        "outliers": total - inliers,
        "inlier_ratio": (
            inliers / float(total)
        )
    }


# ================================================================
# REGISTRATION ERROR
# ================================================================

def calculate_registration_error(
    H,
    points_A,
    points_B
):

    projected = cv2.perspectiveTransform(
        points_A.reshape(
            -1,
            1,
            2
        ).astype(np.float32),
        H
    ).reshape(
        -1,
        2
    )

    errors = np.linalg.norm(
        projected - points_B,
        axis=1
    )

    rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    mean_error = float(
        np.mean(errors)
    )

    maximum_error = float(
        np.max(errors)
    )

    subpixel_percentage = float(
        np.mean(
            errors < 1.0
        ) * 100.0
    )

    return {
        "errors": errors,
        "rmse": rmse,
        "mean_error": mean_error,
        "maximum_error": maximum_error,
        "subpixel_percentage":
            subpixel_percentage
    }


# ================================================================
# SPATIAL ANALYSIS
# ================================================================

def analyze_spatial_distribution(
    points,
    confidence
):

    counts = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=np.int32
    )

    quality = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=np.float32
    )

    for point, conf in zip(
        points,
        confidence
    ):

        x = float(point[0])
        y = float(point[1])

        col = int(
            x
            / TARGET_WIDTH
            * GRID_COLS
        )

        row = int(
            y
            / TARGET_HEIGHT
            * GRID_ROWS
        )

        col = min(
            max(col, 0),
            GRID_COLS - 1
        )

        row = min(
            max(row, 0),
            GRID_ROWS - 1
        )

        counts[row, col] += 1
        quality[row, col] += float(conf)

    for r in range(GRID_ROWS):

        for c in range(GRID_COLS):

            if counts[r, c] > 0:

                quality[r, c] /= (
                    counts[r, c]
                )

    occupied = np.count_nonzero(
        counts
    )

    total_cells = (
        GRID_ROWS * GRID_COLS
    )

    coverage = (
        occupied
        / float(total_cells)
    )

    nonzero = counts[
        counts > 0
    ]

    if len(nonzero) > 1:

        mean_count = float(
            np.mean(nonzero)
        )

        std_count = float(
            np.std(nonzero)
        )

        if mean_count > 0:

            variation = (
                std_count
                / mean_count
            )

            uniformity = (
                1.0
                / (1.0 + variation)
            )

        else:

            uniformity = 0.0

    elif len(nonzero) == 1:

        uniformity = 1.0

    else:

        uniformity = 0.0

    return {
        "counts": counts,
        "quality": quality,
        "coverage": float(coverage),
        "uniformity": float(uniformity)
    }


# ================================================================
# QUALITY SCORE
# ================================================================

def calculate_quality_score(
    inlier_ratio,
    average_confidence,
    coverage,
    uniformity,
    rmse
):

    error_score = max(
        0.0,
        1.0 - min(
            rmse / 10.0,
            1.0
        )
    )

    score = (

        0.30 * inlier_ratio

        + 0.25 * average_confidence

        + 0.20 * coverage

        + 0.15 * uniformity

        + 0.10 * error_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ================================================================
# REGISTER IMAGE
# ================================================================

def create_registered_image(
    image_A,
    H
):

    image_A = resize_image(
        image_A
    )

    registered = cv2.warpPerspective(
        image_A,
        H,
        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        )
    )

    return registered


# ================================================================
# RANSAC VISUALIZATION
# ================================================================

def create_ransac_visualization(
    image_A,
    image_B,
    points_A,
    points_B,
    confidence
):

    image_A = resize_image(
        image_A
    )

    image_B = resize_image(
        image_B
    )

    left = cv2.cvtColor(
        image_A,
        cv2.COLOR_GRAY2BGR
    )

    right = cv2.cvtColor(
        image_B,
        cv2.COLOR_GRAY2BGR
    )

    combined = np.hstack(
        [left, right]
    )

    offset = TARGET_WIDTH

    number = min(
        MAX_VISUAL_MATCHES,
        len(points_A)
    )

    if number == 0:
        return combined

    indices = np.argsort(
        confidence
    )[::-1][:number]

    for index in indices:

        x1 = int(
            points_A[index][0]
        )

        y1 = int(
            points_A[index][1]
        )

        x2 = int(
            points_B[index][0]
        ) + offset

        y2 = int(
            points_B[index][1]
        )

        cv2.circle(
            combined,
            (x1, y1),
            2,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            combined,
            (x2, y2),
            2,
            (0, 255, 0),
            -1
        )

        cv2.line(
            combined,
            (x1, y1),
            (x2, y2),
            (255, 180, 0),
            1
        )

    return combined


# ================================================================
# QUALITY MAP
# ================================================================

def create_quality_map(
    image,
    cell_quality
):

    image = resize_image(
        image
    )

    heat = (
        np.clip(
            cell_quality,
            0,
            1
        ) * 255
    ).astype(np.uint8)

    heat = cv2.resize(
        heat,
        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        ),
        interpolation=cv2.INTER_NEAREST
    )

    heatmap = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_PLASMA
    )

    original = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    result = cv2.addWeighted(
        original,
        0.45,
        heatmap,
        0.55,
        0
    )

    return result


# ================================================================
# SPATIAL VISUALIZATION
# ================================================================

def create_spatial_visualization(
    image,
    points
):

    image = resize_image(
        image
    )

    result = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    for point in points:

        x = int(point[0])
        y = int(point[1])

        if (
            0 <= x < TARGET_WIDTH
            and
            0 <= y < TARGET_HEIGHT
        ):

            cv2.circle(
                result,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    return result


# ================================================================
# SAVE
# ================================================================

def save_image(
    image,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    if not cv2.imwrite(
        path,
        image
    ):

        raise IOError(
            "Could not save: "
            + path
        )


# ================================================================
# MAIN ANALYSIS
# ================================================================

def analyze_lunar_images(
    file_A,
    file_B,
    output_folder="lunar_results"
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    print("\n")
    print("=" * 60)
    print("LUNAR IMAGE ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    image_A = cv2.imread(
        file_A,
        cv2.IMREAD_UNCHANGED
    )

    image_B = cv2.imread(
        file_B,
        cv2.IMREAD_UNCHANGED
    )

    if image_A is None:
        raise ValueError(
            "Could not load Image A."
        )

    if image_B is None:
        raise ValueError(
            "Could not load Image B."
        )

    print("Images loaded")

    # ------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------

    processed_A = preprocess_image(
        image_A
    )

    processed_B = preprocess_image(
        image_B
    )

    save_image(
        processed_A,
        os.path.join(
            output_folder,
            "preprocessed_A.png"
        )
    )

    save_image(
        processed_B,
        os.path.join(
            output_folder,
            "preprocessed_B.png"
        )
    )

    print("Preprocessing completed")

    # ------------------------------------------------------------
    # LoFTR
    # ------------------------------------------------------------

    points_A, points_B, confidence = (
        run_loftr(
            processed_A,
            processed_B
        )
    )

    raw_matches = len(
        points_A
    )

    print(
        "Raw LoFTR matches:",
        raw_matches
    )

    # ------------------------------------------------------------
    # FILTER
    # ------------------------------------------------------------

    points_A, points_B, confidence = (
        filter_matches(
            points_A,
            points_B,
            confidence
        )
    )

    print(
        "After confidence filter:",
        len(points_A)
    )

    # ------------------------------------------------------------
    # DUPLICATES
    # ------------------------------------------------------------

    points_A, points_B, confidence = (
        remove_duplicate_matches(
            points_A,
            points_B,
            confidence
        )
    )

    print(
        "After duplicate removal:",
        len(points_A)
    )

    if len(points_A) < MIN_MATCHES:

        raise RuntimeError(
            "Not enough reliable matches "
            "for RANSAC."
        )

    # ------------------------------------------------------------
    # RANSAC
    # ------------------------------------------------------------

    ransac = run_ransac(
        points_A,
        points_B,
        confidence
    )

    if ransac is None:

        raise RuntimeError(
            "RANSAC could not estimate "
            "a valid transformation."
        )

    inliers_A = (
        ransac["inliers_A"]
    )

    inliers_B = (
        ransac["inliers_B"]
    )

    inlier_confidence = (
        ransac["inlier_confidence"]
    )

    print(
        "RANSAC inliers:",
        len(inliers_A)
    )

    print(
        "RANSAC outliers:",
        ransac["outliers"]
    )

    print(
        "Inlier ratio:",
        round(
            ransac[
                "inlier_ratio"
            ] * 100,
            2
        ),
        "%"
    )

    # ------------------------------------------------------------
    # ERROR
    # ------------------------------------------------------------

    error = calculate_registration_error(
        ransac["H"],
        inliers_A,
        inliers_B
    )

    # ------------------------------------------------------------
    # SPATIAL
    # ------------------------------------------------------------

    spatial = (
        analyze_spatial_distribution(
            inliers_A,
            inlier_confidence
        )
    )

    # ------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------

    average_confidence = float(
        np.mean(
            inlier_confidence
        )
    )

    best_confidence = float(
        np.max(
            inlier_confidence
        )
    )

    lowest_confidence = float(
        np.min(
            inlier_confidence
        )
    )

    # ------------------------------------------------------------
    # QUALITY
    # ------------------------------------------------------------

    quality_score = (
        calculate_quality_score(
            ransac[
                "inlier_ratio"
            ],
            average_confidence,
            spatial[
                "coverage"
            ],
            spatial[
                "uniformity"
            ],
            error[
                "rmse"
            ]
        )
    )

    # ------------------------------------------------------------
    # REGISTERED IMAGE
    # ------------------------------------------------------------

    registered = (
        create_registered_image(
            processed_A,
            ransac["H"]
        )
    )

    save_image(
        registered,
        os.path.join(
            output_folder,
            "registered_refined.png"
        )
    )

    # ------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------

    ransac_visual = (
        create_ransac_visualization(
            processed_A,
            processed_B,
            inliers_A,
            inliers_B,
            inlier_confidence
        )
    )

    save_image(
        ransac_visual,
        os.path.join(
            output_folder,
            "ransac_inliers.png"
        )
    )

    quality_visual = (
        create_quality_map(
            processed_A,
            spatial["quality"]
        )
    )

    save_image(
        quality_visual,
        os.path.join(
            output_folder,
            "correspondence_quality_map.png"
        )
    )

    spatial_visual = (
        create_spatial_visualization(
            processed_A,
            inliers_A
        )
    )

    save_image(
        spatial_visual,
        os.path.join(
            output_folder,
            "spatial_distribution.png"
        )
    )

    # ------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------

    if (
        ransac["inlier_ratio"] >= 0.70
        and
        error["rmse"] < 5.0
        and
        spatial["coverage"] >= 0.50
    ):

        status = (
            "HIGH-QUALITY CORRESPONDENCE"
        )

    elif (
        ransac["inlier_ratio"] >= 0.50
        and
        error["rmse"] < 10.0
    ):

        status = (
            "MODERATE-QUALITY CORRESPONDENCE"
        )

    else:

        status = (
            "LOW-QUALITY CORRESPONDENCE"
        )

    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------

    result = {

        "status":
            status,

        "selected_scale":
            1.0,

        "total_matches":
            len(points_A),

        "average_confidence":
            average_confidence,

        "best_confidence":
            best_confidence,

        "lowest_confidence":
            lowest_confidence,

        "ransac_inliers":
            len(inliers_A),

        "ransac_outliers":
            int(
                ransac["outliers"]
            ),

        "inlier_ratio":
            ransac[
                "inlier_ratio"
            ] * 100.0,

        "rmse":
            error["rmse"],

        "mean_error":
            error["mean_error"],

        "maximum_error":
            error[
                "maximum_error"
            ],

        "subpixel_percentage":
            error[
                "subpixel_percentage"
            ],

        "spatial_coverage":
            spatial[
                "coverage"
            ] * 100.0,

        "spatial_uniformity":
            spatial[
                "uniformity"
            ],

        "overall_quality":
            quality_score,

        "ecc_correlation":
            0.0,

        "cell_counts":
            spatial[
                "counts"
            ].tolist(),

        "cell_quality":
            spatial[
                "quality"
            ].tolist(),

        "homography":
            ransac[
                "H"
            ].tolist()
    }

    print("\nAnalysis completed")
    print(
        "Final status:",
        status
    )

    print("=" * 60)

    return result
