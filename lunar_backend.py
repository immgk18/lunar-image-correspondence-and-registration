# ================================================================
# 🌕 LUNAR IMAGE CORRESPONDENCE AND REGISTRATION SYSTEM
# SIH26166 - Chandrayaan-2
#
# SERVER-OPTIMIZED AI BACKEND
#
# Pipeline:
# Input
#   ↓
# Illumination-Robust Preprocessing
#   ↓
# LoFTR Correspondence
#   ↓
# Confidence Filtering
#   ↓
# Duplicate Removal
#   ↓
# RANSAC Verification
#   ↓
# Registration Refinement
#   ↓
# Spatial Quality Analysis
#   ↓
# Final Metrics
# ================================================================

import os

# ---------------------------------------------------------------
# Limit CPU thread creation before loading PyTorch
# ---------------------------------------------------------------

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

from kornia.feature import LoFTR


# ================================================================
# CONFIGURATION
# ================================================================

# IMPORTANT:
# Reduced from 640x384 to 320x192 for low-memory deployment.
#
# 320 and 192 are both divisible by 8, which is suitable for
# LoFTR's feature pyramid.
TARGET_WIDTH = 320
TARGET_HEIGHT = 192

# Server mode:
# Use one scale to avoid running LoFTR twice.
SCALE = 1.0

# LoFTR confidence threshold
CONFIDENCE_THRESHOLD = 0.30

# Minimum number of matches needed for RANSAC
MIN_MATCHES = 8

# RANSAC reprojection threshold
RANSAC_THRESHOLD = 5.0

# Spatial quality grid
GRID_ROWS = 4
GRID_COLS = 4

# Maximum matches drawn in visualization
MAX_VISUAL_MATCHES = 100


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device("cpu")

# Keep PyTorch CPU thread usage small on the Render instance.
try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


print("\n" + "=" * 70)
print("🌕 LUNAR IMAGE CORRESPONDENCE AND REGISTRATION SYSTEM")
print("=" * 70)

print("🖥️ Processing device:", DEVICE)
print("📐 Server image size:",
      TARGET_WIDTH, "x", TARGET_HEIGHT)


# ================================================================
# LAZY LoFTR MODEL
# ================================================================
#
# IMPORTANT:
# The model is NOT loaded while the module is imported.
#
# This allows Flask /status to start without immediately loading
# the heavy LoFTR model.
# ================================================================

matcher = None


def get_matcher():
    global matcher

    if matcher is None:

        print("\n🤖 Loading LoFTR for first analysis...")

        matcher = LoFTR(
            pretrained="outdoor"
        )

        matcher = matcher.to(DEVICE)

        matcher.eval()

        print("✅ LoFTR loaded successfully")

    return matcher


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def preprocess_image(image):

    if image is None:
        raise ValueError(
            "Input image could not be loaded."
        )

    # ------------------------------------------------------------
    # Remove alpha channel
    # ------------------------------------------------------------

    if (
        len(image.shape) == 3
        and image.shape[2] == 4
    ):
        image = image[:, :, :3]

    # ------------------------------------------------------------
    # Convert to grayscale
    # ------------------------------------------------------------

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()

    # ------------------------------------------------------------
    # Convert to uint8
    # ------------------------------------------------------------

    if gray.dtype != np.uint8:

        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    # ------------------------------------------------------------
    # CLAHE
    # ------------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # ------------------------------------------------------------
    # Mild denoising
    # ------------------------------------------------------------

    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    # ------------------------------------------------------------
    # Final normalization
    # ------------------------------------------------------------

    normalized = cv2.normalize(
        enhanced,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized.astype(np.uint8)


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
# IMAGE → TENSOR
# ================================================================

def image_to_tensor(image):

    tensor = (
        torch.from_numpy(
            image
        )
        .float()
        .div(255.0)
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    return tensor


# ================================================================
# LoFTR MATCHING
# ================================================================

def run_loftr(
    image_A,
    image_B
):

    model = get_matcher()

    scaled_A = resize_image(
        image_A
    )

    scaled_B = resize_image(
        image_B
    )

    tensor_A = image_to_tensor(
        scaled_A
    )

    tensor_B = image_to_tensor(
        scaled_B
    )

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

        # Release temporary tensors.
        del tensor_A
        del tensor_B

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

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
# REMOVE DUPLICATE MATCHES
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

    indices = np.sort(
        indices
    )

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

    inliers_A = (
        points_A[mask]
    )

    inliers_B = (
        points_B[mask]
    )

    inlier_confidence = (
        confidence[mask]
    )

    outliers = (
        len(points_A)
        - len(inliers_A)
    )

    inlier_ratio = (
        len(inliers_A)
        / float(len(points_A))
    )

    return {

        "H": H,

        "mask": mask,

        "inliers_A":
            inliers_A,

        "inliers_B":
            inliers_B,

        "inlier_confidence":
            inlier_confidence,

        "outliers":
            int(outliers),

        "inlier_ratio":
            float(inlier_ratio)
    }


# ================================================================
# REGISTRATION ERROR
# ================================================================

def calculate_registration_error(
    H,
    inliers_A,
    inliers_B
):

    if len(inliers_A) == 0:

        raise ValueError(
            "No RANSAC inliers available."
        )

    projected = cv2.perspectiveTransform(
        inliers_A.reshape(
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
        projected
        - inliers_B,
        axis=1
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    mean_error = float(
        errors.mean()
    )

    maximum_error = float(
        errors.max()
    )

    subpixel_count = int(
        np.sum(
            errors < 1.0
        )
    )

    subpixel_percentage = (
        subpixel_count
        / float(len(errors))
        * 100.0
    )

    return {

        "errors":
            errors,

        "rmse":
            rmse,

        "mean_error":
            mean_error,

        "maximum_error":
            maximum_error,

        "subpixel_percentage":
            float(
                subpixel_percentage
            )
    }


# ================================================================
# SPATIAL ANALYSIS
# ================================================================

def analyze_spatial_distribution(
    points,
    confidence,
    width,
    height
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
            / float(width)
            * GRID_COLS
        )

        row = int(
            y
            / float(height)
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

        quality[row, col] += float(
            conf
        )

    for r in range(
        GRID_ROWS
    ):

        for c in range(
            GRID_COLS
        ):

            if counts[r, c] > 0:

                quality[r, c] /= (
                    counts[r, c]
                )

    occupied = int(
        np.count_nonzero(
            counts
        )
    )

    total_cells = (
        GRID_ROWS
        * GRID_COLS
    )

    coverage = (
        occupied
        / float(total_cells)
    )

    nonzero_counts = (
        counts[counts > 0]
    )

    if len(
        nonzero_counts
    ) > 1:

        mean_count = float(
            nonzero_counts.mean()
        )

        std_count = float(
            nonzero_counts.std()
        )

        if mean_count > 0:

            coefficient_variation = (
                std_count
                / mean_count
            )

            uniformity = (
                1.0
                / (
                    1.0
                    + coefficient_variation
                )
            )

        else:

            uniformity = 0.0

    elif len(
        nonzero_counts
    ) == 1:

        uniformity = 1.0

    else:

        uniformity = 0.0

    return {

        "counts":
            counts,

        "quality":
            quality,

        "coverage":
            float(coverage),

        "uniformity":
            float(uniformity)
    }


# ================================================================
# QUALITY SCORE
# ================================================================

def calculate_quality_score(
    inlier_ratio,
    average_confidence,
    spatial_coverage,
    spatial_uniformity,
    rmse
):

    error_score = max(
        0.0,
        1.0
        - min(
            rmse / 10.0,
            1.0
        )
    )

    score = (

        0.30
        * inlier_ratio

        +

        0.25
        * average_confidence

        +

        0.20
        * spatial_coverage

        +

        0.15
        * spatial_uniformity

        +

        0.10
        * error_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ================================================================
# REGISTRATION REFINEMENT
# ================================================================

def refine_registration(
    image_A,
    image_B,
    H
):

    base_A = resize_image(
        image_A
    )

    base_B = resize_image(
        image_B
    )

    warped = cv2.warpPerspective(
        base_A,
        H,
        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        )
    )

    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        30,
        1e-4
    )

    try:

        correlation, warp_matrix = (
            cv2.findTransformECC(

                base_B.astype(
                    np.float32
                ) / 255.0,

                warped.astype(
                    np.float32
                ) / 255.0,

                warp_matrix,

                cv2.MOTION_AFFINE,

                criteria
            )
        )

        refined = cv2.warpAffine(
            warped,
            warp_matrix,
            (
                TARGET_WIDTH,
                TARGET_HEIGHT
            ),
            flags=cv2.INTER_LINEAR
        )

        return (
            refined,
            float(correlation)
        )

    except Exception:

        return (
            warped,
            0.0
        )


# ================================================================
# RANSAC VISUALIZATION
# ================================================================

def create_ransac_visualization(
    image_A,
    image_B,
    inliers_A,
    inliers_B,
    confidence
):

    display_A = cv2.cvtColor(
        image_A,
        cv2.COLOR_GRAY2BGR
    )

    display_B = cv2.cvtColor(
        image_B,
        cv2.COLOR_GRAY2BGR
    )

    combined = np.hstack(
        (
            display_A,
            display_B
        )
    )

    width = (
        display_A.shape[1]
    )

    number = min(
        MAX_VISUAL_MATCHES,
        len(inliers_A)
    )

    if number > 0:

        indices = np.argsort(
            confidence
        )[::-1][:number]

        for idx in indices:

            x1 = int(
                inliers_A[idx][0]
            )

            y1 = int(
                inliers_A[idx][1]
            )

            x2 = int(
                inliers_B[idx][0]
            ) + width

            y2 = int(
                inliers_B[idx][1]
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

    height, width = (
        image.shape
    )

    normalized = (
        np.clip(
            cell_quality,
            0,
            1
        )
        * 255
    ).astype(
        np.uint8
    )

    heatmap_small = cv2.resize(
        normalized,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    heatmap = cv2.applyColorMap(
        heatmap_small,
        cv2.COLORMAP_PLASMA
    )

    original = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    overlay = cv2.addWeighted(
        original,
        0.45,
        heatmap,
        0.55,
        0
    )

    for r in range(
        GRID_ROWS + 1
    ):

        y = int(
            r
            * height
            / GRID_ROWS
        )

        cv2.line(
            overlay,
            (0, y),
            (width, y),
            (255, 255, 255),
            1
        )

    for c in range(
        GRID_COLS + 1
    ):

        x = int(
            c
            * width
            / GRID_COLS
        )

        cv2.line(
            overlay,
            (x, 0),
            (x, height),
            (255, 255, 255),
            1
        )

    return overlay


# ================================================================
# SPATIAL DISTRIBUTION VISUALIZATION
# ================================================================

def create_spatial_visualization(
    image,
    points
):

    canvas = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    height, width = (
        image.shape
    )

    for r in range(
        GRID_ROWS + 1
    ):

        y = int(
            r
            * height
            / GRID_ROWS
        )

        cv2.line(
            canvas,
            (0, y),
            (width, y),
            (255, 255, 255),
            1
        )

    for c in range(
        GRID_COLS + 1
    ):

        x = int(
            c
            * width
            / GRID_COLS
        )

        cv2.line(
            canvas,
            (x, 0),
            (x, height),
            (255, 255, 255),
            1
        )

    for point in points:

        x = int(
            point[0]
        )

        y = int(
            point[1]
        )

        x = min(
            max(x, 0),
            width - 1
        )

        y = min(
            max(y, 0),
            height - 1
        )

        cv2.circle(
            canvas,
            (x, y),
            2,
            (0, 255, 0),
            -1
        )

    return canvas


# ================================================================
# SAVE IMAGE
# ================================================================

def save_image(
    image,
    path
):

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    success = cv2.imwrite(
        path,
        image
    )

    if not success:

        raise IOError(
            "Could not save image: "
            + path
        )


# ================================================================
# COMPLETE ANALYSIS
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

    print("\n" + "=" * 70)
    print("🚀 STARTING LUNAR ANALYSIS")
    print("=" * 70)

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

    print("✅ Images loaded")

    print(
        "Image A:",
        image_A.shape
    )

    print(
        "Image B:",
        image_B.shape
    )

    # ------------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------------

    print(
        "\n🧹 Preprocessing..."
    )

    processed_A = (
        preprocess_image(
            image_A
        )
    )

    processed_B = (
        preprocess_image(
            image_B
        )
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

    # ------------------------------------------------------------
    # LoFTR
    # ------------------------------------------------------------

    print(
        "\n🤖 Running LoFTR..."
    )

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
    # CONFIDENCE FILTER
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
    # DUPLICATE REMOVAL
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
            "Not enough reliable LoFTR matches "
            "for RANSAC."
        )

    # ------------------------------------------------------------
    # RANSAC
    # ------------------------------------------------------------

    print(
        "\n📐 Running RANSAC..."
    )

    ransac = run_ransac(
        points_A,
        points_B,
        confidence
    )

    if ransac is None:

        raise RuntimeError(
            "RANSAC could not estimate a valid "
            "geometric transformation."
        )

    print(
        "RANSAC inliers:",
        len(
            ransac["inliers_A"]
        )
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
    # REGISTRATION ERROR
    # ------------------------------------------------------------

    error = (
        calculate_registration_error(
            ransac["H"],
            ransac["inliers_A"],
            ransac["inliers_B"]
        )
    )

    # ------------------------------------------------------------
    # SPATIAL ANALYSIS
    # ------------------------------------------------------------

    spatial = (
        analyze_spatial_distribution(
            ransac["inliers_A"],
            ransac[
                "inlier_confidence"
            ],
            TARGET_WIDTH,
            TARGET_HEIGHT
        )
    )

    # ------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------

    average_confidence = float(
        ransac[
            "inlier_confidence"
        ].mean()
    )

    best_confidence = float(
        ransac[
            "inlier_confidence"
        ].max()
    )

    lowest_confidence = float(
        ransac[
            "inlier_confidence"
        ].min()
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
    # REGISTRATION
    # ------------------------------------------------------------

    print(
        "\n🔧 Refining registration..."
    )

    refined_image, ecc_score = (
        refine_registration(
            processed_A,
            processed_B,
            ransac["H"]
        )
    )

    save_image(
        refined_image,
        os.path.join(
            output_folder,
            "registered_refined.png"
        )
    )

    # ------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------

    base_A = resize_image(
        processed_A
    )

    base_B = resize_image(
        processed_B
    )

    ransac_visual = (
        create_ransac_visualization(
            base_A,
            base_B,
            ransac[
                "inliers_A"
            ],
            ransac[
                "inliers_B"
            ],
            ransac[
                "inlier_confidence"
            ]
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
            base_A,
            spatial[
                "quality"
            ]
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
            base_A,
            ransac[
                "inliers_A"
            ]
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
    # FINAL STATUS
    # ------------------------------------------------------------

    if (
        ransac[
            "inlier_ratio"
        ] >= 0.70

        and error[
            "rmse"
        ] < 5.0

        and spatial[
            "coverage"
        ] >= 0.50
    ):

        status = (
            "HIGH-QUALITY CORRESPONDENCE"
        )

    elif (
        ransac[
            "inlier_ratio"
        ] >= 0.50

        and error[
            "rmse"
        ] < 10.0
    ):

        status = (
            "MODERATE-QUALITY CORRESPONDENCE"
        )

    else:

        status = (
            "LOW-QUALITY CORRESPONDENCE"
        )

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------

    total_matches = len(
        points_A
    )

    inliers = len(
        ransac[
            "inliers_A"
        ]
    )

    outliers = int(
        ransac[
            "outliers"
        ]
    )

    inlier_ratio = float(
        ransac[
            "inlier_ratio"
        ]
    )

    print("\n" + "=" * 70)
    print("🌕 FINAL LUNAR ANALYSIS REPORT")
    print("=" * 70)

    print(
        "\n🔭 Processing scale:",
        SCALE
    )

    print(
        "🔗 Total matches:",
        total_matches
    )

    print(
        "🎯 RANSAC inliers:",
        inliers
    )

    print(
        "❌ RANSAC outliers:",
        outliers
    )

    print(
        "📊 Inlier ratio:",
        round(
            inlier_ratio * 100,
            2
        ),
        "%"
    )

    print(
        "🧠 Average confidence:",
        round(
            average_confidence,
            4
        )
    )

    print(
        "⭐ Best confidence:",
        round(
            best_confidence,
            4
        )
    )

    print(
        "📉 Lowest confidence:",
        round(
            lowest_confidence,
            4
        )
    )

    print(
        "\n📏 Registration RMSE:",
        round(
            error["rmse"],
            4
        ),
        "pixels"
    )

    print(
        "📐 Mean error:",
        round(
            error["mean_error"],
            4
        ),
        "pixels"
    )

    print(
        "⚠️ Maximum error:",
        round(
            error[
                "maximum_error"
            ],
            4
        ),
        "pixels"
    )

    print(
        "🎯 <1 pixel error:",
        round(
            error[
                "subpixel_percentage"
            ],
            2
        ),
        "%"
    )

    print(
        "\n🗺️ Spatial coverage:",
        round(
            spatial[
                "coverage"
            ] * 100,
            2
        ),
        "%"
    )

    print(
        "📍 Spatial uniformity:",
        round(
            spatial[
                "uniformity"
            ],
            4
        )
    )

    print(
        "🧠 Overall quality:",
        round(
            quality_score,
            4
        )
    )

    print(
        "🔧 ECC correlation:",
        round(
            ecc_score,
            5
        )
    )

    print(
        "\n🚦 FINAL STATUS:",
        status
    )

    print("\n" + "=" * 70)

    # ------------------------------------------------------------
    # RETURN RESULT
    # ------------------------------------------------------------

    return {

        "status":
            status,

        "selected_scale":
            SCALE,

        "total_matches":
            total_matches,

        "average_confidence":
            average_confidence,

        "best_confidence":
            best_confidence,

        "lowest_confidence":
            lowest_confidence,

        "ransac_inliers":
            inliers,

        "ransac_outliers":
            outliers,

        "inlier_ratio":
            inlier_ratio * 100,

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
            ] * 100,

        "spatial_uniformity":
            spatial[
                "uniformity"
            ],

        "overall_quality":
            quality_score,

        "ecc_correlation":
            ecc_score,

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
            ].tolist(),

        "output_folder":
            output_folder
    }


# ================================================================
# LOCAL TEST MODE
# ================================================================

if __name__ == "__main__":

    print(
        "\nThis module is intended to be "
        "used through IC.py."
    )

    print(
        "Run the Flask server with:"
    )

    print(
        "python -m gunicorn IC:app"
    )
