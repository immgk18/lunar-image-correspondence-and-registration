# ================================================================
# 🌕 LUNAR IMAGE CORRESPONDENCE AND REGISTRATION SYSTEM
# SIH26166 - Chandrayaan-2 Lunar Image Matching
#
# STRENGTHENED RESEARCH BACKEND
#
# Pipeline:
# Input Validation
#      ↓
# Illumination-Robust Preprocessing
#      ↓
# Multi-Scale LoFTR
#      ↓
# Confidence Filtering
#      ↓
# RANSAC Geometric Verification
#      ↓
# Registration Refinement
#      ↓
# Spatial Uniformity Analysis
#      ↓
# Adaptive Quality Control
#      ↓
# Final Evaluation
# ================================================================

import os
import gc
import cv2
import torch
import kornia
import numpy as np

from kornia.feature import LoFTR


# Keep CPU thread pools small on low-memory hosting.
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ================================================================
# ⚙️ CONFIGURATION
# ================================================================

TARGET_WIDTH = 320
TARGET_HEIGHT = 192

# Multi-scale factors
SCALES = [1.0]

# LoFTR confidence threshold
CONFIDENCE_THRESHOLD = 0.30

# Minimum number of matches
MIN_MATCHES = 8

# RANSAC reprojection threshold
RANSAC_THRESHOLD = 5.0

# Spatial grid
GRID_ROWS = 4
GRID_COLS = 4

# Number of correspondences shown in visualization
MAX_VISUAL_MATCHES = 100


# ================================================================
# 🖥️ DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n" + "=" * 75)
print("🌕 LUNAR IMAGE CORRESPONDENCE AND REGISTRATION SYSTEM")
print("=" * 75)

print("\n🖥️ Processing device:", DEVICE)


# ================================================================
# 🤖 LOAD LoFTR
# ================================================================

print("\n🤖 Loading LoFTR...")

matcher = LoFTR(
    pretrained="outdoor"
)

matcher = matcher.to(DEVICE)

matcher.eval()

print("✅ LoFTR loaded successfully")


# ================================================================
# 🧹 ILLUMINATION-ROBUST PREPROCESSING
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
    #
    # Helps preserve local lunar surface details when
    # illumination differs between images.
    # ------------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # ------------------------------------------------------------
    # Mild Gaussian smoothing
    #
    # Reduces sensor noise while preserving larger features.
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
# 🔄 RESIZE
# ================================================================

def resize_image(image, scale=1.0):

    width = int(
        TARGET_WIDTH * scale
    )

    height = int(
        TARGET_HEIGHT * scale
    )

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )


# ================================================================
# 🔢 IMAGE → TENSOR
# ================================================================

def image_to_tensor(image):

    tensor = (
        torch.from_numpy(image)
        .float()
        / 255.0
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    return tensor


# ================================================================
# 🤖 LoFTR MATCHING
# ================================================================

def run_loftr(
    image_A,
    image_B,
    scale
):

    scaled_A = resize_image(
        image_A,
        scale
    )

    scaled_B = resize_image(
        image_B,
        scale
    )

    tensor_A = image_to_tensor(
        scaled_A
    )

    tensor_B = image_to_tensor(
        scaled_B
    )

    with torch.inference_mode():

        output = matcher(
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

    # Release inference tensors immediately. This matters on
    # 512 MB hosting because LoFTR can create large temporary
    # tensors during CPU inference.
    del output
    del tensor_A
    del tensor_B

    # ------------------------------------------------------------
    # Convert coordinates back to base 320×192 coordinates
    # ------------------------------------------------------------

    if scale != 1.0:

        points_A /= scale
        points_B /= scale

    gc.collect()

    return (
        points_A,
        points_B,
        confidence
    )


# ================================================================
# 🎯 CONFIDENCE FILTER
# ================================================================

def filter_matches(
    points_A,
    points_B,
    confidence,
    threshold
):

    mask = (
        confidence >= threshold
    )

    return (
        points_A[mask],
        points_B[mask],
        confidence[mask]
    )


# ================================================================
# 🔗 REMOVE DUPLICATE / NEAR-DUPLICATE MATCHES
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

    # Round coordinates to prevent repeated
    # matches occupying the exact same pixel region.

    rounded = np.round(
        points_A
    ).astype(int)

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
# 📐 RANSAC
# ================================================================

def run_ransac(
    points_A,
    points_B,
    confidence
):

    if len(points_A) < MIN_MATCHES:

        return None

    H, mask = cv2.findHomography(
        points_A,
        points_B,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:

        return None

    mask = mask.ravel().astype(bool)

    inliers_A = points_A[mask]
    inliers_B = points_B[mask]

    inlier_confidence = confidence[mask]

    outliers = (
        len(points_A)
        - len(inliers_A)
    )

    ratio = (
        len(inliers_A)
        / float(len(points_A))
    )

    return {
        "H": H,
        "mask": mask,
        "inliers_A": inliers_A,
        "inliers_B": inliers_B,
        "inlier_confidence":
            inlier_confidence,
        "outliers": outliers,
        "inlier_ratio": ratio
    }


# ================================================================
# 📏 REGISTRATION ERROR
# ================================================================

def calculate_registration_error(
    H,
    inliers_A,
    inliers_B
):

    projected = cv2.perspectiveTransform(
        inliers_A.reshape(
            -1,
            1,
            2
        ).astype(np.float32),
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - inliers_B,
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

    # ------------------------------------------------------------
    # Sub-pixel percentage
    #
    # Percentage of verified matches with
    # geometric error below 1 pixel.
    #
    # This is an evaluation metric, NOT a claim that
    # the complete system has achieved sub-pixel accuracy.
    # ------------------------------------------------------------

    subpixel_count = np.sum(
        errors < 1.0
    )

    subpixel_percentage = (
        subpixel_count /
        float(len(errors))
        * 100.0
    )

    return {
        "errors": errors,
        "rmse": rmse,
        "mean_error": mean_error,
        "maximum_error":
            maximum_error,
        "subpixel_percentage":
            float(subpixel_percentage)
    }


# ================================================================
# 🗺️ SPATIAL ANALYSIS
# ================================================================

def analyze_spatial_distribution(
    points,
    confidence,
    width,
    height
):

    counts = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=int
    )

    quality = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=float
    )

    # ------------------------------------------------------------
    # Assign every match to a spatial cell
    # ------------------------------------------------------------

    for point, conf in zip(
        points,
        confidence
    ):

        x = point[0]
        y = point[1]

        col = int(
            x / width * GRID_COLS
        )

        row = int(
            y / height * GRID_ROWS
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

        quality[row, col] += conf

    # ------------------------------------------------------------
    # Average confidence per cell
    # ------------------------------------------------------------

    for r in range(GRID_ROWS):

        for c in range(GRID_COLS):

            if counts[r, c] > 0:

                quality[r, c] /= (
                    counts[r, c]
                )

    # ------------------------------------------------------------
    # Spatial coverage
    # ------------------------------------------------------------

    occupied = np.count_nonzero(
        counts
    )

    total_cells = (
        GRID_ROWS *
        GRID_COLS
    )

    coverage = (
        occupied /
        float(total_cells)
    )

    # ------------------------------------------------------------
    # Match distribution uniformity
    #
    # A perfectly uniform distribution would have
    # similar match counts across occupied cells.
    # ------------------------------------------------------------

    nonzero_counts = counts[
        counts > 0
    ]

    if len(nonzero_counts) > 1:

        mean_count = (
            nonzero_counts.mean()
        )

        std_count = (
            nonzero_counts.std()
        )

        if mean_count > 0:

            coefficient_variation = (
                std_count /
                mean_count
            )

            uniformity = (
                1.0 /
                (
                    1.0 +
                    coefficient_variation
                )
            )

        else:

            uniformity = 0.0

    else:

        uniformity = (
            1.0
            if len(nonzero_counts) == 1
            else 0.0
        )

    return {
        "counts": counts,
        "quality": quality,
        "coverage":
            float(coverage),
        "uniformity":
            float(uniformity)
    }


# ================================================================
# 🧠 QUALITY SCORE
# ================================================================

def calculate_quality_score(
    inlier_ratio,
    average_confidence,
    spatial_coverage,
    spatial_uniformity,
    rmse
):

    # ------------------------------------------------------------
    # RMSE component
    #
    # Lower error = better score.
    # ------------------------------------------------------------

    error_score = max(
        0.0,
        1.0 -
        min(rmse / 10.0, 1.0)
    )

    score = (

        0.30 *
        inlier_ratio

        +

        0.25 *
        average_confidence

        +

        0.20 *
        spatial_coverage

        +

        0.15 *
        spatial_uniformity

        +

        0.10 *
        error_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ================================================================
# 🔬 ADAPTIVE MATCHING
# ================================================================

def adaptive_matching(
    image_A,
    image_B
):

    print("\n" + "=" * 75)
    print("🔬 ADAPTIVE MULTI-SCALE LoFTR")
    print("=" * 75)

    candidates = []

    for scale in SCALES:

        print(
            "\n🔍 Testing scale:",
            scale
        )

        try:

            pA, pB, conf = run_loftr(
                image_A,
                image_B,
                scale
            )

            print(
                "   Raw matches:",
                len(pA)
            )

            # ----------------------------------------------------
            # Confidence filtering
            # ----------------------------------------------------

            pA, pB, conf = (
                filter_matches(
                    pA,
                    pB,
                    conf,
                    CONFIDENCE_THRESHOLD
                )
            )

            print(
                "   After confidence filter:",
                len(pA)
            )

            # ----------------------------------------------------
            # Remove duplicates
            # ----------------------------------------------------

            pA, pB, conf = (
                remove_duplicate_matches(
                    pA,
                    pB,
                    conf
                )
            )

            print(
                "   After duplicate removal:",
                len(pA)
            )

            if len(pA) < MIN_MATCHES:

                print(
                    "   ❌ Not enough matches"
                )

                continue

            # ----------------------------------------------------
            # RANSAC
            # ----------------------------------------------------

            ransac = run_ransac(
                pA,
                pB,
                conf
            )

            if ransac is None:

                print(
                    "   ❌ RANSAC failed"
                )

                continue

            # ----------------------------------------------------
            # Registration error
            # ----------------------------------------------------

            error = (
                calculate_registration_error(
                    ransac["H"],
                    ransac["inliers_A"],
                    ransac["inliers_B"]
                )
            )

            spatial = (
                analyze_spatial_distribution(
                    ransac["inliers_A"],
                    ransac["inlier_confidence"],
                    TARGET_WIDTH,
                    TARGET_HEIGHT
                )
            )

            # ----------------------------------------------------
            # Quality
            # ----------------------------------------------------

            average_confidence = float(
                ransac[
                    "inlier_confidence"
                ].mean()
            )

            quality_score = (
                calculate_quality_score(
                    ransac["inlier_ratio"],
                    average_confidence,
                    spatial["coverage"],
                    spatial["uniformity"],
                    error["rmse"]
                )
            )

            candidate = {

                "scale": scale,

                "points_A": pA,

                "points_B": pB,

                "confidence": conf,

                "ransac": ransac,

                "error": error,

                "spatial": spatial,

                "quality_score":
                    quality_score
            }

            candidates.append(
                candidate
            )

            print(
                "   RANSAC inliers:",
                len(
                    ransac["inliers_A"]
                )
            )

            print(
                "   Inlier ratio:",
                round(
                    ransac[
                        "inlier_ratio"
                    ] * 100,
                    2
                ),
                "%"
            )

            print(
                "   RMSE:",
                round(
                    error["rmse"],
                    4
                )
            )

            print(
                "   Quality:",
                round(
                    quality_score,
                    4
                )
            )

        except Exception as e:

            print(
                "   ⚠️ Scale failed:",
                e
            )

    # ------------------------------------------------------------
    # No valid candidate
    # ------------------------------------------------------------

    if len(candidates) == 0:

        raise RuntimeError(
            "No valid LoFTR/RANSAC solution found."
        )

    # ------------------------------------------------------------
    # Select highest quality candidate
    # ------------------------------------------------------------

    best = max(
        candidates,
        key=lambda x:
        x["quality_score"]
    )

    print(
        "\n🏆 Selected scale:",
        best["scale"]
    )

    return best


# ================================================================
# 🌕 REGISTRATION REFINEMENT
# ================================================================

def refine_registration(
    image_A,
    image_B,
    H
):

    warped = cv2.warpPerspective(
        image_A,
        H,
        (
            TARGET_WIDTH,
            TARGET_HEIGHT
        )
    )

    # ------------------------------------------------------------
    # ECC alignment
    #
    # This refines image-level alignment from the RANSAC
    # transformation. It is used as a refinement stage,
    # not as a replacement for geometric verification.
    # ------------------------------------------------------------

    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5
    )

    try:

        correlation, warp_matrix = (
            cv2.findTransformECC(
                image_B.astype(
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

        return refined, float(
            correlation
        )

    except Exception:

        return warped, 0.0


# ================================================================
# 🎨 RANSAC VISUALIZATION
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

    width = image_A.shape[1]

    number = min(
        MAX_VISUAL_MATCHES,
        len(inliers_A)
    )

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
            3,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            combined,
            (x2, y2),
            3,
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
# 🗺️ QUALITY MAP VISUALIZATION
# ================================================================

def create_quality_map(
    image,
    cell_quality
):

    height, width = image.shape

    normalized = (
        np.clip(
            cell_quality,
            0,
            1
        ) * 255
    ).astype(np.uint8)

    heatmap_small = cv2.resize(
        normalized,
        (width, height),
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

    # Grid

    for r in range(
        GRID_ROWS + 1
    ):

        y = int(
            r *
            height /
            GRID_ROWS
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
            c *
            width /
            GRID_COLS
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
# 📍 SPATIAL DISTRIBUTION VISUALIZATION
# ================================================================

def create_spatial_visualization(
    image,
    points
):

    canvas = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    height, width = image.shape

    # Grid

    for r in range(
        GRID_ROWS + 1
    ):

        y = int(
            r *
            height /
            GRID_ROWS
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
            c *
            width /
            GRID_COLS
        )

        cv2.line(
            canvas,
            (x, 0),
            (x, height),
            (255, 255, 255),
            1
        )

    # Points

    for point in points:

        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            canvas,
            (x, y),
            3,
            (0, 255, 0),
            -1
        )

    return canvas


# ================================================================
# 💾 SAVE IMAGE
# ================================================================

def save_image(
    image,
    filename
):

    cv2.imwrite(
        filename,
        image
    )

    print(
        "💾 Saved:",
        filename
    )


# ================================================================
# 🚀 COMPLETE PIPELINE
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

    print("\n" + "=" * 75)
    print("🚀 STARTING COMPLETE LUNAR ANALYSIS")
    print("=" * 75)

    # ============================================================
    # 1. LOAD
    # ============================================================

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

    print(
        "\n✅ Input images loaded"
    )

    print(
        "Image A:",
        image_A.shape
    )

    print(
        "Image B:",
        image_B.shape
    )

    # ============================================================
    # 2. PREPROCESSING
    # ============================================================

    print(
        "\n🧹 Running illumination-robust preprocessing..."
    )

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

    # ============================================================
    # 3. ADAPTIVE MULTI-SCALE MATCHING
    # ============================================================

    best = adaptive_matching(
        processed_A,
        processed_B
    )

    ransac = best["ransac"]
    error = best["error"]
    spatial = best["spatial"]

    # ============================================================
    # 4. REGISTRATION REFINEMENT
    # ============================================================

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

    # ============================================================
    # 5. VISUALIZATIONS
    # ============================================================

    ransac_visual = (
        create_ransac_visualization(
            resize_image(
                processed_A
            ),
            resize_image(
                processed_B
            ),
            ransac["inliers_A"],
            ransac["inliers_B"],
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
            resize_image(
                processed_A
            ),
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
            resize_image(
                processed_A
            ),
            ransac["inliers_A"]
        )
    )

    save_image(
        spatial_visual,
        os.path.join(
            output_folder,
            "spatial_distribution.png"
        )
    )

    # ============================================================
    # 6. FINAL METRICS
    # ============================================================

    total_matches = len(
        best["points_A"]
    )

    average_confidence = float(
        best["confidence"].mean()
    )

    best_confidence = float(
        best["confidence"].max()
    )

    lowest_confidence = float(
        best["confidence"].min()
    )

    inliers = len(
        ransac["inliers_A"]
    )

    outliers = ransac[
        "outliers"
    ]

    inlier_ratio = (
        ransac[
            "inlier_ratio"
        ]
    )

    quality_score = (
        best["quality_score"]
    )

    # ============================================================
    # 7. FINAL STATUS
    # ============================================================

    if (
        inlier_ratio >= 0.70
        and error["rmse"] < 5.0
        and spatial["coverage"] >= 0.50
    ):

        status = (
            "HIGH-QUALITY CORRESPONDENCE"
        )

    elif (
        inlier_ratio >= 0.50
        and error["rmse"] < 10.0
    ):

        status = (
            "MODERATE-QUALITY CORRESPONDENCE"
        )

    else:

        status = (
            "LOW-QUALITY CORRESPONDENCE"
        )

    # ============================================================
    # 8. PRINT FINAL REPORT
    # ============================================================

    print("\n" + "=" * 75)
    print("🌕 FINAL LUNAR ANALYSIS REPORT")
    print("=" * 75)

    print(
        "\n🔭 Selected scale:",
        best["scale"]
    )

    print(
        "🔗 Total LoFTR matches:",
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
            error["maximum_error"],
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
            spatial["coverage"] * 100,
            2
        ),
        "%"
    )

    print(
        "📍 Spatial uniformity:",
        round(
            spatial["uniformity"],
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
        "🔧 ECC refinement correlation:",
        round(
            ecc_score,
            5
        )
    )

    print(
        "\n🚦 FINAL STATUS:",
        status
    )

    print("\n" + "=" * 75)
    print(
        "🌕 LUNAR IMAGE CORRESPONDENCE AND REGISTRATION SYSTEM"
    )
    print("=" * 75)

    # ============================================================
    # 9. RETURN EVERYTHING
    # ============================================================

    return {

        "status": status,

        "selected_scale":
            best["scale"],

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
            error["maximum_error"],

        "subpixel_percentage":
            error[
                "subpixel_percentage"
            ],

        "spatial_coverage":
            spatial["coverage"] * 100,

        "spatial_uniformity":
            spatial["uniformity"],

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
            ransac["H"].tolist(),

        "output_folder":
            output_folder
    }


# ================================================================
# 🧪 TEST MODE
# ================================================================

if __name__ == "__main__":

    print("\n📂 Select Image A")

    from tkinter import (
        Tk,
        filedialog
    )

    root = Tk()

    root.withdraw()

    file_A = (
        filedialog
        .askopenfilename(
            title="Select Lunar Image A",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                )
            ]
        )
    )

    root.destroy()

    if not file_A:

        print(
            "❌ Image A not selected."
        )

        exit()


    print("\n📂 Select Image B")

    root = Tk()

    root.withdraw()

    file_B = (
        filedialog
        .askopenfilename(
            title="Select Lunar Image B",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                )
            ]
        )
    )

    root.destroy()

    if not file_B:

        print(
            "❌ Image B not selected."
        )

        exit()


    # ============================================================
    # RUN
    # ============================================================

    results = analyze_lunar_images(
        file_A,
        file_B
    )

    print(
        "\n✅ ANALYSIS COMPLETE!"
    )
