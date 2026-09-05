# ================================================================
# 🌕 LUNAR IMAGE CORRESPONDENCE AND REGISTRATION SYSTEM
# FINAL FLASK BACKEND
# SIH26166 - Chandrayaan-2
# ================================================================

import os
import uuid

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

from lunar_backend import analyze_lunar_images


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

RESULTS_FOLDER = os.path.join(
    BASE_DIR,
    "lunar_results"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# ================================================================
# FLASK
# ================================================================

app = Flask(__name__)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ================================================================
# ALLOWED FILES
# ================================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff"
}


def allowed_file(filename):

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ================================================================
# 🏠 HOME
# ================================================================

@app.route("/")
def home():

    return jsonify({
        "system":
            "Lunar Image Correspondence and Registration System",

        "problem_statement":
            "SIH26166",

        "status":
            "online",

        "backend":
            "LoFTR + RANSAC + Registration + Quality Analysis"
    })


# ================================================================
# ❤️ HEALTH CHECK
# ================================================================

@app.route("/status")
def status():

    return jsonify({

        "flask":
            "online",

        "ai_backend":
            "ready",

        "system":
            "Lunar Image Correspondence and Registration System",

        "pipeline": [

            "Image Acquisition",

            "Illumination-Robust Preprocessing",

            "Multi-Scale LoFTR",

            "Confidence Filtering",

            "RANSAC Verification",

            "Registration Refinement",

            "Spatial Analysis",

            "Quality Evaluation"

        ]

    })


# ================================================================
# 🧠 MAIN AI ANALYSIS API
# ================================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    print("\n" + "=" * 70)

    print(
        "🌕 NEW LUNAR ANALYSIS REQUEST"
    )

    print("=" * 70)


    try:

        # --------------------------------------------------------
        # CHECK IMAGE A
        # --------------------------------------------------------

        if "image_A" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "Reference Image A was not uploaded."

            }), 400


        # --------------------------------------------------------
        # CHECK IMAGE B
        # --------------------------------------------------------

        if "image_B" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "Target Image B was not uploaded."

            }), 400


        image_A = request.files[
            "image_A"
        ]

        image_B = request.files[
            "image_B"
        ]


        # --------------------------------------------------------
        # CHECK FILENAMES
        # --------------------------------------------------------

        if image_A.filename == "":

            return jsonify({

                "success": False,

                "error":
                    "Image A has no filename."

            }), 400


        if image_B.filename == "":

            return jsonify({

                "success": False,

                "error":
                    "Image B has no filename."

            }), 400


        # --------------------------------------------------------
        # CHECK FILE TYPES
        # --------------------------------------------------------

        if not allowed_file(
            image_A.filename
        ):

            return jsonify({

                "success": False,

                "error":
                    "Unsupported Image A format."

            }), 400


        if not allowed_file(
            image_B.filename
        ):

            return jsonify({

                "success": False,

                "error":
                    "Unsupported Image B format."

            }), 400


        # ========================================================
        # UNIQUE REQUEST ID
        # ========================================================

        request_id = uuid.uuid4().hex[:8]


        extension_A = os.path.splitext(
            image_A.filename
        )[1].lower()

        extension_B = os.path.splitext(
            image_B.filename
        )[1].lower()


        file_A_path = os.path.join(
            UPLOAD_FOLDER,
            "image_A_" +
            request_id +
            extension_A
        )


        file_B_path = os.path.join(
            UPLOAD_FOLDER,
            "image_B_" +
            request_id +
            extension_B
        )


        # ========================================================
        # SAVE UPLOADS
        # ========================================================

        image_A.save(
            file_A_path
        )

        image_B.save(
            file_B_path
        )


        print(
            "\n📥 Image A received:"
        )

        print(
            file_A_path
        )


        print(
            "\n📥 Image B received:"
        )

        print(
            file_B_path
        )


        # ========================================================
        # 🧠 RUN OUR ACTUAL AI BACKEND
        # ========================================================

        print(
            "\n🧠 Starting AI processing..."
        )

        results = analyze_lunar_images(

            file_A_path,

            file_B_path,

            output_folder=RESULTS_FOLDER

        )


        # ========================================================
        # 🖼️ RESULT IMAGE URLs
        # ========================================================

        results["images"] = {

            "preprocessed_A":
                "/results/preprocessed_A.png",

            "preprocessed_B":
                "/results/preprocessed_B.png",

            "ransac_inliers":
                "/results/ransac_inliers.png",

            "registered_refined":
                "/results/registered_refined.png",

            "correspondence_quality_map":
                "/results/correspondence_quality_map.png",

            "spatial_distribution":
                "/results/spatial_distribution.png"

        }


        # ========================================================
        # 📊 FINAL RESPONSE
        # ========================================================

        response = {

            "success":
                True,

            "message":
                "Lunar image analysis completed successfully.",

            "request_id":
                request_id,

            "results":
                results

        }


        print(
            "\n✅ AI analysis completed."
        )

        print(
            "📊 Results sent to frontend."
        )


        print("\n" + "=" * 70)


        return jsonify(
            response
        )


    except Exception as e:

        print(
            "\n❌ ERROR:"
        )

        print(
            str(e)
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ================================================================
# 🖼️ SERVE RESULT IMAGES
# ================================================================

@app.route(
    "/results/<filename>"
)
def result_image(filename):

    return send_from_directory(

        RESULTS_FOLDER,

        filename

    )


# ================================================================
# 📂 SERVE UPLOADED IMAGES
# ================================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_image(filename):

    return send_from_directory(

        UPLOAD_FOLDER,

        filename

    )


# ================================================================
# 🚦 ERROR HANDLER
# ================================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return jsonify({

        "success":
            False,

        "error":
            "Uploaded file is too large. Maximum size is 100 MB."

    }), 413


# ================================================================
# 🚀 START SERVER
# ================================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 70)

    print(
        "🌕 LUNAR IMAGE CORRESPONDENCE AND REGISTRATION SYSTEM"
    )

    print("=" * 70)

    print(
        "\n🧠 AI Backend:"
    )

    print(
        "   LoFTR + RANSAC + Registration + Quality Analysis"
    )

    print(
        "\n🌐 Flask Server:"
    )

    print(
        "   http://127.0.0.1:5000"
    )

    print(
        "\n❤️ Status:"
    )

    print(
        "   http://127.0.0.1:5000/status"
    )

    print(
        "\n📡 Analysis API:"
    )

    print(
        "   POST /analyze"
    )

    print(
        "\n🌕 Waiting for frontend..."
    )

    print("=" * 70)

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )
