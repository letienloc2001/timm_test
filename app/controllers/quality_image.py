import os
import json

from flask import Blueprint, request, current_app
from flask.wrappers import Response

from app.services.quality_image import ImageQuality
from app.utils.read_images import convert_from_base64, read_images_from_path

quality_image_bp = Blueprint('quality_image_bp', __name__)


@quality_image_bp.route("/quality_control", methods=["POST"])
def verify_quality_image() -> Response:
    """
    Controller Quality Image
    Returns:
        object: response object containing quality control information.
    """
    
    input_json = request.get_json()

    if "images" not in input_json and not input_json["images"]:
        return current_app.response_class(
        response={},
        status=409,
        mimetype='application/json'
    )
        
    images = input_json['images']
    try:
        if isinstance(images, str) and os.path.exists(images):
            raw_images = read_images_from_path(images)
        else:
            raw_images = [convert_from_base64(image) for image in images] 
    except Exception:
        return current_app.response_class(
            response={},
            status=409,
            mimetype='application/json'
        )
    
    image_quality = ImageQuality()
    prediction = [image_quality.get_score_quality_image(raw_image) for raw_image in raw_images]

    result = []
    for output in prediction:
        result.append({
            "image_id": "IMAGE_ID",
            "score": output[0],
            "is_quality_image": bool(output[1]),
        })
        
    return current_app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json',
    )


