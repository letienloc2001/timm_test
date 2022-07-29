import json

from flask import Blueprint, request, current_app
from flask.wrappers import Response

from app.utils.format_json import format_bbox

detection_bp = Blueprint('detection_bp', __name__)


@detection_bp.route("/product_detection", methods=["POST"])
def detect_product() -> Response:
    """
    Controller Product Detection
    Returns:
        object: response object containing localization of products.
    """
    
    input_json = request.get_json()

    if "images" not in input_json and not input_json["images"]:
        return current_app.response_class(
        response={},
        status=409,
        mimetype='application/json',
    )
        
    images = input_json['images']
    prediction = current_app.models["PRODUCT_DETECTION"].infer(images)
    
    result = []
    for det in prediction:
        products = [format_bbox(idx, int(cls), xyxy, float(conf)) \
                    for idx, (*xyxy, conf, cls) in enumerate(reversed(det))]

        result.append({
            "image_id": "IMAGE_ID",
            "products": products,
        }) 

    return current_app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json',
    )

@detection_bp.route("/milkcan_detection", methods=["POST"])
def detect_milkcan() -> Response:
    """
    Controller Milk Can Detection
    Returns:
        object: response object containing localization of milk can.
    """
    
    input_json = request.get_json()

    if "images" not in input_json and not input_json["images"]:
        return current_app.response_class(
        response={},
        status=409,
        mimetype='application/json',
    )
        
    images = input_json['images']
    prediction = current_app.models["CAN_DETECTION"].infer(images)
    
    result = []
    for det in prediction:
        milkcans = [format_bbox(idx, int(cls), xyxy, float(conf)) \
                    for idx, (*xyxy, conf, cls) in enumerate(reversed(det))]

        result.append({
            "image_id": "IMAGE_ID",
            "milkcan": milkcans,
        }) 

    return current_app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json',
    )

@detection_bp.route("/shelf_detection", methods=["POST"])
def detect_shelf() -> Response:
    """
    Controller Shelf Detection
    Returns:
        object: response object containing localization of shelves.
    """
    
    input_json = request.get_json()

    if "images" not in input_json and not input_json["images"]:
        return current_app.response_class(
        response={},
        status=409,
        mimetype='application/json',
    )
        
    images = input_json['images']
    prediction = current_app.models["SHELF_DETECTION"].infer(images)
    
    result = []
    for det in prediction:
        shelves = [format_bbox(idx, int(cls), xyxy, float(conf)) \
                    for idx, (*xyxy, conf, cls) in enumerate(reversed(det))]
        result.append({
            "image_id": "IMAGE_ID",
            "shelves": shelves,
        })
  
    return current_app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json',
    )




