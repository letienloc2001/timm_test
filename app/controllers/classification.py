from flask import Blueprint, jsonify, make_response, request, current_app
from flask.wrappers import Response

from itertools import chain

classification_bp = Blueprint('classification_bp', __name__)


@classification_bp.route("/infer_anti_spoofing", methods=["POST"])
def infer_anti_spoofing_image() -> Response:
	"""
	Controller Anti-Spoofing Image

	Returns:
		object: response object containing anti-spoofing information.
	"""

	images = request.json.get("username", None)
    
	if not images:
		make_response(jsonify({"message": "Error missing images"}), 409)

	nested_result = current_app.models["ANTI_SPOOFING_CLASSIFICATION"].infer(images)
	result = list(chain(*nested_result))

	prediction = []
	for value in result:
		prediction.append({
			"image_id": "IMAGE_ID",
			"is_spoof_image": int(value[0]),
		})
	return make_response(jsonify(prediction), 200)

@classification_bp.route("/train_anti_spoofing", methods=["POST"])
def train_anti_spoofing() -> Response:
	"""
	Train anti-spoofing model

	Returns:
		response object containing anti-spoofing training information.
	"""

	dataset_path = request.json.get("dataset_path", None)
	if not dataset_path:
		make_response(jsonify({"message": "Error missing dataset_path"}), 409)

	current_app.models["ANTI_SPOOFING_CLASSIFICATION"].train(dataset_path)
	return make_response(jsonify({"message": "Training Completed."}), 200)
