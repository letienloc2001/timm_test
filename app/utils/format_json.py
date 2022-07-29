import numpy as np

def format_bbox(idx: int, object_id: int, xyxy: np.array, 
                conf: float, predicted_type: str = "bbox"):
    """Format output of 1 bounding box

    Args:
        idx (int): Box ID
        object_id (int): Object ID
        xyxy (np.array): Top-left (x,y) and bottom-right (x,y) coordinates of this bounding box
        conf (float): Confidence of this bounding box
        predicted_type (str, optional): The type this bounding box predicted. Defaults to "bbox".

    Returns:
        json: Response information about this bounding box in formatted type
    """
    
    return {
        "id": idx, 
        "object_id": object_id,
        "x1": float(xyxy[0]), 
        "y1": float(xyxy[1]), 
        "x2": float(xyxy[2]), 
        "y2": float(xyxy[3]), 
        "probability": float(conf), 
        "type": predicted_type, 
    }