import numpy as np
import cv2

class ImageQuality:
    """Image Quality Control, including quality illumination and noise. """
    
    def __init__(self, quality_threshold: int = 100):
        self.quality_threshold = quality_threshold
        
    def get_score_quality_image(self, array_image: np.array) -> tuple:
        # compute the Laplacian of the image and then return the focus
	    # measure, which is simply the variance of the Laplacian
        score = cv2.Laplacian(array_image, cv2.CV_64F).var()

        return score, score > self.quality_threshold
