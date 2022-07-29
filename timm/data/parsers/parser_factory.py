import os

from .parser_image_folder import ParserImageFolder
# from .parser_image_tar import ParserImageTar
from .parser_image_in_tar import ParserImageInTar
from app.utils.read_images import convert_from_base64

def create_parser(name, root, split='train', **kwargs):
    if isinstance(root, str) and os.path.exists(root):
        if os.path.isfile(root) and os.path.splitext(root)[1] == '.tar':
            parser = ParserImageInTar(root, **kwargs)
        else:
            parser = ParserImageFolder(root, **kwargs)
    else:
        # Otherwise it's a base64-format string
        parser = [(convert_from_base64(image), None) for image in root]
    return parser
