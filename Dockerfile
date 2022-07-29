# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvcr.io/nvidia/pytorch:22.06-py3
RUN rm -rf /opt/pytorch

EXPOSE 8080

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update pip
RUN pip install --upgrade pip
RUN apt-get update
RUN apt install --no-install-recommends -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 vim git zip htop screen

# Install pip packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip wheel
RUN pip uninstall -y Pillow torchtext  # torch torchvision
RUN pip install --no-cache -r requirements.txt albumentations Pillow>=9.1.0 \
    'opencv-python<4.6.0.66' \
    --extra-index-url https://download.pytorch.org/whl/cu110

WORKDIR /tessel_coreai
COPY . /tessel_coreai

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "run:create_app()"]