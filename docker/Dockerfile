FROM python:3.9-slim-buster 


# Install packages

RUN pip3 install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl  \
https://download.pytorch.org/whl/cpu/torchvision-0.12.0%2Bcpu-cp39-cp39-linux_x86_64.whl \
timm  \
click

WORKDIR /opt/workspace


# Copy the source code

ADD . .

RUN pip3 install --no-cache-dir -r requirements.deployment.txt