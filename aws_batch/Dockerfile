FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /workspace/disttrain

RUN pip install wheel
RUN python -m pip install setuptools==59.5.0
COPY requirements.txt /workspace/disttrain
RUN pip install -r requirements.txt

COPY . /workspace/disttrain


