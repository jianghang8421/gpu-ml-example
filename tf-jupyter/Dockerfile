FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

RUN mkdir -p /tf/rancher
COPY rancher-classification.ipynb /tf/rancher/
COPY loaddata.py /
ENV RANCHER_COM_GPU_MEM_DEV 1.0
ENV RANCHER_COM_GPU_MEM_CONTAINER 1.0
RUN python /loaddata.py