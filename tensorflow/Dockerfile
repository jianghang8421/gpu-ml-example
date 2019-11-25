FROM tensorflow/tensorflow:latest-gpu

COPY classification.py /
COPY loaddata.py /
COPY entrypoint.sh /
ENV RANCHER_COM_GPU_MEM_DEV 1.0
ENV RANCHER_COM_GPU_MEM_CONTAINER 1.0
RUN python /loaddata.py
ENTRYPOINT [ "/entrypoint.sh" ]