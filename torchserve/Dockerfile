FROM pytorch/torchserve:latest
RUN pip install --no-cache-dir numpy safetensors
WORKDIR /home/model-server
COPY model-store /home/model-server/model-store
COPY torchserve/config.properties /home/model-server/config.properties
CMD ["torchserve", "--start", "--ncs", "--model-store", "model-store", "--models", "mymodel=mymodel.mar", "--ts-config", "config.properties", "--disable-token-auth"]
