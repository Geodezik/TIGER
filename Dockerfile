FROM python:3.11-slim

WORKDIR /app
COPY docker_requirements.txt /app/docker_requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r /app/docker_requirements.txt
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs
ENV PYTHONPATH=/app/src
RUN mkdir -p /app/artifacts/model_final
COPY outputs/tiger_encdec/checkpoint-final/ /app/artifacts/model_final/
COPY outputs/beauty_rqkmeans/codes_with_suffix.npy /app/artifacts/codes_with_suffix.npy
COPY outputs/beauty_minilm/item_ids.json /app/artifacts/item_ids.json

ENTRYPOINT ["python", "-m", "src.predict"]
