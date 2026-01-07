FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs
ENV PYTHONPATH=/app/src
RUN mkdir -p /app/artifacts/model_final
COPY outputs/tiger_encdec/final/ /app/artifacts/model_final/
COPY outputs/beauty_rqkmeans/codes_with_suffix.npy /app/artifacts/codes_with_suffix.npy
COPY outputs/beauty_minilm/item_ids.json /app/artifacts/item_ids.json

ENTRYPOINT ["python", "-m", "src.predict"]
