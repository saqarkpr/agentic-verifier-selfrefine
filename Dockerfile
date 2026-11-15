# Pinned environment for the post-training and evaluation pipeline.
# Built for reproducibility, not for cluster deployment: it has not been run
# on an HPC system, and the README says so rather than implying otherwise.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.hf \
    PYTHONHASHSEED=0

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Fail fast if the image cannot reproduce the CPU-side results.
RUN cd src && python verifier.py > /dev/null && echo "verifier self-test passed"

CMD ["bash"]
