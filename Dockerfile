# MiniVectorDB evaluation image.
#
# The judge starts this container with an interactive stdin, no network and a
# writable /data mount.  Everything the program needs is baked in at build time
# and pinned:
#   * base image pinned by digest,
#   * Debian packages pinned by version,
#   * python3-numpy is optional at runtime (the engine falls back to a pure
#     Python scan if the module is missing), so no pip download is required.

FROM ubuntu:24.04@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    TZ=UTC

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3=3.12.3-0ubuntu2.1 \
        python3-numpy=1:1.26.4+ds-6ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ /app/src/

RUN mkdir -p /data

VOLUME ["/data"]

ENTRYPOINT ["python3", "-m", "minivectordb"]
CMD ["--data-dir", "/data"]
