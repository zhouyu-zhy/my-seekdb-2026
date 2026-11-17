# MiniVectorDB - build the kernel variants from source, then run the widest one
# the host CPU supports (selected by docker-entrypoint.sh).  The AVX-512 target
# is optional, so a toolchain without those flags still produces a working
# image; the wildcard copy tolerates its absence.
FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY minivectordb/ /src/
RUN make -j"$(nproc)"

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /data

COPY --from=build /src/minivectordb* /app/
COPY --from=build /src/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh \
    && chmod 0755 /app/docker-entrypoint.sh && chmod 0755 /app/minivectordb*

WORKDIR /app
VOLUME ["/data"]

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--data-dir", "/data"]
