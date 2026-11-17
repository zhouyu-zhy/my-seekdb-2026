# MiniVectorDB - build three kernel variants from source, then run the widest
# one the host CPU supports (selected by docker-entrypoint.sh).
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

COPY --from=build /src/minivectordb /app/minivectordb
COPY --from=build /src/minivectordb-avx2 /app/minivectordb-avx2
COPY --from=build /src/minivectordb-avx512 /app/minivectordb-avx512
COPY --from=build /src/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod 0755 /app/docker-entrypoint.sh \
    /app/minivectordb /app/minivectordb-avx2 /app/minivectordb-avx512

WORKDIR /app
VOLUME ["/data"]

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--data-dir", "/data"]
