ARG BASE_IMAGE=f1.unist.info:443/jhenv6:cu128-260813
FROM ${BASE_IMAGE}

ARG PIXAL3D_RUNTIME_COMMIT
LABEL org.opencontainers.image.revision=${PIXAL3D_RUNTIME_COMMIT}

USER root
WORKDIR /root/dev/Pixal3D-fast
COPY . .
RUN test -n "${PIXAL3D_RUNTIME_COMMIT}" \
    && printf '%s\n' "${PIXAL3D_RUNTIME_COMMIT}" \
        > /root/dev/Pixal3D-fast/.pixal3d-runtime-commit

CMD ["sleep", "infinity"]
