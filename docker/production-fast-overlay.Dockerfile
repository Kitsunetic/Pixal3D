ARG BASE_IMAGE=f1.unist.info:443/jhenv6@sha256:d373b9e28450f3a79dd917708bc81eaa0cdbb1638da82f310bd4e7721161d4ab
FROM ${BASE_IMAGE}

ARG PIXAL3D_RUNTIME_COMMIT
LABEL org.opencontainers.image.revision=${PIXAL3D_RUNTIME_COMMIT}

USER root
WORKDIR /root/dev/Pixal3D-fast
COPY . .
RUN test -n "${PIXAL3D_RUNTIME_COMMIT}" \
    && printf '%s\n' "${PIXAL3D_RUNTIME_COMMIT}" \
        > /root/dev/Pixal3D-fast/.pixal3d-runtime-commit
RUN /home/rvi/conda/envs/torch/bin/python -m pip install --no-cache-dir \
        -r data_toolkit/requirements-native-renderer.txt \
    && /home/rvi/conda/envs/torch/bin/python -c \
        'import bpy; assert bpy.app.version[:3] == (4, 5, 1)'

CMD ["sleep", "infinity"]
