ARG BASE_IMAGE=f1.unist.info:443/jhenv6@sha256:d373b9e28450f3a79dd917708bc81eaa0cdbb1638da82f310bd4e7721161d4ab
FROM ${BASE_IMAGE} AS verified-source

ARG PIXAL3D_RUNTIME_COMMIT
ARG PIXAL3D_RUNTIME_TREE
USER root
WORKDIR /verified-source
COPY . .
RUN test -n "${PIXAL3D_RUNTIME_COMMIT}" \
    && test -n "${PIXAL3D_RUNTIME_TREE}" \
    && test "$(git rev-parse HEAD)" = "${PIXAL3D_RUNTIME_COMMIT}" \
    && test "$(git rev-parse HEAD:data_toolkit)" = "${PIXAL3D_RUNTIME_TREE}" \
    && test -z "$(git status --porcelain=v1 --untracked-files=all)" \
    && rm -rf /verified-source/.git

FROM ${BASE_IMAGE}

ARG PIXAL3D_RUNTIME_COMMIT
ARG PIXAL3D_RUNTIME_TREE
LABEL org.opencontainers.image.revision=${PIXAL3D_RUNTIME_COMMIT}
LABEL org.pixal3d.runtime-tree=${PIXAL3D_RUNTIME_TREE}
USER root
WORKDIR /root/dev/Pixal3D-fast
ENV PYTHONPATH=/root/dev/Pixal3D-fast
ENV PIXAL3D_TOOL_COMMIT=${PIXAL3D_RUNTIME_COMMIT}
COPY --from=verified-source /verified-source/ .
RUN printf '%s\n' "${PIXAL3D_RUNTIME_COMMIT}" > .pixal3d-runtime-commit \
    && printf '%s\n' "${PIXAL3D_RUNTIME_TREE}" > .pixal3d-runtime-tree
RUN /home/rvi/conda/envs/torch/bin/python -m pip install \
        --no-cache-dir --no-deps --require-hashes \
        -r data_toolkit/requirements-native-renderer.txt \
    && /home/rvi/conda/envs/torch/bin/python -c \
        'import bpy; assert bpy.app.version[:3] == (4, 5, 1)'

CMD ["sleep", "infinity"]
