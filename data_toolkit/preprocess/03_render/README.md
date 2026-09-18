# 03_render

압축 해제된 원본 direct GLB를 embedded `bpy`/Cycles/OPTIX로 조건 뷰 8장으로 렌더링한다. 7z를
호출하지 않는다. `--world-size`,
`--rank`와 CUDA device 하나가 필수다.

```bash
CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/03_render/run.py \
  --world-size 1 --rank 0 \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010
```

batch-closed rank 실행에서는 [00_rank](../00_rank/README.md)를 사용한다.
