# 02_dump

embedded `bpy` import 한 번으로 mesh/PBR dump를 함께 만든다. 외부 Blender는 사용하지 않고,
texture extraction까지 CPU-only다. manifest가 가리키는 압축 해제된 direct GLB를 바로 읽으며
7z를 호출하지 않는다. `--world-size`, `--rank`가 필수다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/02_dump/run.py \
  --world-size 1 --rank 0 \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010
```

rank 병렬 실행은 [00_rank](../00_rank/README.md)의 launcher를 사용한다. shared mesh/PBR pickle이
둘 다 있으면 asset은 재처리하지 않으며, legacy 또는 v2 완료 batch는 즉시 skip한다.
