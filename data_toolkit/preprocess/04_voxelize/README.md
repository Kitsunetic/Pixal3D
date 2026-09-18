# 04_voxelize

shared 02 dump와 03 render를 읽어 dual-grid/PBR voxel view를 생성한다. GPU 선택 옵션은 없고
legacy voxel 수학을 batch 안에서 순차 호출한다. `--world-size`, `--rank`가 필수다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/04_voxelize/run.py \
  --world-size 1 --rank 0 \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010
```

다른 node가 생성한 shared 02/03 산출물도 재사용할 수 있다. 전체 흐름은 [00_rank](../00_rank/README.md)에서 실행한다.
