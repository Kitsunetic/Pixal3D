# 01_manifest

control batch의 SHA-256을 raw GLB 또는 7z member로 해석한다. GPU를 사용하지 않으며
`--world-size`, `--rank`가 필수다. 소유하지 않은 batch는 실행을 거부한다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/01_manifest/run.py \
  --world-size 1 --rank 0 --build-index \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010
```

여러 rank가 같은 node에서 시작해도 `--build-index`는 local file lock으로 한 번만 index를
생성한다. legacy 또는 completion marker가 있는 prepared-v2 batch는 빈 manifest를 남기고 skip한다.
