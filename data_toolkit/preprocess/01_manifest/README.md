# 01_manifest

control batch의 SHA-256을 이미 압축 해제된 `glbs_direct`의 raw GLB로 해석한다. GPU를 사용하지 않으며
`--world-size`, `--rank`가 필수다. 소유하지 않은 batch는 실행을 거부한다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/01_manifest/run.py \
  --world-size 1 --rank 0 --build-index \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010
```

여러 rank와 node가 동시에 시작해도 `--build-index`는 shared file lock으로 한 번만 index를
생성한다. legacy NS2 prepared 또는 completion marker가 있는 NS3 prepared-v2 batch는 빈 manifest를 남기고 skip한다.

과거 archive 기반 index가 이미 있으면 `01_manifest`는 7z로 되돌아가지 않고 오류로 종료한다.
`extract_sketchfab.py`가 끝난 뒤에는 shared index를 다음처럼 한 번 재생성한다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/01_manifest/run.py \
  --world-size 1 --rank 0 --rebuild-index \
  --shard ObjaverseXL_sketchfab-00000 --batch batch000
```
