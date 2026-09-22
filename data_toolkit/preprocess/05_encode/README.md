# 05_encode

shape → SS → PBR latent encoder를 한 CUDA process에서 순차 실행한다. `--world-size`,
`--rank`와 CUDA device 하나가 필수다.

```bash
CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/05_encode/run.py \
  --world-size 1 --rank 0 \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010
```

batch-closed 실행은 [00_rank](../00_rank/README.md)를 사용하며, PyTorch Dataset/DataLoader 설정은 YAML의
`stages.encode`가 제어한다.

## 1024 shape voxel 수로 두 환경 분할

`--encode-min-shape-1024-voxels`와 `--encode-max-shape-1024-voxels`는 두 view의
1024 shape VXZ header 중 큰 `num_voxel`을 기준으로 asset을 **포함 범위**로 선택한다.
VXZ payload를 압축 해제하지 않고 header만 읽는다. 두 실행의 경계는 겹치지 않게 정한다.
예를 들어 cutoff가 `2000000`이면 local은 `<= 2000000`, n17은 `>= 2000001`이다.

```bash
# n17: 큰 asset. GPU 7 하나만 쓰고 NFS reader는 16개.
CUDA_VISIBLE_DEVICES=7 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/00_rank/run.py \
  --config data_toolkit/preprocess/config/default.yaml \
  --source ObjaverseXL_sketchfab --world-size 1 --rank 0 --stages 05 \
  --encode-min-shape-1024-voxels 2000001 \
  --encode-partition-name n17-high --encode-loader-workers 16

# n7 local: VRAM-safe asset. 각 GPU process는 자신의 batch modulo rank만 처리한다.
CUDA_VISIBLE_DEVICES=0 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/00_rank/run.py \
  --config data_toolkit/preprocess/config/default.yaml \
  --source ObjaverseXL_sketchfab --world-size 6 --rank 0 --stages 05 \
  --encode-max-shape-1024-voxels 2000000 \
  --encode-partition-name n7-low
```

`rank=0..5`는 local GPU 0..5에 각각 실행한다. partition의 manifest와 stage 정보는
`05_encode/partitions/<name>/`에 남고, 실제 latent은 기존 `05_encode/{shape,pbr,ss}`에
asset별 원자 publish된다. 따라서 두 partition은 동시 실행 가능하다. `06_finalize`는
partition manifest가 아니라 04의 전체 성공 asset을 기준으로 모든 latent family/view 파일을
검증하므로, 양쪽 partition이 모두 끝난 뒤에만 실행해야 한다.
