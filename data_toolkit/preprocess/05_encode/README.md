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
