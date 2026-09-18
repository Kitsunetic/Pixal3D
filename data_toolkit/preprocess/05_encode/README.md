# 05_encode

shape → SS → PBR encoder를 한 CUDA process에서 순차 실행한다. 기존 PyTorch `Dataset`/`DataLoader` 기반 encoder를 그대로 사용한다.

## 단일 batch

```bash
CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/05_encode/run.py \
  --shard ObjaverseXL_sketchfab-00000 --batch batch000
```

GPU 선택은 `CUDA_VISIBLE_DEVICES`로만 한다. loader/saver worker, micro-batch, latent dtype은 `config/default.yaml`의 `stages.encode`에 있다.

## 전체 batch

02와 03과 같이 GPU worker 하나가 batch를 하나씩 순차 처리한다. 04 전체 완료 후 `STAGE=05_encode`으로 실행한다.

```bash
STAGE=05_encode
while IFS= read -r -d '' batch_file; do
  shard="$(basename "$(dirname "$batch_file")")"
  batch="$(basename "$batch_file" .txt)"
  CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
    "data_toolkit/preprocess/$STAGE/run.py" --shard "$shard" --batch "$batch"
done < <(find /home/rvi/ns2/youngwoo/pixal3d/control/shards/ObjaverseXL_sketchfab -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 | sort -z)
```

출력은 `05_encode/{shape,ss,pbr,manifest.jsonl}`이다.
