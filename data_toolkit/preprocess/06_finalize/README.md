# 06_finalize

05의 shape/SS/PBR latent 트리를 검증한다. `--publish`를 주면 검증된 batch 결과를 별도 `prepared-v2` 경로로 publish한다.

## 단일 batch

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/06_finalize/run.py \
  --shard ObjaverseXL_sketchfab-00000 --batch batch000 --publish
```

## 전체 batch 4병렬 예시

batch마다 publish target이 분리되어 있어 batch-level 병렬 실행이 가능하다.

```bash
find /home/rvi/ns2/youngwoo/pixal3d/control/shards/ObjaverseXL_sketchfab -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 -P4 bash -c '
      batch_file="$1"
      shard="$(basename "$(dirname "$batch_file")")"
      batch="$(basename "$batch_file" .txt)"
      exec /home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/06_finalize/run.py \
        --shard "$shard" --batch "$batch" --publish
    ' _
```

기본 publish target은 `/home/rvi/ns2/youngwoo/pixal3d/prepared-v2/<source>/<shard>/<batch>`이고, `06_finalize/report.json`에 결과를 남긴다.
