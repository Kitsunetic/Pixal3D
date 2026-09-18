# 01_manifest

control shard의 `batch*.txt`에 든 asset SHA-256을 ns3 full Objaverse archive의 direct GLB 또는 7z member로 해석해 `manifest.jsonl`을 만든다. GPU를 사용하지 않는다.

## 단일 batch

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/01_manifest/run.py \
  --build-index --shard ObjaverseXL_sketchfab-00000 --batch batch000
```

`--build-index`는 raw source index가 없거나 raw 설정이 바뀌었을 때 한 번만 사용한다. 기본 설정은 `/home/rvi/ns3/jaehyeok/ds/Objaverse-full`을 사용한다.

## 전체 batch 16병렬

아래 명령으로 673개 batch의 manifest를 생성했다. `batch*.txt`는 control shard에 이미 존재하는 입력이며 이 단계가 생성하지 않는다.

```bash
find /home/rvi/ns2/youngwoo/pixal3d/control/shards/ObjaverseXL_sketchfab -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 -P16 bash -c '
      batch_file="$1"
      shard="$(basename "$(dirname "$batch_file")")"
      batch="$(basename "$batch_file" .txt)"
      exec /home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/01_manifest/run.py \
        --shard "$shard" --batch "$batch"
    ' _
```

출력은 `data/preprocess_v2/<source>/<shard>/<batch>/01_manifest/manifest.jsonl`이다.
