# 04_voxelize

02의 mesh/PBR dump와 03의 camera transform을 읽어 dual-grid 및 PBR voxel view를 만든다. GPU 선택 옵션은 없으며 legacy voxel 수학 함수를 batch 안에서 순차 호출한다.

## 단일 batch

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/04_voxelize/run.py \
  --shard ObjaverseXL_sketchfab-00000 --batch batch000
```

## 전체 batch 4병렬 예시

```bash
find /home/rvi/ns2/youngwoo/pixal3d/control/shards/ObjaverseXL_sketchfab -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 -P4 bash -c '
      batch_file="$1"
      shard="$(basename "$(dirname "$batch_file")")"
      batch="$(basename "$batch_file" .txt)"
      exec /home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/04_voxelize/run.py \
        --shard "$shard" --batch "$batch"
    ' _
```

해상도, 대상 view, native thread 수는 `config/default.yaml`의 `stages.voxelize`에서 변경한다. 출력은 `04_voxelize/`이다.
