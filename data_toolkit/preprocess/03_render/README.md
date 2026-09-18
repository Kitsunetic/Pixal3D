# 03_render

원본 GLB를 embedded `bpy`/Cycles/OPTIX로 8개 조건 뷰로 렌더링한다. `02_dump`의 성공 manifest를 입력으로 쓰지만 dump pickle은 읽지 않는다.

## 단일 batch

```bash
CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/03_render/run.py \
  --shard ObjaverseXL_sketchfab-00000 --batch batch000
```

기본 해상도와 camera policy는 `config/default.yaml`의 `stages.render` 및 legacy camera contract를 따른다.

## 전체 batch

GPU worker 하나는 batch를 하나씩 순차 처리한다. GPU 5 한 장만 쓸 때에는 다음처럼 `STAGE=03_render`로 02 전체 완료 후 실행한다.

```bash
STAGE=03_render
while IFS= read -r -d '' batch_file; do
  shard="$(basename "$(dirname "$batch_file")")"
  batch="$(basename "$batch_file" .txt)"
  CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
    "data_toolkit/preprocess/$STAGE/run.py" --shard "$shard" --batch "$batch"
done < <(find /home/rvi/ns2/youngwoo/pixal3d/control/shards/ObjaverseXL_sketchfab -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 | sort -z)
```

출력은 `03_render/renders_cond/<asset_id>/`와 `03_render/manifest.jsonl`이다.
