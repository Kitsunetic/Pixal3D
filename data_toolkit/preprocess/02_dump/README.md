# 02_dump

한 asset을 embedded `bpy`로 한 번 import하여 PBR dump와 mesh dump를 함께 만든다. ns3 7z raw member는 처리 중에만 `/dev/shm/tmp`로 추출된다. 외부 Blender 실행 파일은 사용하지 않는다.

## 단일 batch

```bash
CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/02_dump/run.py \
  --shard ObjaverseXL_sketchfab-00000 --batch batch000
```

`CUDA_VISIBLE_DEVICES`가 GPU 선택의 유일한 방법이다. 프로세스 내부에서 선택 GPU는 `cuda:0`으로 보인다.

## 전체 batch 8병렬 실행

02는 Blender import/PBR 추출이 CPU 중심이고 CUDA는 texture quantization에만 쓰인다. 현재 실행한 명령은 GPU 5를 8개 worker가 공유하는 형태다. `xargs`가 끝난 batch의 다음 batch를 시작하므로 shared queue나 scheduler는 사용하지 않는다.

```bash
find /home/rvi/ns2/youngwoo/pixal3d/control/shards/ObjaverseXL_sketchfab -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 -P8 bash -c '
      batch_file="$1"
      shard="$(basename "$(dirname "$batch_file")")"
      batch="$(basename "$batch_file" .txt)"
      exec env CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
        data_toolkit/preprocess/02_dump/run.py --shard "$shard" --batch "$batch"
    ' _
```

출력은 `02_dump/{mesh_dumps,pbr_dumps,manifest.jsonl}`이다. 기존 mesh/PBR pickle이 모두 있으면 해당 asset은 건너뛴다.
