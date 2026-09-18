# Pixal3D batch preprocessing v2

기존 `data_toolkit/pipeline`은 보존한다. 이 디렉터리는 큐, scheduler, worker를 사용하지 않는 새 전처리 경로다. 각 `run.py`는 **하나의 batch를 순차 처리한 뒤 종료**하며, 병렬 처리는 서로 다른 batch에 대해 같은 명령을 별도 프로세스로 실행한다.

기본 batch 작업 경로는 로컬 NVMe인 다음과 같다.

```text
./data/preprocess_v2/ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000/
```

`/home/rvi/ns2/youngwoo/pixal3d_local/preprocess_v2`는 NFS이므로 작업 중간 산출물에는 사용하지 않는다. 아주 작은 일시 파일만 필요할 때에는 `/dev/shm/tmp`를 사용할 수 있다. 각 단계의 정상 출력은 위 작업 루트의 `01_manifest`부터 `06_finalize`까지에 남는다. `prepared`에는 중간 결과를 쓰지 않으며, 최종 결과만 명시적으로 publish할 때 기존 `prepared`와 분리된 `/home/rvi/ns2/youngwoo/pixal3d/prepared-v2/...`로 복사된다.

## 단계

| 단계 | 역할 | 입력 | 출력 |
| --- | --- | --- | --- |
| `01_manifest` | control batch의 SHA-256을 direct GLB 또는 archive member로 해석 | `control/shards`, metadata, object-path index | `01_manifest/manifest.jsonl` |
| `02_dump` | embedded `bpy`로 한 번 import하여 PBR dump와 mesh dump를 동시에 생성 | manifest, GLB | `02_dump/{mesh_dumps,pbr_dumps}` |
| `03_render` | GLB를 조건 뷰로 렌더링 | manifest, GLB | `03_render/renders_cond` |
| `04_voxelize` | mesh/PBR dump와 camera transform을 voxel로 변환 | dump, render | `04_voxelize/{dual_grid_view_*,pbr_voxels_view_fix_*}` |
| `05_encode` | shape → SS → PBR encoder를 한 CUDA 프로세스에서 실행 | voxel | `05_encode/{shape,ss,pbr}` |
| `06_finalize` | latent 트리를 확인하고 선택적으로 별도 prepared-v2에 publish | encode | `06_finalize/report.json`, 선택적 `prepared-v2` |

`02_dump`는 PBR dump가 이미 보유한 geometry를 재사용하므로 mesh/PBR를 따로 import하거나 triangulate하지 않는다. Blender의 `image.pixels` 읽기는 CPU 경계지만, 읽은 뒤의 float clamp/quantization은 PyTorch CUDA에서 수행한다. `03_render`는 dump를 참조하지 않고 rendering만 한다.

## 설정

모든 기본값은 [config/default.yaml](config/default.yaml)에 있다. 경로를 코드에 하드코딩하지 않으며, 각 `run.py`는 같은 YAML을 `--config`로 읽는다. CLI 인자가 있으면 YAML의 해당 값만 덮어쓴다. 기존 n17 `youngwoo_diyscene` container의 mount 경로를 그대로 쓸 때에는 [config/youngwoo_dyscene.yaml](config/youngwoo_dyscene.yaml)을 사용한다.

| YAML 항목 | 용도 |
| --- | --- |
| `paths.control_root` | control shard와 metadata 입력 경로 |
| `paths.work_root` | 단계별 로컬 작업 출력 루트 (`01`~`06`) |
| `paths.prepared_root` | `06_finalize --publish`의 최종 publish 대상 |
| `paths.scratch_root` | 7z GLB member를 푸는 짧은 수명의 local temporary 경로 |
| `raw.*` | raw source 방식, Objaverse archive root, object-path index, extractor binary |
| `stages.*` | render 해상도, voxel 해상도/view, encoder DataLoader·microbatch 설정 |

예를 들어 환경별 경로만 바꾸려면 `config/local.yaml`을 만들고 다음처럼 실행한다.

```yaml
# data_toolkit/preprocess/config/local.yaml
paths:
  control_root: /your/control
  work_root: /your/local-work
  prepared_root: /your/prepared-v2
  scratch_root: /dev/shm/tmp
```

`local.yaml`은 `default.yaml`의 전체 구조를 포함해야 한다. YAML merge/overlay를 암묵적으로 하지 않아 설정 결과가 모호해지지 않게 했다.

## 실행

아래 예시는 n17의 `youngwoo_diyscene` container 안에서 실행하는 기준이다. 다른 환경은 `CONFIG`만 해당 YAML로 바꾼다. embedded `bpy`가 있는 torch 환경만 사용하며 외부 Blender 실행 파일은 사용하지 않는다.

```bash
TORCH_PYTHON=/home/rvi/conda/envs/torch/bin/python
CONFIG=data_toolkit/preprocess/config/youngwoo_dyscene.yaml
SOURCE=ObjaverseXL_sketchfab
BATCH_ROOT=/root/data2/pixal3d/control/shards/$SOURCE
export TORCH_PYTHON CONFIG
```

GPU 단계(`02`, `03`, `05`)에는 `--gpu` 옵션이 없다. launcher가 `CUDA_VISIBLE_DEVICES`로 **정확히 하나의 device**를 지정한다. 예를 들어 `CUDA_VISIBLE_DEVICES=5`면 프로세스 내부의 `cuda:0`은 물리 GPU 5다.

### 한 batch만 실행

`batch000` 하나를 처음부터 끝까지 실행하는 예시다. `01`의 `--build-index`는 raw source index가 없거나 raw 설정을 바꾼 경우에만 먼저 한 번 실행한다.

```bash
$TORCH_PYTHON data_toolkit/preprocess/01_manifest/run.py \
  --config "$CONFIG" --build-index \
  --shard ObjaverseXL_sketchfab-00000 --batch batch000

CUDA_VISIBLE_DEVICES=5 $TORCH_PYTHON data_toolkit/preprocess/02_dump/run.py \
  --config "$CONFIG" --shard ObjaverseXL_sketchfab-00000 --batch batch000

CUDA_VISIBLE_DEVICES=5 $TORCH_PYTHON data_toolkit/preprocess/03_render/run.py \
  --config "$CONFIG" --shard ObjaverseXL_sketchfab-00000 --batch batch000

$TORCH_PYTHON data_toolkit/preprocess/04_voxelize/run.py \
  --config "$CONFIG" --shard ObjaverseXL_sketchfab-00000 --batch batch000

CUDA_VISIBLE_DEVICES=5 $TORCH_PYTHON data_toolkit/preprocess/05_encode/run.py \
  --config "$CONFIG" --shard ObjaverseXL_sketchfab-00000 --batch batch000

$TORCH_PYTHON data_toolkit/preprocess/06_finalize/run.py \
  --config "$CONFIG" --shard ObjaverseXL_sketchfab-00000 --batch batch000 --publish
```

### 전체 batch: `01_manifest`

먼저 index는 병렬 실행 전에 단 한 번 생성한다. 이후 `xargs -P16`이 batch file 하나를 child process 하나에 주며, 최대 16개를 동시에 실행한다.

```bash
$TORCH_PYTHON data_toolkit/preprocess/01_manifest/run.py --config "$CONFIG" --build-index

find "$BATCH_ROOT" -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 -P16 bash -c '
      batch_file="$1"
      shard="$(basename "$(dirname "$batch_file")")"
      batch="$(basename "$batch_file" .txt)"
      exec "$TORCH_PYTHON" data_toolkit/preprocess/01_manifest/run.py \
        --config "$CONFIG" \
        --shard "$shard" --batch "$batch"
    ' _
```

### 전체 batch: `02_dump`, `03_render`, `05_encode` (GPU 5 한 장)

한 GPU에서는 batch를 순차적으로 실행한다. 아래에서 `STAGE`를 하나씩 `02_dump`, `03_render`, `05_encode`로 바꿔 실행한다. 다음 단계는 이전 단계가 모든 batch에서 끝난 뒤 실행한다.

```bash
STAGE=02_dump
while IFS= read -r -d '' batch_file; do
  shard="$(basename "$(dirname "$batch_file")")"
  batch="$(basename "$batch_file" .txt)"
  CUDA_VISIBLE_DEVICES=5 "$TORCH_PYTHON" "data_toolkit/preprocess/$STAGE/run.py" \
    --config "$CONFIG" --shard "$shard" --batch "$batch"
done < <(find "$BATCH_ROOT" -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 | sort -z)
```

### 전체 batch: `04_voxelize`

`04`는 GPU를 선택하지 않는다. `-P4`는 네 batch process를 동시에 실행한다. 각 process 내부의 native thread 수는 YAML의 `stages.voxelize.native_threads`로 제어한다.

```bash
find "$BATCH_ROOT" -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 -P4 bash -c '
      batch_file="$1"
      shard="$(basename "$(dirname "$batch_file")")"
      batch="$(basename "$batch_file" .txt)"
      exec "$TORCH_PYTHON" data_toolkit/preprocess/04_voxelize/run.py \
        --config "$CONFIG" \
        --shard "$shard" --batch "$batch"
    ' _
```

### 전체 batch: `06_finalize`

`--publish`는 batch별로 서로 다른 `prepared-v2/<source>/<shard>/<batch>` target만 생성하므로 batch-level 병렬 실행이 가능하다.

```bash
find "$BATCH_ROOT" -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print0 \
  | sort -z \
  | xargs -0 -n1 -P4 bash -c '
      batch_file="$1"
      shard="$(basename "$(dirname "$batch_file")")"
      batch="$(basename "$batch_file" .txt)"
      exec "$TORCH_PYTHON" data_toolkit/preprocess/06_finalize/run.py \
        --config "$CONFIG" \
        --shard "$shard" --batch "$batch" --publish
    ' _
```

### GPU가 여러 장일 때

`02`, `03`, `05`는 GPU 하나당 worker 하나를 띄운다. 아래는 GPU 0·1 두 장에 batch 목록을 modulo로 고정 분할하는 예시이며, `STAGE`는 한 번에 하나만 실행한다. worker가 재시작되어도 같은 batch 분할을 다시 얻으며 scheduler나 shared queue는 없다.

```bash
run_gpu_worker() {
  local gpu="$1" slot="$2" slots="$3" stage="$4"
  find "$BATCH_ROOT" -mindepth 2 -maxdepth 2 -name 'batch*.txt' -print \
    | sort \
    | awk -v slot="$slot" -v slots="$slots" '(NR - 1) % slots == slot' \
    | while IFS= read -r batch_file; do
        shard="$(basename "$(dirname "$batch_file")")"
        batch="$(basename "$batch_file" .txt)"
        CUDA_VISIBLE_DEVICES="$gpu" "$TORCH_PYTHON" "data_toolkit/preprocess/$stage/run.py" \
          --config "$CONFIG" --shard "$shard" --batch "$batch"
      done
}

STAGE=02_dump
run_gpu_worker 0 0 2 "$STAGE" &
run_gpu_worker 1 1 2 "$STAGE" &
wait
```

같은 batch에 대해 이후 단계를 재실행하면 이미 완성된 mesh/PBR dump와 render directory는 건너뛴다. `04`는 legacy voxel 수학 함수를 직접 순차 호출하며, production pipeline CLI, queue, scheduler를 호출하지 않는다. `05_encode`는 기존 PyTorch `Dataset`/`DataLoader` encoder들을 하나의 CUDA 프로세스에서 shape → SS → PBR 순으로 실행한다. 모델은 필요한 family/resolution 동안만 cache하며 다음 family 전에 해제하므로, 현재 GPU 메모리 사용량을 실제로 측정한 뒤 세 model의 동시 상주 여부를 결정할 수 있다.
