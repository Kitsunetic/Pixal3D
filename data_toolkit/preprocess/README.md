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

`02_dump`는 PBR dump가 이미 보유한 geometry를 재사용하므로 mesh/PBR를 따로 import하거나 triangulate하지 않는다. Blender image extraction과 quantization까지 CPU-only로 수행한다. `03_render`는 dump를 참조하지 않고 rendering만 한다.

## 설정

모든 기본값은 [config/default.yaml](config/default.yaml)에 있다. 경로를 코드에 하드코딩하지 않으며, 각 `run.py`는 같은 YAML을 `--config`로 읽는다. CLI 인자가 있으면 YAML의 해당 값만 덮어쓴다. 기존 n17 `youngwoo_diyscene` container의 mount 경로를 그대로 쓸 때에는 [config/youngwoo_dyscene.yaml](config/youngwoo_dyscene.yaml)을 사용한다.

| YAML 항목 | 용도 |
| --- | --- |
| `paths.control_root` | control shard와 metadata 입력 경로 |
| `paths.work_root` | 단계별 로컬 작업 출력 루트 (`01`~`06`) |
| `paths.prepared_root` | `06_finalize --publish`의 최종 publish 대상 |
| `paths.existing_prepared_root` | 기존 production `prepared/index`를 읽어 이미 완결된 legacy batch를 제외하는 경로 |
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

## 실행: world-size/rank 필수

모든 `01`~`06` stage에는 `--world-size`, `--rank`가 **필수**다. control의
`<shard>/batch*.txt` 전체를 shard/batch 상대경로로 정렬한 위치가 `batch_index`이며,
stage는 `batch_index % world_size == rank`가 아니면 실행을 거부한다. 완료 여부와
관계없이 전체 목록을 기준으로 index를 계산하므로 재시작해도 분할이 변하지 않는다.

중간 산출물은 worker node의 `./data/preprocess_v2`에만 남긴다. 다른 node의 중간
결과를 읽지 않는다. legacy `prepared`와 completion marker가 있는 `prepared-v2` batch는
모든 stage 및 rank launcher가 skip한다.

GPU stage(`03`, `05`)는 `CUDA_VISIBLE_DEVICES`로 정확히 한 device만 지정한다.
`02_dump`는 CPU-only다.

### 한 batch 확인

world size 1, rank 0은 모든 batch의 유효한 소유자이므로 단일 batch 확인에 사용한다.

```bash
TORCH_PYTHON=/home/rvi/conda/envs/torch/bin/python
$TORCH_PYTHON data_toolkit/preprocess/02_dump/run.py \
  --world-size 1 --rank 0 \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010
```

### rank worker: batch-closed 실행

각 process는 하나의 rank만 갖고, 자신이 소유한 batch를 로컬에서 01부터 06까지
순서대로 실행한다. outer launcher는 queue/scheduler가 아니라 deterministic 목록 확장기다.

```bash
WORLD_SIZE=16
RANK=3
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/00_rank/run.py \
  --world-size "$WORLD_SIZE" --rank "$RANK" --publish
```

`03`, `05`까지 포함하므로 위 process에는 필요시 GPU를 하나만 노출한다.

```bash
CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/00_rank/run.py \
  --world-size 16 --rank 3 --publish
```

CPU-only 02만 실행하려면 stage 목록을 명시한다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/00_rank/run.py \
  --world-size 16 --rank 3 --stages 01,02
```

동일한 `(world_size, rank)`를 중복 실행하면 안 된다. 물리 서버나 process 수를 바꿔
새 invocation을 시작할 수 있지만, 아직 publish되지 않은 batch가 새 rank로 이동하면
그 node의 local work root에서 처음부터 다시 생성된다.
