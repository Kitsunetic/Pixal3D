# Pixal3D batch preprocessing v2

기존 `data_toolkit/pipeline`은 보존한다. 이 디렉터리는 큐, scheduler, worker를 사용하지 않는 새 전처리 경로다. 각 `run.py`는 **하나의 batch를 순차 처리한 뒤 종료**하며, 병렬 처리는 서로 다른 batch에 대해 같은 명령을 별도 프로세스로 실행한다.

기본 batch 작업 경로는 NS3 shared root인 다음과 같다.

```text
/home/rvi/ns3/youngwoo/pixal3d/preprocess_v2/
  ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000/
```

NS3 shared root에 01~05 intermediate를 저장하므로 다른 node가 같은 batch를 이어서 실행하거나 world-size를 바꾸어 재시작할 수 있다. 아주 작은 일시 파일만 필요할 때에는 `/dev/shm/tmp`를 사용할 수 있다. 기존 NS2 `prepared`는 legacy 결과를 판정하는 read-only 입력이며, 새 결과는 `/home/rvi/ns3/youngwoo/pixal3d/prepared-v2/...`로만 publish한다. 두 결과 tree를 물리적으로 합치지 않는다.

## 단계

| 단계 | 역할 | 입력 | 출력 |
| --- | --- | --- | --- |
| `01_manifest` | control batch의 SHA-256을 이미 압축 해제된 direct GLB로 해석 | `control/shards`, metadata, `glbs_direct` | `01_manifest/manifest.jsonl` |
| `02_dump` | embedded `bpy`로 한 번 import하여 PBR dump와 mesh dump를 동시에 생성 | manifest, GLB | `02_dump/{mesh_dumps,pbr_dumps}` |
| `03_render` | GLB를 조건 뷰로 렌더링 | manifest, GLB | `03_render/renders_cond` |
| `04_voxelize` | mesh/PBR dump와 camera transform을 voxel로 변환 | dump, render | `04_voxelize/{dual_grid_view_*,pbr_voxels_view_fix_*}` |
| `05_encode` | shape → SS → PBR encoder를 한 CUDA 프로세스에서 실행 | voxel | `05_encode/{shape,ss,pbr}` |
| `06_finalize` | latent 트리를 확인하고 선택적으로 별도 prepared-v2에 publish | encode | `06_finalize/report.json`, 선택적 `prepared-v2` |

`02_dump`는 PBR dump가 이미 보유한 geometry를 재사용하므로 mesh/PBR를 따로 import하거나 triangulate하지 않는다. Blender image extraction과 quantization까지 CPU-only로 수행한다. `03_render`는 dump를 참조하지 않고 rendering만 한다.

## 최종 `prepared` 구조

중간 단계 산출물은 NS3 shared `work_root`에 둔다. 최종 학습용 결과만
`06_finalize --publish`에서 기존 학습 loader와 호환되는 `prepared` layout으로 publish한다.
한 batch는 아래의 **8개 family tar와 각 tar의 manifest**가 모두 검증된 경우에만 완료다.

```text
prepared/
├── common/<source>/<shard>/<batch>.tar
├── ss/64/<source>/<shard>/<batch>.tar
├── shape/{256,512,1024}/<source>/<shard>/<batch>.tar
├── pbr/{256,512,1024}/<source>/<shard>/<batch>.tar
└── index/<source>/<shard>.json
```

각 `*.tar` 옆에는 동일한 이름의 `*.tar.manifest.json`이 있다. manifest에는 tar와
member의 SHA-256, member 크기, 포함 asset 목록, 생성 설정/commit 정보가 들어간다.

| prepared 경로 | tar에 들어가는 내용 | 원본 shared stage |
| --- | --- | --- |
| `common/.../<batch>.tar` | asset별 `renders_cond/<asset>/{000..007}.png`, `transforms.json` | `03_render/renders_cond` |
| `ss/64/.../<batch>.tar` | `ss/64/<asset>/view{00,01}.npz` 및 각 scale JSON | `05_encode/ss` |
| `shape/<resolution>/.../<batch>.tar` | `shape/<resolution>/<asset>/view{00,01}.npz` 및 각 scale JSON | `05_encode/shape` |
| `pbr/<resolution>/.../<batch>.tar` | `pbr/<resolution>/<asset>/view{00,01}.npz` 및 각 scale JSON | `05_encode/pbr` |
| `index/<source>/<shard>.json` | batch별 8개 tar와 manifest의 상대 경로·checksum·완료 정보 | `06_finalize`가 마지막에 갱신 |

`common`의 포함 asset은 모든 manifest asset이 아니라, `SS-64`, shape, PBR 중 적어도
하나의 최종 family에 유효하게 포함된 asset의 합집합이다. family별 성공 범위는 서로
다를 수 있으므로 `06_finalize`가 각 출력 파일을 검증해 확정한다.

publish 순서는 다음과 같다.

1. shared `06_finalize` staging에서 8개 tar와 manifest를 생성하고 검증한다.
2. 검증된 tar와 manifest만 `prepared`에 원자적으로 publish한다.
3. source/shard lock 아래에서 8개 family가 모두 존재하고 검증된 뒤에만 `index`를 갱신한다.

따라서 `index`에 batch가 기록된 것은 해당 batch가 학습에 바로 사용할 수 있는 완결된
결과라는 뜻이다. `02_dump`의 mesh/PBR pickle, voxel, renderer의 임시 산출물은
`prepared`에 넣지 않는다.

`qualification/`과 `recovery/`는 기존 scheduler 기반 pipeline의 gate/checkpoint
경로다. v2 단계형 workflow는 이 두 경로를 생성하거나 사용하지 않는다.

> 현재 `06_finalize` 구현은 `prepared-v2`로 `05_encode` 트리를 복사하는 과도기
> 동작이다. 위 legacy-compatible 8-family pack 및 `index` publication은 `06`에
> 구현할 목표 contract이며, 구현 전에는 기존 학습 loader와 호환되지 않는다.

## 설정

모든 기본값은 [config/default.yaml](config/default.yaml)에 있다. 경로를 코드에 하드코딩하지 않으며, 각 `run.py`는 같은 YAML을 `--config`로 읽는다. CLI 인자가 있으면 YAML의 해당 값만 덮어쓴다. 기존 n17 `youngwoo_diyscene` container의 mount 경로를 그대로 쓸 때에는 [config/youngwoo_dyscene.yaml](config/youngwoo_dyscene.yaml)을 사용한다.

| YAML 항목 | 용도 |
| --- | --- |
| `paths.control_root` | control shard와 metadata 입력 경로 |
| `paths.work_root` | NS3 shared 단계별 작업 출력 루트 (`01`~`06`) |
| `paths.prepared_root` | `06_finalize --publish`의 최종 publish 대상 |
| `paths.existing_prepared_root` | 기존 production `prepared/index`를 읽어 이미 완결된 legacy batch를 제외하는 경로 |
| `raw.root`, `raw.glb_roots` | 압축 해제된 raw GLB tree. 기본값은 새 `glbs_direct`와 archive 밖 기존 direct GLB가 있는 `glbs`를 함께 읽는다 |
| `stages.*` | render 해상도, voxel 해상도/view, encoder DataLoader·microbatch 설정 |

예를 들어 환경별 경로만 바꾸려면 `config/local.yaml`을 만들고 다음처럼 실행한다.

```yaml
# data_toolkit/preprocess/config/local.yaml
paths:
  control_root: /your/control
  work_root: /your/local-work
  prepared_root: /your/prepared-v2
raw:
  mode: direct_glb
  root: /your/raw-parent
  glb_roots:
    - glbs_direct
```

`local.yaml`은 `default.yaml`의 전체 구조를 포함해야 한다. YAML merge/overlay를 암묵적으로 하지 않아 설정 결과가 모호해지지 않게 했다.

기존 archive 기반 `asset_index.sqlite`는 새 direct GLB 설정과 혼용할 수 없다. Sketchfab
압축 해제가 끝난 뒤 각 worker node에서 `01_manifest/run.py --rebuild-index`를 한 번 실행해
shared index를 재생성한다. archive record가 남은 index는 01 단계가 명시적으로 거부하므로
02/03이 7z를 다시 호출하는 경로는 없다.

## 실행: world-size/rank 필수

모든 `01`~`06` stage에는 `--world-size`, `--rank`가 **필수**다. control의
`<shard>/batch*.txt` 전체를 shard/batch 상대경로로 정렬한 위치가 `batch_index`이며,
stage는 `batch_index % world_size == rank`가 아니면 실행을 거부한다. 완료 여부와
관계없이 전체 목록을 기준으로 index를 계산하므로 재시작해도 분할이 변하지 않는다.

중간 산출물은 NS3 shared `work_root`에 남긴다. 다른 node도 같은 완료된 stage 결과를 읽을 수 있다. legacy NS2 `prepared`와 completion marker가 있는 NS3 `prepared-v2` batch는
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

각 process는 하나의 rank만 갖고, 자신이 소유한 batch를 shared work root에서 01부터 06까지
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
새 invocation을 시작할 수 있다. shared work root의 완료된 asset·stage 산출물은 재사용되며,
미완료 asset만 해당 stage가 다시 생성한다.
