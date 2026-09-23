# Pixal3D batch preprocessing v2

기존 `data_toolkit/pipeline`은 보존한다. 이 디렉터리는 큐, scheduler, worker를 사용하지 않는 새 전처리 경로다. 각 `run.py`는 **하나의 batch를 순차 처리한 뒤 종료**하며, 병렬 처리는 서로 다른 batch에 대해 같은 명령을 별도 프로세스로 실행한다.

기본 batch 작업 경로는 NS3 shared root인 다음과 같다.

```text
/home/rvi/ns3/youngwoo/pixal3d/preprocess_v2/
  ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00000/batch000/
```

NS3 shared root에 01~05 intermediate를 저장하므로 다른 node가 같은 batch를 이어서 실행하거나 world-size를 바꾸어 재시작할 수 있다. 아주 작은 일시 파일만 필요할 때에는 `/dev/shm/tmp`를 사용할 수 있다. 06은 큰 tar를 각 서버의 로컬 `./data/preprocess_finalize`에 임시 생성하고, 학습 Dataset 검증을 통과한 batch만 기존 NS2 `prepared`에 합친다.

## 단계

| 단계 | 역할 | 입력 | 출력 |
| --- | --- | --- | --- |
| `01_manifest` | control batch의 SHA-256을 이미 압축 해제된 direct GLB로 해석 | `control/shards`, metadata, `glbs_direct` | `01_manifest/manifest.jsonl` |
| `02_dump` | embedded `bpy`로 한 번 import하여 PBR dump와 mesh dump를 동시에 생성 | manifest, GLB | `02_dump/{mesh_dumps,pbr_dumps}` |
| `03_render` | GLB를 조건 뷰로 렌더링 | manifest, GLB | `03_render/renders_cond` |
| `04_voxelize` | mesh/PBR dump와 camera transform을 voxel로 변환 | dump, render | `04_voxelize/{dual_grid_view_*,pbr_voxels_view_fix_*}` |
| `05_encode` | shape → SS → PBR encoder를 한 CUDA 프로세스에서 실행. VXZ 입력은 PyTorch Dataset/DataLoader로 prefetch하고, latent 저장은 별도 saver queue/thread가 담당 | voxel | `05_encode/{shape,ss,pbr}` |
| `06_finalize` | 로컬에서 8개 tar 생성·학습 loader 확인 후 선택적으로 NS2 prepared에 합침 | render, encode | `06_finalize/report.json`, 선택적 `prepared` tar·manifest·index |

`02_dump`는 PBR dump가 이미 보유한 geometry를 재사용하므로 mesh/PBR를 따로 import하거나 triangulate하지 않는다. Blender image extraction과 quantization까지 CPU-only로 수행한다. `03_render`는 dump를 참조하지 않고 rendering만 한다.

## `prepared` 구조

중간 단계 산출물은 NS3 shared `work_root`에 둔다. `06_finalize --publish`는
로컬 임시 경로에서 검증한 tar 결과를 기존 NS2 `prepared`에 합친다.
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
| `ss/64/.../<batch>.tar` | `ss_latents/<model>/<asset>/view{00,01}.npz` 및 각 scale JSON | `05_encode/ss` |
| `shape/<resolution>/.../<batch>.tar` | `shape_latents/<model>/<asset>/view{00,01}.npz` 및 각 scale JSON | `05_encode/shape` |
| `pbr/<resolution>/.../<batch>.tar` | `pbr_latents/<model>/<asset>/view{00,01}.npz` 및 각 scale JSON | `05_encode/pbr` |
| `index/<source>/<shard>.json` | batch별 8개 tar와 manifest의 상대 경로·checksum·완료 정보 | `06_finalize`가 마지막에 갱신 |

현재 06은 04 성공 asset 모두에 대해 8개 family와 모든 view가 완전해야 publish한다.
하나라도 누락되면 부분 tar를 발행하지 않는다.

publish 순서는 다음과 같다. 기존 패커와 같이 tar는 PAX 형식의 비압축 `.tar`다.

1. 로컬 `paths.local_temp_root` 아래에서 8개 tar와 manifest를 생성하고 SHA-256 검증한다.
2. tar에서 실제 어셋 하나를 꺼내 control metadata의 aesthetic score와 임시 metadata projection으로 7개 학습 설정의 Dataset/DataLoader를 검사한다. 두 view를 각각 직접 읽어 loader의 재귀 재시도에 의한 무한 대기를 피한다. 이 projection은 검사에만 쓰며 tar에 넣지 않는다.
3. 통과한 tar를 NS2 `prepared` 내부 임시 staging으로 복사하고 checksum을 다시 확인한다.
4. source/shard lock 아래에서 기존 batch와 파일 충돌이 없는지 확인한 뒤 8개 family를 발행하고, 기존 index 항목을 보존한 채 새 batch를 마지막에 추가한다.

따라서 `index`에 batch가 기록된 것은 8개 family tar의 발행이 끝났다는 뜻이다.
`02_dump`의 mesh/PBR pickle와 voxel은 `prepared`에 넣지 않는다. 기존 batch를 덮어쓰지 않는다.

`qualification/`과 `recovery/`는 기존 scheduler 기반 pipeline의 gate/checkpoint
경로다. v2 단계형 workflow는 이 두 경로를 생성하거나 사용하지 않는다.

학습 loader 검사는 batch 중 학습 조건을 충족하는 한 어셋의 두 view에 대한 smoke test다.
모든 어셋의 시각적 품질 심사나 전체 학습용 `metadata.csv` 생성은 별도 작업이다.
1024 PBR 학습 설정의 `full_pbr` texture-count 필터는 이 임시 metadata projection에
원본 texture-count 열이 없어 검사하지 않는다.
v2 설정 해시는 기존 production 설정 해시와 다를 수 있다.

## 설정

모든 기본값은 [config/default.yaml](config/default.yaml)에 있다. 경로를 코드에 하드코딩하지 않으며, 각 `run.py`는 같은 YAML을 `--config`로 읽는다. CLI 인자가 있으면 YAML의 해당 값만 덮어쓴다. 기존 n17 `youngwoo_diyscene` container의 mount 경로를 그대로 쓸 때에는 [config/youngwoo_dyscene.yaml](config/youngwoo_dyscene.yaml)을 사용한다.

| YAML 항목 | 용도 |
| --- | --- |
| `paths.control_root` | control shard와 metadata 입력 경로 |
| `paths.work_root` | NS3 shared 단계별 작업 출력 루트 (`01`~`06`) |
| `paths.prepared_root` | 검증된 family tar·manifest·index를 합칠 NS2 `prepared` 경로 |
| `paths.existing_prepared_root` | 이미 완결된 `prepared/index` batch를 제외하는 경로. 일반적으로 `prepared_root`와 동일 |
| `paths.local_temp_root` | 06의 로컬 tar·학습 loader 임시 검사 경로. 완료/실패 뒤 batch 임시 디렉터리 삭제 |
| `raw.root`, `raw.glb_roots` | 압축 해제된 raw GLB tree. 기본값은 새 `glbs_direct`와 archive 밖 기존 direct GLB가 있는 `glbs`를 함께 읽는다 |
| `stages.*` | render 해상도, voxel 해상도/view, encoder DataLoader·microbatch 설정 |

예를 들어 환경별 경로만 바꾸려면 `config/local.yaml`을 만들고 다음처럼 실행한다.

```yaml
# data_toolkit/preprocess/config/local.yaml
paths:
  control_root: /your/control
  work_root: /your/local-work
  existing_prepared_root: /your/prepared
  prepared_root: /your/prepared
  local_temp_root: ./data/preprocess_finalize
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

중간 산출물은 NS3 shared `work_root`에 남긴다. 다른 node도 같은 완료된 stage 결과를 읽을 수 있다. NS2 `prepared/index`에 완료된 batch는 모든 stage 및 rank launcher가 skip한다.

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
