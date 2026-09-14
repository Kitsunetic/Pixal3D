# Pixal3D 작업 기록

## 2026-08-31 — Exact-output 데이터 전처리 최적화

### 목적

Pixal3D latent 전처리에서 GPU 메모리를 크게 점유하면서 GPU utilization이
낮아지는 현상을 조사하고, 기존 FP16 latent 출력을 그대로 유지하면서 메모리
사용량과 전체 처리량을 개선했다.

작업에는 기존 `/home/rvi/conda/envs/torch` 환경을 사용했으며 새 dependency는
설치하지 않았다. 원본 데이터 검증에는 다음 Objaverse GLB 디렉터리의 sample을
사용했다.

```text
/home/rvi/ns3/jaehyeok/ds/Objaverse-full/glbs/000-082
```

### 조사 결과

- 기존 shape/PBR/SS latent encoder는 sample을 하나씩 처리하지만, 직전 sample의
  sparse tensor와 spatial cache가 다음 iteration의 encoder 실행 시점까지 살아
  있었다. 이 때문에 연속 처리 peak GPU memory가 불필요하게 증가했다.
- 기본 I/O 설정은 loader 32개가 각각 VXZ reader thread 4개를 생성할 수 있어,
  최대 128개 decode thread와 최대 32개 decoded input prefetch가 발생했다. CPU와
  네트워크 스토리지의 oversubscription 및 큰 host-memory 사용 가능성이 있었다.
- 여러 sparse sample을 하나의 batch로 합치면 GPU 구간은 빨라졌지만 FP16 latent
  feature가 원본과 bitwise 동일하지 않았다. 3개 sample 실험에서 좌표는 같았지만
  feature의 최대 절대오차가 약 `0.014`였으므로 exact-output 기본 경로에는 batching을
  적용하지 않았다.
- 현재 shell의 `base` conda 환경에는 PyTorch가 없었고, 기존 `torch` 환경에는
  PyTorch `2.11.0+cu128`, `o_voxel`, Pixal3D sparse backend가 준비되어 있었다.

### 구현 내용

#### GPU tensor 수명 및 allocator

- 모든 shape/PBR/SS, view/non-view latent encoder에 `torch.inference_mode()`를
  적용했다.
- latent pack을 CPU로 복사한 직후 GPU input, output 및 연결된 sparse spatial
  cache를 명시적으로 해제하도록 변경했다.
- CUDA allocator fragmentation을 줄이기 위해
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 기본 적용했다. 사용자가 이미
  allocator 설정을 지정한 경우에는 덮어쓰지 않는다.
- `--empty_cache_interval N` 옵션을 추가했다. `N=1`이면 매 sample 처리 후 사용하지
  않는 allocator block을 GPU driver에 반환한다. 기본값 `0`은 allocator cache를
  유지하여 최대 throughput을 우선한다.

대상 스크립트:

- `data_toolkit/encode_shape_latent.py`
- `data_toolkit/encode_shape_latent_view.py`
- `data_toolkit/encode_pbr_latent.py`
- `data_toolkit/encode_pbr_latent_view.py`
- `data_toolkit/encode_ss_latent.py`
- `data_toolkit/encode_ss_latent_view.py`

#### Bounded I/O pipeline

공통 설정을 `data_toolkit/latent_io.py`로 분리하고 다음 CLI 옵션을 추가했다.

| 옵션 | 기본값 | 설명 |
| --- | ---: | --- |
| `--loader_workers` | 8 이하 CPU 수 | 병렬 input loader 수 |
| `--loader_threads` | 2 | 각 VXZ reader가 사용하는 thread 수 |
| `--saver_workers` | 8 이하 CPU 수 | 병렬 NPZ writer 수 |
| `--prefetch` | 8 | host memory에서 대기할 decoded input 상한 |
| `--empty_cache_interval` | 0 | CUDA allocator cache 반환 주기 |

모든 값은 CLI에서 storage 및 CPU 특성에 맞게 조절할 수 있으며 잘못된 0 또는
음수 설정은 실행 전에 거부한다.

#### Exact-output 멀티 GPU 실행

`data_toolkit/launch_multi_gpu.py`를 추가했다. 기존 encoder의 `--rank`와
`--world_size` partition을 이용하고 GPU마다 독립 process를 실행하므로 sample 내부
연산 및 출력은 단일 GPU 경로와 동일하다.

```bash
conda run -n torch python data_toolkit/launch_multi_gpu.py --gpus 0-7 -- \
    data_toolkit/encode_shape_latent_view.py \
    --root <DATASET_ROOT> \
    --resolution 1024 \
    --view_indices 0-1
```

launcher는 각 process에 GPU 하나만 노출하고 rank별 `part_<rank>.csv`를 생성한다.
모든 shard가 끝난 후 기존과 같이 `build_metadata.py`를 실행해야 한다.

#### Dataset adapter 수정

- `data_toolkit/datasets/__init__.py`를 추가해 설치된 Hugging Face `datasets`
  package와 로컬 adapter 이름이 충돌하던 문제를 해결했다.
- ObjaverseXL의 `objaverse.xl` import를 download 함수 내부로 이동해 이미 받은
  파일의 voxelization에는 `objaverse` package가 필요하지 않도록 했다.
- ObjaverseXL, ABO, TexVerse adapter에 `no_file` mode를 구현했다. 이에 따라
  `dual_grid.py`와 `voxelize_pbr.py`가 원본 파일 경로 없이 전체 metadatum을 worker에
  전달하는 기존 호출 방식으로 정상 실행된다.

### 검증 결과

#### 1024 shape latent 메모리 비교

공식 TRELLIS.2 경로와 tensor/file hash 동일성이 확인된 O-Voxel을 사용했다. 입력은
active voxel `2,324,327`개였으며, 연속 처리 시 tensor 수명에 따른 peak 차이를
분리하기 위해 동일한 큰 입력을 3개 asset ID로 처리했다. Flex-GEMM autotune cache가
준비된 warm 상태에서 비교했다.

| 항목 | 원본 | 수정본 |
| --- | ---: | ---: |
| GPU peak memory (`nvidia-smi`) | 약 6.16 GiB | 약 4.17 GiB |
| GPU peak 감소 | — | 약 32% |
| encoder progress 시간 | 2.56초 | 2.56초 |
| NPZ 좌표 | 기준 | bitwise 동일 |
| NPZ feature | 기준 | bitwise 동일 |
| NPZ SHA-256 | 기준 | 3개 모두 동일 |

`--empty_cache_interval 1`을 사용하면 sample 사이 GPU 점유가 약 `1.1 GiB`까지
반환되었다. 같은 실험에서 encoder progress 시간은 `2.72초`로 약 6% 증가했으며,
생성된 NPZ SHA-256은 여전히 원본과 동일했다. 이 옵션은 idle memory 반환이 중요한
공유 GPU 환경을 위한 선택 사항이다.

#### 제공된 Objaverse GLB 검증

제공된 디렉터리에서 다음 GLB 3개를 선택해 Blender 4.4와 O-Voxel resolution 256으로
변환했다.

- `39fc7702443f45f3902b52b230037e9d.glb`: active voxel 99,405개
- `9ee3b84f94574bf0aab486f96ef52d5d.glb`: active voxel 131,068개
- `b3119cf53e2d4681a62b1013cf5113f8.glb`: active voxel 65,562개

warm 상태의 shape latent encoder progress 시간은 원본 `1.50초`, 수정본 `1.46초`로
유사했다. 세 asset의 수정본 NPZ SHA-256은 각각 대응하는 원본 NPZ와 완전히 같았다.

#### View 및 멀티 GPU 검증

- 1024 shape view latent의 원본/수정본 NPZ SHA-256이 동일함을 확인했다.
- GPU 3개에 asset 하나씩 배정한 launcher 실행 결과가 단일 GPU 출력과 SHA-256까지
  동일했다.

#### Objaverse 100개 실데이터 benchmark

제공된 디렉터리에서 파일명을 정렬한 뒤 앞 100개 GLB를 고정 sample로 사용했다.
입력은 합계 `1.132 GiB`, 중앙값 `2,912,306 bytes`, p90 `26,344,384 bytes`, 최대
`149,900,732 bytes`였다. shape 전처리의 geometry 경로인
`GLB -> normalized mesh dump -> O-Voxel 1024 -> shape latent`를 측정했으며, 이미지
rendering과 PBR/SS/view latent는 이 benchmark 범위에 포함하지 않았다.

Geometry/O-Voxel 단계는 Blender 4.4 persistent worker 8개, worker당 최대 10개 asset
조건으로 실행했다. 첫 실행은 잘못된 skin vertex를 가진 GLB 하나에서 전체 worker
pool이 중단되는 것도 확인했다. 최종 측정에서는 asset별 예외를 격리해 나머지를 계속
처리했다.

| 항목 | 결과 |
| --- | ---: |
| 입력 시도 / 성공 / 실패 | 100 / 99 / 1 |
| wall time | 345.89초 (5분 45.89초) |
| 성공 처리량 | 17.17개/분 |
| 99개 serial work 합계 | 2,248.85초 |
| worker 병렬화 효과 | 6.50배 |
| asset total 중앙값 / p90 / 최대 | 18.24 / 42.23 / 86.55초 |
| mesh dump work 합계 | 470.54초 |
| O-Voxel work 합계 | 1,778.28초 |
| active voxel 중앙값 / p90 / 최대 | 2.219M / 4.961M / 13.647M |
| VXZ 출력 합계 | 0.297 GiB |

실패 asset은 `0107bf9b8c7345cdaa78ea4f71f082d7.glb`이며 Blender가
`Some vertices are not assigned to bone(s)`를 보고한 뒤 geometry가 없는 mesh dump가
생성되었다. 데이터 오류 하나가 전체 장시간 job을 중단하지 않도록 asset 단위 실패
격리가 필요하다.

성공한 99개 VXZ를 수정된 shape encoder로 처리했다. 측정 wall time에는 Python/model
startup과 출력 저장이 모두 포함된다. 8 GPU 실행은 각 process의 host-side 동시성을
`loader 2 x reader thread 2`, `saver 2`, `prefetch 2`로 제한했다.

| 항목 | 1 x RTX 3090 | 8 x RTX 3090 |
| --- | ---: | ---: |
| 성공 출력 | 99 | 99 |
| wall time | 103.40초 | 31.48초 |
| 처리량 | 57.45개/분 | 188.68개/분 |
| 1 GPU 대비 wall speedup | 1.00배 | 3.28배 |
| peak GPU memory | 17,595 MiB | GPU당 5,093–17,615 MiB |
| active GPU utilization 평균 | 75.49% | GPU별 45.64–55.92% |
| active GPU utilization p90 | 100% | GPU별 약 99–100% |

99개처럼 작은 작업에서는 GPU마다 model을 읽는 고정 startup 비용과 shard별 active
voxel 불균형 때문에 8배 선형 확장에는 도달하지 않는다. GPU 계산 중에는 p90
utilization이 약 100%였고, 복잡한 asset이 배정된 GPU의 peak memory가 더 높았다.

두 실행의 NPZ 99개를 파일 단위 SHA-256으로 비교한 결과 `99/99`가 완전히 같았고,
총 출력 크기도 각각 `104,966,597 bytes`로 동일했다. 순차 실행 기준 전체 geometry와
latent wall time은 1 GPU `449.28초`(7분 29.28초), 8 GPU `377.37초`(6분 17.37초)다.
현재 8 GPU 구성에서는 전체 시간의 약 91.7%가 CPU geometry/O-Voxel 단계이므로,
추가 end-to-end 개선의 우선순위는 이 단계의 worker 확장과 장시간 job의 실패 격리다.

##### Benchmark 범위 및 1 GPU 항목당 시간 해석

위 `449.28초`는 **Pixal3D 전체 preprocessing 시간이 아니라**, 이번에 측정한
비-view shape 부분 경로의 1 GPU 결과다. 100개 입력 중 성공 출력 99개를 기준으로
하면 CPU 8-worker geometry/O-Voxel의 amortized wall time은 `3.49초/개`, 1 GPU shape
latent는 `1.04초/개`, 두 단계를 순차 실행한 합계는 `4.54초/성공 출력`이다. 입력
시도 100개로 나누면 `4.49초/입력`이다.

이 값은 병렬 job의 처리량을 항목 수로 나눈 값이지 단일 asset의 실제 latency가
아니다. CPU asset work 자체는 평균 `22.72초`, 중앙값 `18.24초`, p90 `42.23초`였고
8개 worker가 이를 겹쳐 실행했다.

공식 전체 preprocessing과 비교하면 이번 100개 실험에서는 metadata/download를
이미 준비된 GLB로 대체했고, PBR dump와 asset statistics, condition image rendering,
view-aligned shape/PBR voxelization, view shape/PBR latent, SS latent 및 최종 metadata
갱신을 실행하지 않았다. 따라서 전체 파이프라인의 1 GPU 항목당 최종 시간은 아직
측정되지 않았다.

##### Coworker 장시간 실행과 이번 benchmark의 차이

Coworker의 정확한 실행 명령과 stage별 log가 없으므로 단일 원인을 확정하지는
않았다. 코드상 가능한 주요 원인은 다음과 같다.

- 이번 benchmark는 asset당 non-view shape latent 1개만 만들었다. 공식 예시처럼
  view 2개와 512/1024 두 해상도에서 shape와 PBR을 모두 처리하면 shape/PBR latent만
  asset당 8개이며, SS view latent 2개가 추가된다. view 수를 24로 사용하면 작업 수는
  훨씬 더 커진다.
- mesh/PBR dump, condition rendering, shape/PBR O-Voxel 변환은 CPU/Blender 중심이다.
  이 단계에서 GPU utilization이 낮은 것은 정상이고 전체 preprocessing 시간에는 크게
  포함될 수 있다.
- 수정 전 latent loader는 worker 32개가 VXZ reader thread 4개씩 사용할 수 있었고
  decoded input 32개를 queue에 둘 수 있었다. storage read/decode와 NPZ compression의
  CPU contention 사이에는 GPU가 기다리므로 utilization이 낮아질 수 있다.
- 수정 전에는 직전 sparse output/input cache가 다음 forward 시작 시점까지 남아
  있었다. 따라서 GPU 계산을 기다리는 동안에도 큰 allocator/cache 점유가 유지되어
  `높은 GPU memory + 낮은 순간 utilization` 조합이 나타날 수 있었다.
- `nvidia-smi`의 순간 표시는 짧은 sparse GPU burst 사이의 CPU/I/O gap을 0%로 잡을
  수 있다. 수정본 99개 shape latent 실측에서는 active 평균 75.49%, p90 100%였으므로
  latent 계산 구간 자체는 GPU를 실제로 사용했다.

어느 원인이 coworker job을 지배했는지 확인하려면 실제 command, view 수, resolution,
현재 stage와 stage별 시작/종료 시간이 필요하다.

## 2026-08-31 — n17 coworker 실행 상태 확인

`../4d-gen/AGENTS.md`, `agent/remote_training.md`, `agent/training.md`의 원격 점검
규칙을 읽고 `n17`의 `youngwoo_diyscene` container를 변경 없이 조사했다.

### 현재 상태

- 조사 시점에 Pixal3D preprocessing process는 없었다.
- GPU 4–7에서는 `train_surface_warping_wr4_ss.py`의 oracle-teacher 4-GPU 학습이
  실행 중이었다. 2026-08-31 09:14 KST에 시작했고 14:05 KST 기준 약 4시간 52분,
  `1800/2000` step이었다.
- paired GT-flow arm은 같은 시각에 시작해 13:54 KST에 `2000/2000` step과 약
  149 MB checkpoint를 완료했다.
- 실행 중인 학습은 GPU당 약 71 GB의 process memory를 사용했고, sampling 순간에는
  GPU utilization 100%였다. 따라서 현재 관측되는 큰 GPU memory는 preprocessing이
  아니라 이 학습 job의 사용량이다.
- 학습 입력은 `/root/data3/pixal3d/experiments/anchor-hybrid/runs/robust-surface-warping-wr4-20260831/cache`의
  24 GB cache다. 3D-Future, ABO, HSSD의 유효 asset 293개에 bend/bulge/pull/mixed
  변형을 적용한 총 1,172개 episode다.

### 과거 production preprocessing 기록

`/root/data3/pixal3d/preprocess/active`에는 2026-08-03~08-05에 수행한 ObjaverseXL
기록이 남아 있었다. 6,656개 asset을 256개씩 26 batch로 고정하고, 각 batch를
64개씩 4 chunk로 나눴다. 총 104개 batch-chunk checkpoint 기록 중 11개가
finalize까지, 5개가 1024 geometry까지, 88개가 prepare까지만 완료됐다. 여기서
`prepare까지만` 완료됐다는 것은 prepare가 멈췄다는 뜻이 아니라, raw staging/mesh
dump/PBR dump/asset statistics를 마친 뒤 다음 `render_cond`를 완료하지 못했다는
뜻이다.

설정은 condition render 8개(512), target view 2개, resolution 256/512/1024,
SS resolution 64였다. 완료 stage 기록에는 shape와 PBR voxelization/latent,
SS latent 및 output validation이 포함된다. full stage가 실행된 chunk는 주로 encoder
rank 3개와 render GPU 3개를 사용했다.

64-input chunk의 기록된 평균 stage wall time과 scheduled input당 amortized time은
다음과 같다. 이 값은 병렬 처리량이며 단일 asset 또는 단일 view의 실제 latency가
아니다.

| stage | chunk 평균 | scheduled input당 |
| --- | ---: | ---: |
| prepare: raw/dump mesh/PBR/stats | 265.46초 | 4.15초 |
| condition render 8 views | 3,079.38초 | 48.12초 |
| geometry 256 | 97.92초 | 1.53초 |
| shape+PBR encode 256 | 48.83초 | 0.76초 |
| geometry 512 | 285.77초 | 4.47초 |
| shape+PBR encode 512 | 74.47초 | 1.16초 |
| geometry 1024 | 1,116.97초 | 17.45초 |
| shape+PBR encode 1024 | 140.11초 | 2.19초 |
| SS/final validation | 55.10초 | 0.86초 |
| 합계 | 약 5,164초 | 약 80.69초 |

이 기록에서 condition rendering이 약 59.6%, 1024 geometry가 약 21.6%를 차지한다.
세 해상도의 GPU latent encoding 합계는 약 5.1%다. Coworker의 전체 preprocessing이
오래 걸린 주된 이유는 latent encoder보다 8-view Blender rendering과 고해상도
shape/PBR geometry였다. 이번 로컬 benchmark는 render/PBR/view 단계를 제외했기
때문에 이 workload를 대표하지 못했다.

#### Rendering 및 checkpoint 의미 재조사

위 표의 `condition render 48.12초/개`는 1-view latency가 아니다. 64개 입력을 최대
6개 Blender process(3 GPU x GPU당 2 worker)로 겹쳐 처리한 chunk wall time을 64로
나눈 값이며, 각 입력의 8-view 전체에 대한 처리량 지표다.

remote production 구현은 asset마다 Blender subprocess를 한 번 시작하고 mesh도 한
번만 load한 뒤 8개 view를 순차 처리한다. 따라서 같은 mesh를 view마다 재로딩하는
구조는 아니다. 병목은 각 view를 512x512 Cycles 32 sample + denoising으로 렌더한 뒤
PNG alpha mask의 화면 경계 거리를 검사하고, camera radius를 0.9/1.1배 조정하면서
최대 10회까지 full render를 반복하는 부분이다.

render stage가 완료된 16개 chunk의 실제 산출물 992개 asset, 7,936개 final view를
집계한 결과는 다음과 같다.

- 최종 view 중 7,487개(94.34%)가 적어도 한 번 재렌더됐다.
- view당 평균 retry는 2.93회였고, 총 Cycles render 호출은 31,158회였다.
- 최종 view 하나당 평균 Cycles 호출 수는 3.93회였으며, 41개 view는 10회 한도에
  도달했다.
- 기록된 stage wall time을 기준으로 final view 처리량은 약 6.21초/view지만, 이는
  6개 process가 겹쳐 실행된 수치다. 6-way concurrency를 단순 역산한 단일 asset
  stream의 평균 latency는 약 298초/asset(8 views), 즉 약 37.25초/final view다.

`geometry 256/512/1024`는 각각 하나의 GPU 연산 이름이 아니다. 각 resolution에서
shape용 `dual_grid_<resolution>`과 PBR용 `voxelize_pbr_<resolution>`을 묶은 scheduler
stage 이름이며, latent encode는 다음 `encode_<resolution>` stage에 별도로 기록된다.
세 geometry stage의 scheduled input당 amortized 합계는 `1.53 + 4.47 + 17.45 =
23.45초`다. 현재 checkpoint에는 두 geometry 하위 명령의 시간이 분리되어 있지 않아
shape와 PBR 각각의 비중은 이 기록만으로 계산할 수 없다.

prepare까지만 완료된 88개 checkpoint는 모두 `render_cond` 시도 기록이 있고 현재
active attempt는 없다. 시도 횟수는 1회 12개, 2회 28개, 3회 48개다. 이 104개 기록은
batch 이름을 제외하면 40개의 동일한 shard/chunk ID에 매핑되므로 escalation report가
과거 batch별로 보존되지 않고 최신 report로 덮인다. 현재 남은 40개 최신
`render_cond` escalation은 모두 실제 Blender rendering 실패가 아니라
`ranked command exceeds the 28-process cap`이라는 worker/rank 구성 오류다. 따라서
88개 전체의 과거 실패 이유를 동일하다고 단정할 수는 없지만, 적어도 최신 상태는
prepare가 느려 정지한 것이 아니라 render command launch 전 infrastructure admission
실패다.

`../4d-gen/utils/render`의 최적화 아이디어는 최종 shading 교체보다 camera-radius
탐색에 적용하는 편이 exact-output 조건에 가깝다. persistent nvdiffrast context와
한 번 load한 mesh로 8-view alpha mask를 batch rasterization해 현재와 같은 radius를
결정한 뒤, 최종 radius에서만 Blender Cycles를 한 번 실행하면 현재 평균 3.93회의
Cycles 호출을 1회로 줄일 가능성이 있다. 다만 alpha silhouette의 경계 판정과 최종
camera transform이 기존 구현과 일치하는지 corpus 비교가 필요하다. 또한 현재
Blender script는 retry마다 unseeded random lighting을 다시 생성하므로, skipped retry의
RNG 소비까지 정의하지 않으면 bitwise 동일 출력은 원래 구현 자체에서도 재실행 간
보장되지 않는다.

### 자동 검증

```text
12 passed
```

`tests/test_preprocess_runtime.py`에서 GPU 목록/range parsing, I/O 옵션 검증,
ObjaverseXL/ABO/TexVerse의 `no_file` adapter 동작을 검사한다. 추가로 수정된 모든
Python 파일에 `py_compile`을 실행하고 `git diff --check`를 통과했다.

### 사용 시 참고 사항

- 최초 Flex-GEMM 실행은 kernel autotuning 때문에 warm 실행보다 크게 느릴 수 있다.
  실제 throughput 비교에는 autotune 완료 후 장시간 구간을 사용해야 한다.
- exact-output 조건 때문에 sparse batching은 기본 구현에 포함하지 않았다.
- 한 sample 자체의 inference peak는 해당 asset의 active voxel 수에 따라 달라진다.
  이번 수정은 특히 이전/현재 sample cache가 겹쳐 발생하던 추가 peak와 idle memory
  점유를 제거한다.
- 최고 단일 GPU throughput은 `--empty_cache_interval 0`, 공유 GPU의 낮은 idle
  memory가 우선이면 `--empty_cache_interval 1`을 권장한다.

## 2026-09-02 — n17 slow-render 로컬 재현

n17 `youngwoo_diyscene` container의 완료된 ObjaverseXL chunk에서 GLB 크기가 비슷한
두 asset을 선별해 로컬로 임시 복사했다. 원격 production Blender script
SHA-256은 `ef4226a38e568651a50c25051ceca3d33284cdcc7e4ba04c7b5ec749407d7ec9`였고,
입력 GLB의 SHA-256은 각각 asset ID와 일치했다.

- control: `431f607a...d73163`, 4.69 MB, 원격 retry 합 13회
- high-retry: `4320138a...5ea8`, 4.17 MB, 원격 retry 합 65회

로컬 RTX 3090 GPU 0 하나에서 Blender 4.5.1, OPTIX, 512x512, Cycles 32 samples,
denoising, 8 condition views로 순차 실행했다. 1-view warm-up 후 측정한 결과는 다음과
같다.

| case | asset 수 | final view | Cycles 호출/asset | wall time | 처리량/asset |
| --- | ---: | ---: | ---: | ---: | ---: |
| matched control | 1 | 8 | 21 | 22.715초 | 22.715초 |
| matched high-retry | 1 | 8 | 73 | 71.530초 | 71.530초 |
| high-retry, GPU당 worker 2개 | 2 | 16 | 73 | 89.910초 | 44.955초 |
| high-retry, camera fitting 비활성 진단 | 1 | 8 | 8 | 14.162초 | 14.162초 |

production 경로의 로컬 retry 배열은 control `[1,2,2,1,2,2,1,2]`, high-retry
`[8,8,9,8,8,9,7,8]`로 원격 기록과 완전히 같았다. 최종 radius와 4x4 camera
transform도 두 asset 모두 원격 대비 최대 절대 오차 `0.0`이었다. 각 실행은 8개의
512x512 RGBA PNG와 `transforms.json`을 정상 생성했다.

high-retry asset은 크기가 비슷한 control보다 3.15배 느렸다. 동일 asset에서 기존
연구용 `--lock_camera` 진단 옵션으로 radius fitting만 끄면 full Cycles 호출이
73회에서 8회로 줄고 wall time도 71.530초에서 14.162초로 5.05배 감소했다. 이 옵션은
camera output을 바꾸므로 output-preserving 개선안이 아니라 원인 toggle로만 사용했다.
따라서 최종 view마다 full Cycles로 alpha boundary를 찾는 retry loop가 rendering
병목의 직접 원인임을 로컬에서도 재현했다.

GPU dmon 기준 high-retry 단일 실행의 평균 SM utilization은 12.24%, active sample
평균은 16.31%, peak는 27%였고 memory-controller utilization 평균은 0.78%, peak
framebuffer memory는 1,803 MiB였다. worker 2개를 같은 GPU에서 실행해도 평균 SM은
13.26%, peak는 44%, peak memory는 3,602 MiB였다. 따라서 낮은 GPU utilization은
재현됐지만, coworker가 보았다는 수십 GB GPU memory는 rendering 단계에서는 재현되지
않았다.

같은 원격 chunk에는 64 asset, final view 512개, 실제 Cycles 호출 2,055회가 있었고
6개 Blender process의 기록된 render wall time은 1,573.55초였다. 두 로컬 단일-process
측정으로 fitting한 단순 모델은 같은 호출량을 6-worker에서 약 353.54초로 추정하므로,
원격 절대 시간에는 약 4.45배의 추가 slowdown이 남는다. 이 값은 두 asset에서 외삽한
모델이므로 원격 전체 asset 복잡도, NFS I/O, 동시 process 경쟁과 당시 system load를
분리한 직접 측정은 아니다.

결론적으로 camera retry에 따른 반복 렌더와 낮은 GPU 활용은 로컬에서 재현됐고,
원격의 절대적인 4배 추가 slowdown 및 수십 GB memory는 재현되지 않았다. 다음 개선
실험에서는 최종 Cycles shading을 유지한 채 radius 탐색용 alpha silhouette만 저비용
renderer로 대체하고, 원격과 retry 결정/final camera transform이 일치하는지 먼저
검증해야 한다.

## 2026-09-02 — n17 NFS I/O 및 GPU high-memory 원인 조사

`n17jh`의 `/home/rvi/ns3`와 로컬 `n1jh`의 같은 mount를 순차 측정했다. 로컬 mount는
`172.30.1.7:/file3` NFSv4.0, n17 mount는 `10.20.22.215:/file3` NFSv4.2이고 둘 다
1 MiB `rsize/wsize`다. Coworker container의 `/root/data3`는 n17 host의
`/file3/youngwoo` bind mount이므로 같은 `10.20.22.215:/file3` NFS client를 사용한다.
측정 시작 시 두 호스트의 GPU는 idle이었고, n17 load average는 약 2.95였다.

### NFS benchmark

두 호스트에서 동시에 부하를 주지 않고 512 MiB direct sequential I/O와 Pixal3D가
반복하는 작은 파일 overwrite/read 패턴을 측정했다.

| workload | local n1jh | n17jh | n17/local 시간비 |
| --- | ---: | ---: | ---: |
| 512 MiB direct sequential write | 4.663초, 115 MB/s | 12.838초, 41.8 MB/s | 2.75x |
| 512 MiB direct sequential read | 3.204초, 168 MB/s | 72.818초, 7.4 MB/s | 22.73x |
| 같은 256 KiB 파일 direct overwrite 128회 | 12.689초 | 94.475초 | 7.45x |
| 같은 256 KiB 파일 direct read 128회 | 6.454초 | 81.558초 | 12.64x |
| 160 KiB buffered overwrite 후 즉시 read 64회 | 13.467초 | 62.695초 | 4.66x |

서로 다른 256 KiB 파일 128개를 한 번씩 쓰고 읽는 짧은 시험에서는 n17이 더 빠르게
나왔지만, 이는 NFSv4.0/4.2 client cache와 metadata 상태의 영향을 크게 받는 패턴이다.
실제 renderer처럼 같은 경로를 overwrite하고 즉시 다시 읽는 시험에서는 일관되게
n17이 4.66~12.64배 느렸다.

완료된 실제 chunk
`ObjaverseXL_sketchfab-00008/batch015/chunk003`에서도 같은 차이를 확인했다. GLB 64개는
총 583,855,516 bytes(평균 9.12 MB), 최종 PNG 512개는 총 78,382,202 bytes(평균
153,090 bytes)였다.

| 실제 파일 metadata scan | local n1jh | n17 coworker container | n17/local |
| --- | ---: | ---: | ---: |
| GLB 64개 `stat` | 1.137초 | 51.728초 | 45.5x |
| PNG 512개 `stat` | 0.524초 | 78.658초 | 150.1x |

누적 NFS mountstats에서도 local/remote READ RPC 평균 RTT는 각각 약 18.4/98.5 ms였고,
n17 WRITE RPC에는 평균 약 28.7초의 누적 queue time이 기록돼 있었다. 이는 packet loss
증거라기보다 공유 1 Gbps 경로와 server/client writeback congestion의 증거다. n17의
순차 write는 약 334 Mbps, read는 약 59 Mbps로 1 Gbps line rate에도 훨씬 못 미쳤다.

### 관측된 render slowdown 중 NFS 비중

이 chunk의 원격 render wall time은 1,573.55초이고, 로컬 두 asset의 Cycles 호출당
모델을 6-worker에 적용한 값은 353.54초이므로 설명해야 할 차이는 1,220.01초다.
실제 코드는 camera fitting retry마다 PNG를 다시 쓰고 `Image.open`으로 즉시 읽으며,
이 chunk에서는 최종 view 512개를 얻기 위해 2,055번의 Cycles write/read가 발생했다.

- GLB payload read 차이: 약 75.71초
- GLB/PNG 최초 metadata scan 차이: 약 128.73초
- 평균 PNG 크기로 보정한 2,055회 buffered overwrite/read 차이: 약 1,476.95초

현재 시점 synthetic I/O를 그대로 대입하면 NFS 차이만으로 관측된 1,220초를 모두
설명하고도 남는다. 당시 cache 효과와 load 변동을 고려해 retry I/O는 최종 512개만
materialize된다고 극단적으로 보수적으로 잡아도 약 572.41초, 즉 관측 추가 지연의
46.9%다. 실제 코드는 모든 retry에서 write/read하므로 합리적인 결론은 NFS가 추가
slowdown의 최소 절반, 당시 상태에 따라 거의 전부를 설명하는 지배적 병목이라는 것이다.
현재 측정의 n17 I/O가 과거 chunk 실행 때보다 나빴을 수 있으므로 1,476.95초를 과거
실행의 정확한 분해값으로 사용해서는 안 된다.

### GPU high-memory 및 leak 판정

조사 시점에는 n17 GPU process가 없어 실시간 process-level leak trace를 새로 얻지는
못했다. 대신 2026-07-17~08-04 escalation 193개에서 38,416개의 중복 제거 GPU sample을
집계하고, encoder 코드와 당시 command line을 함께 조사했다.

- encoder escalation 40개에서 memory peak는 94.97 GiB였다. 전체 중복 제거
  telemetry에는 30 GiB 이상이면서 utilization 5% 이하인 sample도 502개 있었다.
- 30 GiB 이상의 memory가 1 GiB 이내로 변하지 않은 plateau window가 16개 있었고,
  최장 window는 약 1,189초였다. 반면 일부 종료 sample에서는 30~55 GiB peak가
  process 종료 후 약 1.59 GiB baseline으로 반환됐다.
- 5 GiB 이상 증가한 window도 17개 있었지만 escalation telemetry는 해당 command만이
  아니라 8개 GPU의 다른 동시 job까지 기록한 host-global snapshot이라 asset별 leak의
  증거로 귀속할 수 없다.
- 명시적 CUDA OOM report는 없었다. `exit status 245` 여섯 건은 Linux의
  `256 - SIGSEGV(11)`에 해당하며 memory leak 판정 근거가 아니다.

High-memory의 직접 원인은 코드와 설정에서 확인됐다.

1. 1024 encoder는 `micro_batch_size=4`, `gpu_memory_target_percent=80`이고 rank 3개가
   각각 model을 GPU에 올린다. 따라서 수십 GiB 사용은 설정상 허용된 동작이다.
2. batch가 80%를 넘으면 다음 batch size만 절반으로 줄이고 `empty_cache()`를 호출하지
   않는다. `empty_cache()`는 OOM retry에서만 호출되므로 `nvidia-smi`에는 PyTorch
   caching allocator의 high-water reserved memory가 idle 구간에도 남는다. 80% target은
   hard cap이 아니어서 이미 실행한 큰 batch는 95 GiB 가까이 도달할 수 있다.
3. encoder output은 GPU tensor인 채 bounded saver queue로 넘어간다. saver는 NFS에서
   temporary NPZ 생성, validation read, rename, parent directory `fsync`, final validation
   read를 끝낸 뒤에야 tensor 참조를 놓는다. 느린 n17 NFS 동안 GPU utilization은 0에
   가깝지만 output 및 allocator reservation은 계속 남는다.
4. queue는 loader/saver worker 수에 비례해 bounded이고, asset별 GPU output을 영구
   list에 누적하는 경로는 없었다. stage process가 끝나면 memory도 반환됐다.

따라서 현재 증거로는 unbounded GPU memory leak이 주원인이 아니다. `80%`를 목표로 한
adaptive micro-batch, allocator cache, 그리고 NFS saver backpressure가 결합해
`VRAM 수십 GB + GPU utilization 거의 0%` 상태를 만든다. 다만 완전한 leak 부재 증명은
동일 encoder를 실행하면서 `memory_allocated`와 `memory_reserved`를 asset index별로
기록해야 하며, 현재 idle 상태에서는 그 동적 검증을 수행하지 않았다.

## 2026-09-02 — n17 coworker container를 n1으로 이식한 동일 render 재현

n17의 실행 중인 `youngwoo_diyscene` container는 중단하지 않고 `--pause=false`로
commit했다. 원래 bind mount라 commit에 포함되지 않는 coworker worktree
`/root/dev/Pixal3D/.worktrees/anchor-hybrid-prestudy`(5.2 GB, source commit
`3ff6fca5fa65246e9b85b90d4c86a0aebb488d4a`, dirty status 870개)는 별도 staging
container를 통해 `/opt/pixal3d-repro`에 포함했다. 최종 image는 다음과 같다.

- image: `pixal3d-youngwoo-repro:20260902`
- image ID: `sha256:008f75a9467575518e7450dfc2b082be6fd50cf7cfc02576ec6f3f3893a03d40`
- 비압축 image 크기: 292,819,479,021 bytes
- 포함 runtime: Python 3.11.15, Blender 4.5.1 LTS

`ssh n1`은 n1jh에서 SSH key exchange 중 reset되어, n17을 ProxyJump로 사용해 실제
n1 host `rvi-node001`의 port 55555에 접속했다. export 대상은 요청한
`/home/rvi/ns1/jaehyeok/docker/pixal3d-youngwoo-repro-20260902.tar.gz`이며, n17에서는
같은 파일이 `/file1/jaehyeok/docker/...`로 보인다.

| 작업 | 결과 | wall time |
| --- | ---: | ---: |
| n17 `docker save` + `pigz -1 -p4` export | 201,218,264,834 bytes | 1시간 14분 29초 |
| SHA-256 전체 파일 scan | `65644f0471d1e7886ae8c01636bed21d11cb8fe6baf36523a0bc2f1808fc1b00` | 약 23분 |
| n1 `pigz -dc -p4 \| docker load` | image ID 일치, gzip CRC와 image 구조 검증 통과 | 1시간 15분 8초 |

export 첫 약 20분 동안에는 Docker가 204 GB writable layer의 임시 tar를 준비해 gzip
header 10 bytes만 보였다. n1 import는 비압축 tar 임시본과 overlay2 layer를 동시에
유지해 peak disk 사용량이 image 크기보다 컸지만, 완료 후 임시 공간이 회수되어
`/data2` 여유는 565 GB였다. 이 287~293 GB 환경 자체가 지나치게 커서 container
복제 비용만 2시간 30분가량 든다는 점도 확인됐다.

### 같은 코드·데이터의 n1 1-GPU 측정

loaded image를 새 container로 `docker run`하고 GPU 0 하나만 할당했다. 입력은 n1의
빠른 `/file3/youngwoo` mount, metadata는 `/file2/youngwoo`, 출력은 n1 local NVMe
`/data4`를 사용했다. production `data_toolkit/render_cond.py`를 OPTIX, 512x512,
8 condition views, `max_workers=1`로 실행했으며, 이전 로컬 재현과 같은 control 및
high-retry GLB를 사용했다.

| case | 측정 | container 포함 wall | render_cond asset time | prior n1jh baseline |
| --- | --- | ---: | ---: | ---: |
| control | cold | 34.32초 | 25.88초 | 22.715초 |
| control | warm repeat | 29.07초 | 22.68초 | 22.715초 |
| high-retry | cold | 105.98초 | 99.26초 | 71.530초 |
| high-retry | warm repeat | 97.01초 | 91.34초 | 71.530초 |

control warm 결과는 기존 n1jh baseline과 0.04초 이내로 일치했다. high-retry 측정
당시 n1 load average는 `66.79 / 50.38 / 44.20`(96 cores)였고, 다른 사용자의
container process 하나가 약 20.7 CPU cores를 사용 중이었다. 따라서 high-retry의
약 28% 차이는 n1 host contention이 섞인 값이며, n17 NFS slowdown으로 해석하면 안
된다. 현재 부하에서도 과거 n17 chunk로부터 역산한 약 298초/asset보다 훨씬 짧았다.

GPU dmon 결과는 다음과 같다.

| case | 평균 SM | peak SM | peak framebuffer memory |
| --- | ---: | ---: | ---: |
| control cold | 3.38% | 24% | 1,809 MiB |
| high-retry cold | 7.23% | 29% | 1,803 MiB |

즉 동일 image를 n1에서 실행해도 camera retry 때문에 GPU utilization이 낮은 현상은
재현되지만, render stage 자체에서 수십 GB VRAM은 재현되지 않았다. 수십 GB plateau는
앞 절에서 조사한 1024 encoder의 adaptive batch/caching allocator 및 느린 NFS saver
backpressure 설명과 일치한다.

두 asset 모두 8개의 512x512 RGBA PNG와 `transforms.json`을 만들었고 alpha channel,
scene AABB/scale/offset, camera transform/radius가 기존 결과와 일치했다. high-retry의
두 `camera_angle_x` 값만 최대 약 `2e-16`의 부동소수점 반올림 차이가 있었다. PNG RGB
byte는 기존 파일과 같지 않았지만, 같은 n1 image에서 control을 즉시 재실행해도 run간
RGB MAE가 14.20/255였고 alpha/camera는 완전히 같았다. 따라서 이는 server migration
차이가 아니라 현행 Blender script가 retry마다 unseeded random lighting을 생성하는
기존 비결정성이다.

결론적으로, 동일 coworker 환경을 n1으로 옮기면 보통 asset이 기존 n1jh 기준 속도로
돌아왔다. 따라서 n17에서 `/home/rvi/ns3`를 직접 read/write하며 작업하는 것이 원격의
추가적인 절대 slowdown의 주된 문제라는 기존 판단이 강화됐다. 다만 camera-radius
retry로 같은 view를 여러 번 Cycles render하는 알고리즘 병목은 n1에서도 남으므로,
NFS를 고쳐도 low GPU utilization과 asset별 큰 latency 편차 자체는 사라지지 않는다.

## 2026-09-03 — n1/n17 동일 image의 격리된 1-GPU NFS output A/B

위 절의 “ns3가 추가 slowdown의 주된 원인”이라는 결론을 직접 토글 실험으로 다시
검증했다. 두 host에서 image ID가 완전히 같은
`pixal3d-youngwoo-repro:20260902`
(`sha256:008f75a9467575518e7450dfc2b082be6fd50cf7cfc02576ec6f3f3893a03d40`)
를 사용했다. production `data_toolkit/render_cond.py`, OPTIX, 512x512, condition view
8개, `max_workers=1`, GPU 한 개로 고정했다.

### 격리와 실험 범위

- coworker의 `youngwoo_diyscene` container에는 `exec`, restart, stop을 하지 않았다.
- 원본 `/file3/youngwoo/.../chunk003/source`는 staging 때 `:ro`로만 mount했다.
- 입력 GLB와 두 행짜리 metadata는 host-local scratch에 복사해 read-only mount했다.
- n1 출력은 `/data4/jaehyeok/pixal3d-ab-20260903`, n17 local 출력은
  `/tmp/pixal3d-ab-20260903`, ns3 출력은
  `/file3/jaehyeok/pixal3d-ab-20260903` 아래의 실행별 고유 디렉터리만 사용했다.
- 따라서 아래 A/B는 같은 n17 GPU와 local input을 유지하고 **output 위치만 local에서
  ns3로 바꾼 측정**이다. 원본의 167 MB global metadata 및 NFS input overhead는
  포함하지 않는다.

control은 일반 샘플
`431f607a090666a1b75d552b7eefa9600cbef4a69f713e74a36b800cd8d73163`,
high-retry는 camera-radius 조건으로 같은 mesh를 더 많이 다시 render하는
`4320138ad568bc104499f757b32aa831a0ae25500fffaf4ab30534be22ff5ea8`이다.
warm-up 뒤 반복 실행했으며, 다른 작업이 겹쳐 n17 GPU 0 framebuffer가 각각
75,946 MiB와 53,080 MiB까지 증가한 control 두 회는 결과에서 제외했다.

| sample | n1 local wall (반복) | n1 중앙값 | n17 local wall (clean 반복) | n17 local 중앙값 | n17 ns3-output wall (clean 반복) | ns3 중앙값 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 24.05 / 24.01초 | 24.03초 | 30.40 / 32.59초 | 31.50초 | 34.11 / 38.34초 | 36.23초 |
| high-retry | 68.47 / 67.94초 | 68.21초 | 95.66 / 102.79초 | 99.23초 | 107.60 / 113.00 / 115.48초 | 113.00초 |

n17 내부 output 토글의 중앙값 차이는 control `+4.73초`(`+15.0%`), high-retry
`+13.78초`(`+13.9%`)였다. 별도의 깨끗한 GPU 4 연속 pair에서도 control은
30.40→34.11초(`+3.71초`, `+12.2%`), high-retry는 95.66→107.60초
(`+11.94초`, `+12.5%`)로 같은 방향과 크기가 재현됐다. output 하나의 실제 크기는
약 0.9~1.95 MB였다.

반면 local output끼리 비교해도 n17은 n1보다 control `+7.47초`(`+31%`),
high-retry `+31.02초`(`+45%`) 느렸다. n1은 RTX 3090, n17은 RTX PRO 6000
Blackwell이라 이 host 간 차이를 단일 원인으로 분해할 수는 없지만, n17의 추가 지연이
NFS output만의 문제는 아니라는 점은 분명하다. n1 local에서 n17 ns3 output까지의
총 차이 중 NFS output toggle이 차지하는 비율은 control 약 39%, high-retry 약 31%다.

깨끗한 n17 GPU 4 실행의 평균 SM은 control 1.70~2.22%, high-retry 1.80~1.98%,
peak framebuffer는 3,998~4,004 MiB였다. 따라서 낮은 GPU utilization은 재현되지만
render stage에서 수십 GiB VRAM은 재현되지 않았다. 수십 GiB 현상은 앞서 확인한
encoder의 adaptive batch/caching allocator와 saver backpressure 설명에 더 가깝다.

### 결과 동일성 및 안전 확인

retained run은 모두 exit 0이었고 각 sample에서 512x512 RGBA PNG 8개와 frame 8개의
`transforms.json`을 생성했다. n1/n17 사이에서 `selected_devices`만 제외한 scene/camera
metadata SHA-256와 8장 전체 alpha SHA-256가 두 sample 모두 정확히 일치했다. RGB는
같은 host 반복에서도 달라지는 unseeded lighting의 기존 비결정성이 있었지만,
camera/scene/alpha 계약은 동일했다.

실험 뒤 원본 GLB 두 개의 SHA-256가 asset ID와 그대로 일치하고,
`youngwoo_diyscene`가 계속 running이며, `pixal3d-ab-20260903-*` 임시 container가
0개임을 확인했다. 원본 NFS input까지 포함하는 추가 측정은 시작 직전 coworker가
8-GPU evaluation을 실행해 모든 GPU를 점유했으므로 container launch 전에 중단했다.

### 수정된 결론

ns3 output latency는 실제로 존재하지만 이 두 representative asset에서는 약
12~15%이고, 과거 chunk에서 역산한 약 298초/asset을 단독으로 설명할 정도의 주원인은
아니다. 더 큰 요인은 camera retry에 따른 asset별 반복 render, n17 host/GPU runtime
차이, 동시 작업에 의한 contention이다. 따라서 위 절의 “ns3 직접 사용이 추가 절대
slowdown의 주된 문제”라는 표현은 이번 직접 A/B 결과로 철회한다. 다음 최적화 우선순위는
NFS staging만이 아니라 같은 Blender process에서 여러 view/retry를 처리해 startup과
scene import를 재사용하는 rendering 경로다.

## 2026-09-03 — n1/n17 전체 전처리 단계별 1-GPU wall-time 분해

render만 비교했던 앞 실험을 전체 preprocessing leaf command로 확장했다. 두 host 모두
같은 `pixal3d-youngwoo-repro:20260902` image
(`sha256:008f75a9467575518e7450dfc2b082be6fd50cf7cfc02576ec6f3f3893a03d40`),
같은 control/high-retry GLB 두 개, host-local output, GPU 한 장, leaf command당 worker
하나를 사용했다. 최종 output은 view 0/1만 학습 대상으로 만들었지만 condition render는
production 설정 그대로 asset당 8개 512x512 view를 생성했다.

측정한 경로는 다음과 같다.

1. 167 MB global metadata와 raw metadata scan, GLB 두 개(합계 8.86 MB) local staging
2. `dump_mesh`, `dump_pbr`, `asset_stats`
3. OPTIX condition render 8 view
4. 256/512/1024 각각 shape dual-grid와 PBR voxelization, view 0/1
5. 256/512/1024 각각 shape/PBR latent encoding, view 0/1
6. 1024 shape latent에서 SS64 encoding, view 0/1
7. 모든 최종 PNG/JSON/NPZ/CSV 실제 readback validation

### n1 Ampere 호환화와 측정 규칙

복제 image의 FlexGEMM binary는 n17 Blackwell용으로만 컴파일되어 n1 RTX 3090에서 첫
shape encode가 `no kernel image is available`로 실패했다. 양쪽 설치 metadata가 동일한
upstream commit `6dd94a859c26ee8246888502eada3dd8ad85532e`를 가리키는 것을 확인하고,
임시 n1 container 안에서만 그 commit을 `TORCH_CUDA_ARCH_LIST=8.6`으로 다시 빌드했다.
Pixal3D source, model, PyTorch `2.8.0+cu128`, 입력 및 image 자체는 바꾸지 않았다.

처음 n1 shape encode에는 새 sparse kernel의 Triton/JIT/autotune이 섞여
256/512/1024가 각각 94.125/33.174/44.124초 걸렸다. 이 비용은 image를 새로운 GPU
architecture에 처음 적응시키는 1회성 setup이므로, 별도 output root에서 같은 명령을
warm repeat한 8.470/9.262/12.321초를 정상 preprocessing 표에 사용했다. 두 repeat의
공통 shape/PBR NPZ 24개는 schema뿐 아니라 byte-level array 값도 모두 같았다.

모든 표의 시간은 command 시작부터 process 종료까지의 wall time이라 Python/model
startup, 파일 read/write, fsync와 record CSV publish를 포함한다. 단계들은 병렬 합산이
아니라 실제 pipeline 순서대로 하나씩 실행했으며, 아래 합계는 두 asset의 latency이다.

### 큰 단계별 총시간과 비중

| 단계 | n1 wall | n1 비중 | n17 wall | n17 비중 | n17 - n1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| metadata/GLB staging | 4.004초 | 0.88% | 8.368초 | 1.92% | +4.364초 |
| prepare 3종 합계 | 13.703초 | 3.00% | 11.536초 | 2.64% | -2.167초 |
| condition render | 88.057초 | 19.29% | 138.340초 | 31.70% | +50.283초 |
| geometry 256 합계 | 17.457초 | 3.82% | 14.653초 | 3.36% | -2.804초 |
| geometry 512 합계 | 56.489초 | 12.37% | 44.391초 | 10.17% | -12.098초 |
| geometry 1024 합계 | 210.399초 | 46.09% | 151.127초 | 34.63% | -59.272초 |
| encoding 256 합계 | 16.751초 | 3.67% | 17.383초 | 3.98% | +0.632초 |
| encoding 512 합계 | 18.384초 | 4.03% | 19.045초 | 4.37% | +0.661초 |
| encoding 1024 합계 | 24.421초 | 5.35% | 26.761초 | 6.13% | +2.340초 |
| SS64 encode | 6.693초 | 1.47% | 4.578초 | 1.05% | -2.115초 |
| final readback validation | 0.168초 | 0.04% | 0.173초 | 0.04% | +0.004초 |
| **전체 2 assets** | **456.527초** | **100%** | **436.355초** | **100%** | **-20.172초** |
| **관측 평균/asset** | **228.263초** |  | **218.177초** |  | **-10.086초** |

따라서 이 2-asset local-output latency 실험에서는 n17이 전체적으로 느리지 않았다.
n17 render는 n1보다 50.28초 느렸지만, CPU-only 1024 geometry가 59.27초 빨라 전체는
n17이 약 4.4% 빨랐다. 두 asset 중 하나가 의도적으로 high-retry이므로 이 평균을 임의의
Objaverse asset 평균으로 일반화하면 안 된다.

### leaf command별 상세 분해

| leaf command | 작업량 | n1 wall / 비중 | n17 wall / 비중 |
| --- | ---: | ---: | ---: |
| `dump_mesh` | 2 assets | 2.984초 / 0.65% | 2.764초 / 0.63% |
| `dump_pbr` | 2 assets | 8.750초 / 1.92% | 6.972초 / 1.60% |
| `asset_stats` | 2 assets | 1.969초 / 0.43% | 1.800초 / 0.41% |
| `render_cond` | 16 final views, 94 Cycles calls | 88.057초 / 19.29% | 138.340초 / 31.70% |
| `dual_grid_256` | 4 asset-views | 8.770초 / 1.92% | 7.441초 / 1.71% |
| `voxelize_pbr_256` | 4 asset-views | 8.687초 / 1.90% | 7.212초 / 1.65% |
| `dual_grid_512` | 4 asset-views | 28.195초 / 6.18% | 21.605초 / 4.95% |
| `voxelize_pbr_512` | 4 asset-views | 28.294초 / 6.20% | 22.786초 / 5.22% |
| `dual_grid_1024` | 4 asset-views | 107.315초 / 23.51% | 75.952초 / 17.41% |
| `voxelize_pbr_1024` | 4 asset-views | 103.084초 / 22.58% | 75.175초 / 17.23% |
| `encode_shape_256` | 4 asset-views | 8.470초 / 1.86% | 8.973초 / 2.06% |
| `encode_pbr_256` | 4 asset-views | 8.281초 / 1.81% | 8.410초 / 1.93% |
| `encode_shape_512` | 4 asset-views | 9.262초 / 2.03% | 9.930초 / 2.28% |
| `encode_pbr_512` | 4 asset-views | 9.122초 / 2.00% | 9.115초 / 2.09% |
| `encode_shape_1024` | 4 asset-views | 12.321초 / 2.70% | 13.193초 / 3.02% |
| `encode_pbr_1024` | 4 asset-views | 12.100초 / 2.65% | 13.568초 / 3.11% |
| `encode_ss_64` | 4 asset-views | 6.693초 / 1.47% | 4.578초 / 1.05% |

1024 geometry 한 output당 겉보기 wall은 n1에서 shape 26.83초, PBR 25.77초이고
n17에서 shape 18.99초, PBR 18.79초였다. 반면 1024 encoder는 model startup까지
포함하고도 output당 약 3.0~3.4초였다. 따라서 이 구성에서 geometry 최적화는 encoder
micro-batch 조정보다 latency에 미치는 영향이 훨씬 크다.

### render retry와 GPU/VRAM

두 host의 `transforms.json`에 기록된 retry 수는 완전히 같았다.

| asset | view별 retry | retry 합계 | 최종 view 포함 Cycles calls | n1 asset time | n17 asset time |
| --- | --- | ---: | ---: | ---: | ---: |
| control `431f...3163` | 1/2/2/1/2/2/1/2 | 13 | 21 | 약 20.57초 | 약 30.36초 |
| high-retry `4320...ea8` | 8/8/9/8/8/9/7/8 | 65 | 73 | 약 65.23초 | 약 105.86초 |

즉 final PNG는 asset당 8장이지만 실제 Cycles 호출은 각각 21회와 73회였다. 이번 묶음의
render wall을 호출 수로 나누면 n1 약 0.94초/call, n17 약 1.47초/call이다. high-retry
asset의 절대시간이 긴 직접 원인은 mesh를 다시 import해서가 아니라 한 Blender process
안에서 boundary fitting 때문에 같은 view를 최대 10회까지 full render하는 루프다.

0.5초 간격 `nvidia-smi` telemetry는 다음과 같았다.

| stage | n1 평균/peak SM | n1 peak VRAM | n17 평균/peak SM | n17 peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| render | 8.93% / 30% | 1,809 MiB | 1.70% / 12% | 4,004 MiB |
| shape encoder 256 | 3.76% / 32% | 2,037 MiB | 0.06% / 1% | 2,390 MiB |
| shape encoder 512 | 9.37% / 85% | 6,403 MiB | 1.00% / 20% | 6,754 MiB |
| shape encoder 1024 | 20.56% / 100% | 23,609 MiB | 4.78% / 98% | 26,044 MiB |
| PBR encoder 1024 | 16.00% / 100% | 21,941 MiB | 4.19% / 98% | 22,292 MiB |

500 ms sampling은 짧은 kernels를 놓치므로 평균 SM을 kernel 효율로 해석하면 안 된다.
다만 render가 low-utilization이어도 VRAM은 1.8~4.0 GiB뿐이고, 1024 encoder가 실제로
22~26 GiB까지 reserve/use한다는 구분은 명확하다. coworker가 본 “수십 GiB VRAM인데
GPU utilization은 거의 0%”는 render가 아니라 큰 encoder batch, caching allocator,
CPU loader/NFS saver 사이의 대기구간과 일치한다.

n17의 1024 geometry telemetry에는 34~47 GiB 및 높은 SM이 동시에 보였지만 이 값은
해당 stage에 귀속하지 않았다. 실제 실행 source는 vertices/faces와 O-Voxel input을
명시적으로 CPU tensor로 유지하고, 같은 단계의 n1은 GPU memory 3 MiB 이하였다. 따라서
n17 geometry 시각의 GPU sample에는 다른 process가 섞였으며 wall-time 표만 유효하다.

### output readback과 cross-host 차이

각 host의 최종 결과 82개, 약 13.9 MB를 전부 다시 읽었다.

- RGBA 512x512 PNG 16개와 8-frame `transforms.json` 2개
- shape/PBR/SS NPZ 28개: key, shape, dtype, non-empty, 모든 numeric value finite
- scale JSON 28개와 record CSV 8개

모든 검사가 통과했다. n1/n17 camera metadata, 16장 alpha, scale JSON은 정확히 같았다.
render RGB는 기존에 확인한 unseeded lighting 때문에 run마다 달라져 MAE가 11.03/255였다.
latent는 두 GPU architecture에서 schema/dtype/shape가 모두 같았으나 bitwise equal은
아니었다. mean absolute difference는 shape 약 0.001792, PBR 약 0.001799, SS 약
0.000272이고 최대는 각각 0.01473/0.02097/0.00241이었다. 이는 host를 바꿀 때의
CUDA 수치 차이이며, 같은 n1 warm repeat 24개는 bitwise identical이었다.

### production 64-asset chunk와의 차이

위 2-asset 표는 단일 GPU latency를 상세 분해하기 위한 것이고, 실제 coworker의
64-asset chunk는 여러 worker/rank에서 startup이 amortize되고 render asset 구성이
달라진다. 이미 완료된 n17 production 기록의 합계 5,164.01초를 같은 방식으로
정규화하면 다음과 같다.

| production 구간 | wall | 비중 | asset당 환산 |
| --- | ---: | ---: | ---: |
| prepare | 265.46초 | 5.14% | 4.15초 |
| render | 3,079.38초 | 59.63% | 48.12초 |
| geometry 256 | 97.92초 | 1.90% | 1.53초 |
| encoding 256 | 48.83초 | 0.95% | 0.76초 |
| geometry 512 | 285.77초 | 5.53% | 4.47초 |
| encoding 512 | 74.47초 | 1.44% | 1.16초 |
| geometry 1024 | 1,116.97초 | 21.63% | 17.45초 |
| encoding 1024 | 140.11초 | 2.71% | 2.19초 |
| SS/final | 55.10초 | 1.07% | 0.86초 |
| **합계** | **5,164.01초** | **100%** | **80.69초** |

실제 throughput에서는 render가 59.6%로 가장 크고 1024 geometry가 21.6%로 두 번째다.
반면 controlled pair에서는 high-retry 하나가 포함돼도 command startup을 asset 두 개만
나눴고 geometry를 worker 하나로 실행해 1024 geometry 비중이 더 커졌다. 최적화 우선순위는
production 비중 기준으로 **render retry/process 재사용 → 1024 geometry 병렬화/재사용 →
encoder memory/saver backpressure** 순서가 합리적이다.

### “모든 preprocessing” 범위 판정

현재 coworker checkpoint가 최종 학습 artifact로 인정하는 경로는 SS encode 뒤
`validate_outputs`까지다. 위 실험은 그 산출물 계약을 모두 생성하고 readback했다.
다만 orchestrator의 더 새로운 command graph에는 이후 `build_packs`, `archive_raw`,
`cleanup_local`이 있다. 이들은 학습 tensor 생성이 아니라 packaging/retention 단계이며
coworker의 현재 측정 checkpoint에는 없어서 이번 총시간에 포함하지 않았다. 각 resolution의
voxel 파일 삭제 자체도 별도 계측하지 않았다. 따라서 “현재 coworker가 수행 중인 feature
preprocessing 전체”는 평가했지만, 선택적으로 이어질 dataset pack/archive 운영시간까지
평가한 것은 아니다.

모든 실험은 coworker의 `youngwoo_diyscene`에 `exec`/restart/stop 없이 진행했고 원본
`/file3/youngwoo` 입력은 read-only로만 사용했다. 임시 결과와 container는 조사 완료 뒤
삭제했다.

## 2026-09-04 — 1-GPU autoresearch 기준선과 품질 가드

전처리 최적화가 일부 stage만 빠르게 보이는 것을 막기 위해 프로젝트 안에 고정된 전체
pipeline benchmark를 추가했다. 입력은
`/home/rvi/ns3/jaehyeok/ds/Objaverse-full/glbs/000-082`의 representative GLB 두 개를
로컬 container overlay의 `/tmp/pixal3d-autoresearch-jaehyeok/inputs`로 한 번 복사하고,
모든 측정은 이 local copy에서 수행한다. 원본 NFS 데이터와 coworker output은 수정하지
않는다. 처음 계획한 `/data4/jaehyeok`은 현재 `n1jh` container에 mount되지 않아 같은
host-local overlay인 `/tmp`로 변경했다.

측정 범위는 mesh/PBR dump와 asset stats, 512x512 condition render 8 view, view 0-1의
shape/PBR geometry 256/512/1024, 각 shape/PBR latent, 1024 shape 기반 SS64, 단계별
metadata merge와 최종 readback이다. 단일 RTX 3090(`CUDA_VISIBLE_DEVICES=0`), worker 1,
고정 seed 20260904를 사용했다.

첫 retained baseline은 **301.552초/2 assets**였고, 변경 없는 warm repeat는
**296.411초**, 즉 측정 speedup **1.0173x**였다. 주요 baseline stage는 mesh/PBR/통계
prepare 약 29.9초, render 35.6초, dual-grid 83.2초, PBR voxelization 86.9초,
shape/PBR encoding 6종 약 57.1초, SS64 4.5초였다. 기준 결과는 164 files, 119MB이며
PNG 16개, VXZ 24개, NPZ 28개를 실제로 다시 읽어 검증했다.

변경 없는 repeat의 품질 비교는 최소 RGB PSNR 97.545dB, 평균 102.767dB, alpha IoU
1.0이었고 sparse coordinates와 모든 numeric artifacts가 guard를 통과했다. 이후 실험의
retention 기준은 image별 RGB PSNR 40dB 이상, 평균 45dB 이상, alpha IoU 0.999 이상,
camera JSON 1e-6 이내, sparse coordinate exact, attributes/latents absolute error 0.002
이내, 동일 artifact set/schema/dtype/finite 값이다. 속도 metric은 baseline wall-time을
candidate wall-time으로 나눈 `speedup`이며 목표는 4.0x 이상이다.

## 2026-09-04 — 1-GPU 전처리 4.33x 최적화 결과

동일한 2-asset 전체 pipeline에서 최종 wall time은 **301.552초에서 69.663초**로
감소해 **4.3287x** speedup을 달성했다. 단순히 2로 나눈 latency는 asset당
150.776초에서 34.831초로 줄었지만, 두 asset의 독립 stage를 겹쳐 실행하므로 이 값은
단일 asset을 따로 실행한 latency가 아니라 같은 처리량에서의 평균이다. 모든 측정은
RTX 3090 한 장만 사용했다.

최종 결과는 기준선과 같은 164 files(약 119MB), PNG 16개, VXZ 24개, NPZ 28개를
생성했다. 품질 비교 결과 최소 RGB PSNR **97.545dB**, 평균 RGB PSNR
**102.604dB**, alpha IoU **1.0**이었고, camera/scale JSON, sparse coordinates,
attributes/latents, artifact schema와 finite 검사를 모두 통과했다. 따라서 처음 논의한
느슨한 PSNR 허용치를 실제로 소모하지 않았으며, 출력 차이는 변경 없는 warm repeat에서
관측된 수준이다. 최종 regression suite는 **23 passed**였다.

retained 변경은 다음과 같다.

- asset별 CPU geometry 작업과 shape/PBR geometry를 독립 task로 나누고, resolution과
  view별 작업을 collision-free record tag로 동시에 실행했다.
- mesh dump, PBR dump, condition render처럼 서로 의존하지 않는 준비 단계를 겹쳐
  실행하고, 두 asset의 Blender render worker도 같은 단일 GPU에서 병렬화했다.
- resolution마다 shape/PBR encoder model을 다시 load하지 않고 한 process에서
  256/512/1024를 처리하도록 합쳤다. SS64도 같은 lifecycle에 포함했다.
- geometry가 생성되는 즉시 fused encoder가 결과를 기다렸다가 소비하도록 pipeline해
  CPU geometry와 GPU encoding의 유휴 구간을 겹쳤다.
- 최종 산출물에 필요하지 않은 중간 asset-stats metadata merge를 제거했다.

최종 run에서 dump mesh 6.44초, dump PBR 27.09초, render 23.94초가 서로 겹쳐
실행됐다. 12개 geometry task도 겹쳐 실행되어 각각 약 15.0~33.5초였고, fused
shape/PBR/SS encoding은 40.91초였다. 따라서 이 stage 시간은 합산하면 안 되며 전체
critical path가 69.663초다.

discard한 실험도 기록한다. boundary fitting만 낮은 해상도로 바꾸는 방식은 카메라
radius와 geometry까지 달라져 최소 PSNR 18.257dB, alpha IoU 0.693으로 guard를
통과하지 못했다. transformed mesh/PBR 재사용과 Cycles 1-sample probe도 속도 개선이
없어 되돌렸다. 즉 이번 4.33x 결과는 해상도나 render 품질을 낮춘 결과가 아니라 model
load 제거, 중복 lifecycle 통합, task concurrency와 CPU/GPU pipeline 개선으로 얻었다.

실험 로그와 HTML 보고서는
`autoresearch-results/archive/20260903-182012/`에 보존했다. 이 결과는 고정된
representative 2 assets의 전체 feature preprocessing 계약을 검증한 값이다. production
배포 전에는 더 다양한 100 assets에서 실패율, tail latency와 peak VRAM을 별도로
확인해야 한다.

## 2026-09-04 — coworker용 fast container 전달 구조 검증

n17의 실행 중인 `youngwoo_diyscene`는 `exec`, stop, restart, source 수정 없이
`docker inspect/top/logs`와 host bind source만 읽었다. 원본은 host
`/home/youngwoo/dev`를 container `/root/dev`에 RW bind하므로 image 내부 같은 경로에
코드를 넣으면 mount에 가려진다. 또한 원본 container writable layer가 204GB이고 n17
root 여유 공간이 282GB뿐이어서 live container를 다시 commit하는 방식은 배제했다.

대신 이미 보존된 동일환경 image `pixal3d-youngwoo-repro:20260902`
(`sha256:008f75a9...`, 292.82GB)를 parent로 사용했다. 기준선 `ff02d16`과 최적화본
`71b928e`의 clean Git archive를 각각 `/opt/pixal3d-validation`에 넣은 thin image를
만들었다. fast image는
`youngwoo_diyscene_fast:validation-20260904`
(`sha256:47896f1b...`)이며 추가 code layer는 79.6MB다. `latest`나 production tag는
사용하지 않았다.

두 headless container에는 GPU 0 하나, 동일 Objaverse 입력 디렉터리 `:ro`, 서로 다른
n17 host-local `/tmp` output만 mount했다. `/root/dev`, `/root/data*`, production queue와
control root는 mount하지 않았다. 두 image에서 다음 argv를 동일하게 실행했다.

```bash
conda run --no-capture-output -n pixal3d \
  python data_toolkit/benchmark_preprocess.py baseline \
  --workspace /validation \
  --blender /tmp/blender-4.5.1-linux-x64/blender \
  --source /inputs/d0fb772e66fb48dab7ab97749390d073.glb \
  --source /inputs/16be52ed0f3f4a06a3f10e2ed5db8f0f.glb
```

| image | 2-asset wall | 처리량 기준 평균 | speedup |
| --- | ---: | ---: | ---: |
| reference | 261.181초 | 130.590초/asset | 1.000x |
| fast | 64.413초 | 32.206초/asset | **4.0548x** |

품질 guard는 NPZ 28개, PNG 16개, VXZ 24개를 비교해 실패 0건, 최소 RGB PSNR
98.057dB, 평균 PSNR infinite, alpha IoU 1.0으로 통과했다. timestamped/internal
`merged_records`를 제외한 최종 계약 파일은 양쪽 모두 147개이며 상대경로가 정확히
같고 최종 `metadata.csv` SHA-256도 동일했다. fast 쪽 전체 파일이 164개가 아니라
170개인 이유는 concurrent view writer collision을 막는 geometry/PBR record shard가
해상도별로 하나씩 더 생겨 내부 CSV가 6개 증가했기 때문이다.

검증 후 원본 container ID, 시작 시각, restart count 0이 그대로였고 shared
`workers.json`/`units.json`과 입력 GLB checksum도 변하지 않았다. validation container는
모두 `--rm`으로 제거했다. 기준선 image와 scratch output도 검증 후 제거했고 fast image는
후속 최적화용 validation tag로만 남겼다.

중요하게도 **현재 image는 coworker에게 전달 가능한 production image가 아니다.**
coworker의 실제 명령은 production branch `12b2a638...`의
`python -m data_toolkit.pipeline.cli supervisor ...`인데, 현재 4.33x branch에는
`data_toolkit.pipeline`이 없다. fast image에서 같은 production CLI를 probe하면
`ModuleNotFoundError`로 종료됐다. container/image/mount 전달 방식과 전체 산출물
동일성은 검증됐지만, 실제 handoff 전에는 최적화된 command graph와 fused encoder를
production branch에 이식하고 `supervisor`/`benchmark-parallelism` CLI를 그대로
보존한 A/B를 다시 통과시켜야 한다.

## 2026-09-04 — production branch 이식 및 격리 1-GPU 검증

production 기준 commit `12b2a638b1399ad6d6683ab8bd57e0af03e331dc`에서 별도
`codex/production-preprocess-fast` worktree를 만들고, 기존 supervisor/parallelism,
checkpoint, pack/archive 계약을 유지한 채 최적화를 이식했다. 기존
`youngwoo_diyscene` container에는 exec/stop/restart/source 변경을 하지 않았다.

production DAG의 leaf command를 다음 두 lifecycle로 합쳤다.

- `prepare_bundle`: mesh dump, PBR dump, asset stats와 rank-expanded condition render를
  겹쳐 실행한다.
- `geometry_encode_bundle`: shape/PBR 256/512/1024, view 0/1 geometry를 독립 job으로
  실행하고 atomic geometry가 생기는 즉시 fused encoder가 소비한다. shape/PBR model은
  resolution마다 다시 load하지 않는다.

ABO adapter에는 metadata의 기존 local GLB가 SHA-256까지 맞으면 155GB 원본 TAR를 다시
열고 scan하지 않는 fast path를 추가했다. geometry 전체 native thread 수는 adaptive
profile의 CPU 예산 안으로 제한했고, encoder rank 실패 시 sibling process를 즉시
terminate/reap하며 geometry도 같은 poll loop에서 감시한다. terminate 후 5초 안에
끝나지 않는 child는 kill한다. family별 instance manifest는 ambient environment가 아니라
runner가 생성한 명시적 CLI 파일로만 전달한다. `prepare_bundle` 재시도에서는 실제
`--render_workers_per_gpu` argument가 downshift된다. family quality exclusion도 bundle
내부까지 전달해 PBR-only 제외 asset이 shape/SS는 처리하되 PBR voxel/encode를 불필요하게
재실행하지 않도록 했다. bundle leaf fan-out은 `ThreadPoolExecutor` 대신 fail-fast
subprocess monitor를 사용하며, outer supervisor는 worker가 비정상 종료하면 같은 process
group의 Blender/geometry descendants까지 TERM 후 5초 뒤 KILL하고 reap한다. 실제
grandchild가 60초 sleep 중인 회귀 테스트에서 parent exit 7 뒤 orphan 없이 종료됨을 확인했다.

최종 no-bind validation image는
`youngwoo_diyscene_fast:production-validation-20260904`
(`sha256:a23f50b1357a41927c554a5fd56b68de0da6c99f3dfc882e447f6b04fa394113`)다.
코드는 `/opt/pixal3d-production`에 bake돼 있고 `WORKDIR`와 `PYTHONPATH`도 그 경로를
가리키므로 coworker의 `/root/dev` bind mount가 image 코드를 가리는 문제가 없다.

n17에서 동일한 ABO 2 assets, headless, Docker `--gpus device=1`, validation config의
`gpu_count: 1`, 분리된 data2/data3 및 `/dev/shm` scratch로
`python -m data_toolkit.pipeline.cli run --gate smoke --count 2` 전체를 실행했다.
입력과 output은 기존 coworker 경로가 아닌 validation 전용 경로만 사용했다.

| production image | 전체 2-assets wall | 처리량 기준 평균 | reference 대비 |
| --- | ---: | ---: | ---: |
| unchanged `12b2a638...` | 253.06초 | 126.53초/asset | 1.000x |
| production-fast exact final | 160.62초 | 80.31초/asset | **1.575520x** |

최종 run은 8개 pack family(common, shape 3종, PBR 3종, SS64)를 모두 만들고 audit를
exit 0으로 통과했다. reference/candidate raw archive SHA-256은
`516a5e53ebf0e8eac3f148b9606725c727cecf3183cfadb0e1e0d3ba610b6726`로 동일했다.
28 NPZ 안의 52 arrays는 모두 exact(`max_abs=0`), 30 JSON도 byte-exact였으며 alpha
IoU는 1.0이었다. 동일 모델 GPU 1의 telemetry는 759 samples, peak 11,140MiB,
sampled mean utilization 1.22%, non-zero sample 22.3%, peak 97%였다.
0.2초 `nvidia-smi`
sampling은 짧은 kernels 사이 CPU 구간을 크게 포함하므로
mean utilization 자체를 kernel 효율로 해석하지 않는다.

RGB PSNR은 최종 reference/candidate run에서 최소 16.676dB, 평균 23.615dB였지만 unchanged
reference를 다시 실행해도 최소 17.616dB, 평균 23.855dB였다. 두 reference run 사이에도
52 arrays와 30 JSON은 전부 exact였다. 원인은 `render_cond.py`가 각 retry마다 random
lighting을 seed 없이 다시 뽑는 기존 동작이다. 따라서 독립 run의 RGB PSNR 40/45dB를
production 회귀 gate로 쓰는 것은 유효하지 않고, camera/latent/geometry exactness와 alpha
IoU를 우선해야 한다. 이전 autoresearch의 fixed-seed benchmark에서는 97dB 이상이었으므로
그 결과와 모순되지 않는다.

최종 image 내부 regression suite는 **833 passed, 1 warning**이며 warning은 기존
`torch.cross` deprecation뿐이다. `supervisor --help`와
`benchmark-parallelism --help`가 통과했고, 같은 1-GPU config에서
`benchmark-parallelism --count 32 --dry-run`은 `assets=32`, `chunk_assets=32`,
views 0/1, resolutions 256/512/1024를 출력하고 exit 0이었다. live
`youngwoo_diyscene`는 검증 뒤에도 동일 ID `a21b8f...`, running, restart count 0이다.

non-dry `benchmark-parallelism`을 동일 2 assets로도 실행해 보았으나 이 command는
`count == chunk_assets`를 요구하고 production schema는 `chunk_assets`를 32 또는 64로만
허용해 의도대로 거부했다. 이 계약을 2로 완화하면 production 명령 호환성 검증이 아니므로
코드는 바꾸지 않았다. 따라서 이번 단계는 실제 `run` 2-asset A/B와 count 32 dry-run까지며,
non-dry benchmark는 fresh 32/64-asset handoff gate에서 수행해야 한다.

### 기존 전처리와의 output-equivalence 검증 설계

현재 2-asset A/B는 최종 pack을 풀어 28 NPZ의 52 arrays를 bitwise 비교하고 30 JSON을
byte 비교했으며, render alpha IoU 1.0과 동일한 8개 family membership을 확인했다. 이는
최종 학습 입력인 shape/PBR/SS latent와 camera/scale metadata가 같다는 강한 증거다.
하지만 production cleanup 뒤의 pack을 비교했기 때문에 중간 shape/PBR VXZ 24개는
`compared_vxz=0`으로 직접 비교되지 않았다. 또한 production renderer의 random lighting은
seed가 없어 unchanged reference끼리도 RGB PSNR이 낮으므로 단일 독립 run의 RGB PSNR은
동등성 gate로 사용할 수 없다.

다음 32/64-asset non-dry benchmark에서는 cleanup 전 workspace를 보존하고 같은 frozen
asset list와 raw-input SHA-256, 동일 GPU/model/dependency를 사용해 reference-repeat와
fast를 실행한다. 검증 계약은 다음과 같이 분리한다.

- artifact relative-path set, family membership, 성공/격리/실패 category가 asset별로 같다.
- camera transforms/radius/retry와 scale JSON은 exact 또는 absolute `1e-6` 이내다.
- VXZ와 NPZ의 coordinate/support는 canonical coordinate 순서로 exact하고, dtype/shape/key가
  같다. 같은 GPU에서는 attributes와 latent도 bitwise exact를 우선 요구한다.
- cross-GPU 비교가 필요하면 float tolerance를 임의로 정하지 않고 unchanged
  reference-repeat의 수치 noise envelope에서 family별 max/p99/mean error 한계를 정한다.
- render alpha는 view별 IoU `>=0.999`를 요구한다. RGB는 asset-derived fixed seed를 양쪽에
  동일하게 주는 별도 evaluation run에서 view별 PSNR `>=50dB`를 failure floor로 사용하고,
  기본 unseeded production run에서는 reference-repeat 분포보다 악화되지 않는지만 본다.
- 평균값이 나쁜 asset을 가리지 않도록 모든 metric의 worst asset/view와 p50/p95/p99를
  함께 저장하고 한 asset이라도 hard contract를 어기면 전체 gate를 실패시킨다.
- tar 자체의 byte hash는 timestamp/member order에 민감하므로 raw archive 외에는 extract한
  training-facing member를 canonical path와 content hash로 비교한다.

가장 신뢰할 수 있는 실행 순서는 unchanged reference 두 번으로 noise floor를 먼저 고정하고,
그 다음 exact same frozen scope에서 fast를 한 번 실행하는 것이다. `pipeline.cli audit`은
각 결과의 완전성/유효성을 확인하지만 reference와의 동등성을 비교하지 않으므로 별도의
`compare-outputs` report가 필요하다. report에는 per-asset/family exact count, max/p99/mean
numeric error, camera error, alpha IoU, seeded RGB PSNR, membership/failure diff와 source/image/
config/input manifest hash를 포함해야 한다.

이 결과는 production CLI 호환 image 전달 경로가 성립함을 확인한 것이지만 아직 coworker
handoff 승인은 아니다. exact final 2-asset run의 speedup은 1.576x로 standalone benchmark의
4.33x보다 작고, production acceptance에 필요한 fresh 64/100-asset throughput, tail
latency, 실패율, peak VRAM 측정이 남았다. 최신 run의 critical path 첫 구간은 두 render
worker가 각각 43.04/47.60초 걸린 Cycles boundary fitting이고, 그 뒤 geometry/encoding이
약 102초를 차지한다. low-resolution boundary fitting은 이전 실험에서 camera radius와
geometry를 바꿔 alpha IoU 0.693으로 실패했으므로 그대로 재도입하지 않는다. 다음 안전한
후보는 geometry leaf 12개가 같은 mesh/PBR를 반복 deserialize하는 부분을 한 asset
lifecycle로 합쳐 resolution/view 간 read-only 구조를 재사용하는 것이다.

그 전 단계로 geometry leaf 12개를 resolution/family 6개로 줄이고 각 leaf가 view 0/1을
함께 처리하도록 해 dump 역직렬화를 절반으로 줄이는 실험도 했다. 그러나 동일 2 assets의
headless 1-GPU 전체 실행이 160.35초에서 169.49초로 5.7% 느려졌다. asset 내부 두 view가
직렬화되며 잃은 병렬성이 deserialize 절감보다 커서 이 변경은 되돌렸다. 따라서 다음
재사용 최적화는 view 병렬성을 유지하면서 shared-memory/read-only cache를 쓰는 구조여야
한다.

최종 리뷰에서 운영 경계도 추가로 보강했다. `PIXAL3D_GPU_INDICES`가 config의 GPU
수보다 짧으면 encoder rank를 실제 allowlist 길이로 제한해 한 GPU에 여러 rank가 몰리지
않게 했다. `prepare_bundle`의 mesh/PBR dump pool과 render worker는 하나의 물리 CPU
예산을 나누며, parallel scheduler도 fused render/encode stage의 CPU 점유를 명시한다.
family manifest와 그 instance list는 전체 경로 no-follow, regular-file/inode 안정성,
동일 control directory, 정확한 family key, 정렬·중복 없는 64자리 SHA-256을 검증하고
누락 key는 전체 batch로 확장하지 않고 fail-closed한다. multi-rank shape 완료 barrier도
추가해 family별 partition이 달라도 SS encoder가 다른 rank의 미완성 shape latent를 먼저
읽지 않도록 했다. fused prepare에서 PBR 검증이 실패한 경우에는 자기 attempt key로
retry를 기록하도록 고쳐, 제거된 별도 render stage의 retry counter를 잘못 소비하거나
이어받지 않게 했다.

최종 evidence에 원문 timing이 남은 동일 hot path 성공 실행은 160.35초와 160.62초이며,
중앙값은 160.485초(80.243초/asset, reference 대비 1.577x)다. 최신 exact-final run은
시작 시 GPU 1이 idle이었고 render 43.04/47.60초를 포함해 전체 160.62초였다. 따라서
n17 shared node의 단일 wall time보다 fresh 64/100 asset 처리량을 production 판단에
사용해야 한다.

## 2026-09-04 — 1-GPU boundary-render 추가 최적화

`codex/production-preprocess-fast`에서 headless n17 GPU 1, 격리된 validation
data2/data3 및 `/dev/shm` scratch로 ABO 2 assets 전체 smoke pipeline을 다시 측정했다.
geometry native-thread 배분 최적화 뒤의 기준은 118.63초였고, render boundary retry가
alpha mask만 사용한다는 점을 이용해 같은 view의 retry마다 lighting scene을 다시 만들지
않도록 변경했다. camera/radius/retry 조건과 Cycles resolution/sample 설정은 유지했다.

| candidate | 2-assets wall | 기준 대비 |
| --- | ---: | ---: |
| native-thread retained baseline | 118.63초 | 1.000x |
| lighting reuse (`7d617a9`) | 116.67초 | **1.0168x** |

보호 suite는 249 passed (40.09초)였다. 동일 chair asset의 direct Blender A/B에서는 8개
view 모두 retry count와 camera radius가 exact했고 alpha IoU도 모두 1.0이었다. RGB는 기존
renderer가 unseeded random lighting을 사용하므로 독립 run의 pixel PSNR을 deterministic
equivalence gate로 해석하지 않았다.

세 가지 추가 후보도 실제 2-assets end-to-end 측정 또는 direct A/B에서 폐기했다.

- 128px boundary fitting 후 512px final fallback: 134.24초, 1-sample fitting: 129.38초.
  낮은 해상도여도 Cycles render의 process/kernel 고정비가 남아 전체 render 횟수를 줄이지
  못했다.
- scene bounding-box의 camera projection으로 radius를 추정한 뒤 기존 retry를 fallback으로
  쓰는 방식: direct A/B alpha IoU 평균 0.762로 framing 계약을 만족하지 못했다.
- 2 assets에서는 이미 asset-level render worker가 둘이므로 workers-per-GPU를 2보다
  높여도 일감이 늘지 않는다. peak GPU memory는 약 11.1GiB/95.6GiB였으나, 3/4 worker
  실험은 32/64 asset throughput gate에서만 의미가 있다.

현재 추가 후보는 full-resolution Cycles retry 자체를 줄이되 raster alpha framing을
보존하는 방법, 또는 cleanup 전 intermediate artifact까지 포함하는 automated
compare-outputs gate를 갖춘 32/64 asset 검증이다.

## 2026-09-04 — prepared-output automated equivalence guard

`data_toolkit.pipeline.output_comparison`를 추가했다. 이 CLI는 두 prepared root의
tar/manifest를 먼저 무결성 검증한 뒤 tar를 disk에 풀지 않고 member별로 비교한다. pack 및
member set, 일반 파일과 JSON은 exact content, NPZ는 key/dtype/shape/value array exact,
render PNG는 transforms content와 alpha channel exact를 요구한다. production lighting은
기존에도 unseeded이므로 RGB 값은 기본 hard gate에서 제외한다. 이는 RGB의 의미 있는 PSNR
평가는 asset-derived fixed-seed evaluation run으로 별도 실행해야 한다는 기존 계약을
명시적으로 보존한 것이다.

현재 `youngwoo_diyscene_fast:autoresearch` 이미지와 untouched reference를 각각 isolated
`ref_data2`, `fastgpu1_data2` root에서 비교한 결과는 8 packs, 74 members, alpha 16/16
exact, latent arrays 52/52 exact였다. 이 run의 wall time은 shared n17 상태 때문에
147.09초였으므로 speed metric에는 사용하지 않았다. controlled autoresearch retained
metric은 계속 116.67초다. 새 guard와 production pipeline regression suite는 GPU-1을
노출한 기존 conda image에서 252 passed (39.97초)였다.

추가 speed 후보도 둘을 측정해 모두 자동 revert했다. Cycles
`use_persistent_data`는 117.25초, single-GPU Blender renderer를 2개에서 1개로
직렬화한 경우는 147.42초였다. 따라서 현재 2-asset workload에서는 renderer 병렬성을
보존하는 것이 맞고, 32/64 asset에서는 별도 throughput gate로만 worker-count를 재평가한다.

## 2026-09-05 — fast camera-fit 후보의 output-validity 교정

full-resolution EEVEE/Workbench silhouette로 boundary fitting만 수행하고, 최종 이미지는
기존 Cycles로 다시 render하는 후보는 raw wall time 31.55초와 33.06초까지 내려갔다. 그러나
각 run의 prepared common tar에 regular-file member가 없었고, quality 결과도 두 asset 모두
`failure`/quarantine이었다. 따라서 이 수치는 speed 결과가 아니며 production에 채택하지
않는다.

이 검증에서 기존 benchmark가 pipeline exit code만 보고 빈 pack을 성공으로 오인할 수 있음을
확인했다. 이후 metric guard는 quality JSON에서 두 asset 모두 `completed`인지 확인하고,
common/SS/shape/PBR의 8개 tar 각각에 regular-file member가 하나 이상 있는지 확인한다.
이 strict guard를 통과한 clean 2-asset baseline은 114.65초였다. 현재 output-equivalence를
유지하는 retained production 후보는 여전히 `7d617a9`의 116.67초 결과이며, 3배 fast path는
Cycles alpha framing과 호환되는 renderer/pass를 확보하기 전까지 연구 전용이다.

같은 strict guard에서 추가 안전 후보도 모두 폐기했다. Blender `Render Result`의 alpha를
`foreach_get`으로 읽어 intermediate PNG를 없애는 방식은 215.80초, geometry native thread
상한 5/3은 각각 115.57/115.54초, geometry leaf 동시 실행 4→3은 122.77초였다. 따라서 이
2-asset workload에서 4 threads·4 leaf topology가 유지되어야 하며, 100초 target을 만족하는
유효 후보는 아직 없다.

## 2026-09-05 — lossless PNG retry encoding 최적화

Cycles boundary retry는 최종적으로 쓰이지 않는 PNG도 반복해서 lossless compression한다.
Blender PNG compression level을 기본값 대신 0과 1로 실험했고, decoded RGBA가 변하지
않는다는 성질을 이용했다. strict guard는 두 asset quality 완료, 8 family pack의 regular
file 존재, reference와 transforms/retry/camera matrix exact 및 16개 PNG alpha exact를 모두
요구한다.

| PNG compression | 2-assets wall | 결과 |
| --- | ---: | --- |
| baseline | 116.80초 | - |
| 0 | 110.53초 | retained |
| 1 | **107.09초** | retained |

`2b90dbe`는 compression 1을 retained했다. RGB는 production renderer의 unseeded lighting
때문에 byte gate에서 제외했지만 PNG compression은 lossless이며 camera/alpha는 exact였다.
100초 target까지는 7.09초가 남아 있으므로 다음 실험은 retry PNG 자체의 lossless writer
비용을 더 줄이면서 alpha controller를 exact하게 유지하는 경로에 집중해야 한다.

후속 writer 후보도 strict guard에서 폐기했다. compression 2는 114.70초, TGA intermediate
후 final PNG 저장은 117.28초, process-local `/tmp`에 retry PNG를 쓰고 final만 output mount로
copy하는 방식은 120.73초였다. 따라서 이 container에서는 mounted output directory가 overlay
temporary storage보다 빠르고, PNG compression 1이 검증된 최적점이다.

Render Layers Alpha를 compositor grayscale PNG로 저장해 retry RGB encoding을 생략하는
후속 후보는 163.50초였고 두 asset이 `schema_failure`로 quarantine됐다. compositor의
grayscale state가 final RGBA image contract까지 오염시킨 것으로 확인되어 자동 revert했고
production에는 사용하지 않는다.

## 2026-09-06 — alpha-only boundary-fitting 후속 실험

`codex-autoresearch-harness`로 isolated n17 GPU 1의 2-assets full pipeline을 다시
측정했다. 기준은 `117.14초`, 목표는 `100초 이하`였다. 기준 검증은 두 asset의 quality
`completed`, 8 family pack의 regular member, `transforms.json` exact, 16개 render PNG의
alpha exact를 확인한다. 각 후보는 metric이 기준보다 좋아질 때만 같은 strict guard로
승격하도록 구성했고, 네 후보 모두 느려 자동 revert됐다.

| 후보 | 2-assets wall | 기준 대비 | 결과 |
| --- | ---: | ---: | --- |
| compositor File Output alpha sidecar | 120.32초 | +3.18초 | 폐기 |
| 1-sample fitting 후 32-sample final render | 132.07초 | +14.93초 | 폐기 |
| PNG alpha channel만 decode | 119.87초 | +2.73초 | 폐기 |
| `/dev/shm` retry PNG, final만 output으로 copy | 117.78초 | +0.64초 | 폐기 |

특히 `/dev/shm` 후보도 이 controlled container에서는 output mount 직접 write보다 느렸다.
따라서 PNG compression 1 retained 경로 외에 이 네 변경은 production에 포함되지 않는다.
4회 iteration 한도에 도달하여 run `3224cb2d920a457581ac4e3c5b6a495e`는 target 미달
상태로 안전하게 stopped 되었다.

## 2026-09-11 — n1 isolated production-path validation

`n1`에서 기존 `youngwoo_diyscene`과 n17 작업을 변경하지 않고,
`pixal3d_n1_validation` 컨테이너를 별도로 만들었다. 코드는 fork `master`
`903fe647bbc40ac8ef1bc84d4bf61c95c8f4df55`를 read-only mount했고, 물리 GPU 1–7만
컨테이너에 노출했다. input raw와 canonical metadata도 read-only였으며 output,
data3, scratch는 각각 `pixal3d-n1-validation-final2` 전용 경로로 분리했다.

production hardware preflight와 derived hardware report는 통과했다. 그러나 smoke
1-asset full run은 download가 완료된 뒤 내부 `stage_raw`에서 `[Errno 22]
Invalid argument`로 중단되었고, 그 뒤 frozen batch file-set guard가 재실행을 거부했다.
원인은 NFSv4.2 scratch mount가 `renameat2(RENAME_NOREPLACE)`를 `EINVAL`로 거부하는
것이었다. 동일 syscall과 `_atomic_stage_stream()` call path를 별도 probe로 재현했다.
추가 재검증에서 coworker의 **n17 원본** 설정은 `local_root`가
`/root/node17/data/pixal3d`이며 실제 `/dev/nvme0n1p2` ext4임을 확인했다. 같은 clean
`_atomic_stage_stream()`은 그 실제 local volume에서 성공했다. 반면 n1 격리 검증의
`local_root=/root/local`은 `10.20.22.41:/volume1/rvi/youngwoo/...` NFSv4.2이고, 같은
clean 구현이 `EINVAL`로 실패했다. 즉 raw 입력이 NFS인 것 자체가 아니라 `stage_raw`의
atomic publish 대상인 scratch/local root가 NFS였던 것이 직접 조건이다.

기존 **n1의 동명** `youngwoo_diyscene`은 별도 설정에서 `local_root`를
`/root/data2/pixal3d-objaversexl-sketchfab-scratch` NFS에 두지만, 작업 트리에
same-directory `link()` 후 `unlink()`로 publish하는 NFS fallback이 미커밋 상태로
존재한다. 같은 NFS 디렉터리 A/B에서 fork master `903fe647`의 clean 구현은 `EINVAL`,
기존 n1 수정 코드는 publish 성공을 보였고 기존 destination payload도 덮어쓰지 않았다.
따라서 실패는 n17 원본과 달라진 n1 scratch 배치와 clean fork에 fallback이 없는 상태의
조합이며, download/GPU/raw data/renderer 문제는 아니다. pack/latent/quality 산출물은
0개이므로 유효한 end-to-end asset time이나 output equivalence 결과는 아직 보고하지
않는다. GPU 0 공유로 인한 OOM은 별도로 기존 n1 컨테이너의 GPU 0–6 기본 선택 문제였고,
새 컨테이너는 Docker device isolation으로 이를 배제했다.

## 2026-09-11 — NFS atomic staging compatibility fix

NFSv4.2에서 `renameat2(RENAME_NOREPLACE)`가 `EINVAL` 또는 `ENOSYS`를 반환할
때, 같은 directory FD 안에서 `os.link(..., follow_symlinks=False)`로 destination을
원자적으로 생성한 뒤 source 이름을 제거하는 최소 fallback을 `_rename_noreplace()`에
추가했다. destination이 이미 존재하면 `FileExistsError`가 그대로 발생하므로 기존
no-replace 계약과 payload 보존은 유지된다. 다른 errno도 기존처럼 그대로 전파한다.

회귀 테스트는 clean master에서 먼저 `2 failed`로 red를 확인한 뒤 fallback 적용 후
`3 passed`가 되었고, `tests/data_toolkit/test_orchestrator.py` 전체는 `202 passed`였다.
n1 격리 validation container의 실제 NFS scratch에서 첫 publish와 동일-payload 재실행이
성공하고, 다른 payload 교체는 거부되며 기존 payload가 유지됨을 확인했다. 실제
ObjaverseXL 1-asset `stage_raw`도 1,718,688-byte GLB와 metadata 2개 파일을 publish하고
GLB SHA-256 `000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10`을
검증했다. 생성한 검증 파일은 제거했다.

전체 `tests/data_toolkit`은 `826 passed, 7 failed`였으며, 실패는 local torch 환경의
missing `objaverse`, noexec temp fixture, pre-existing path permission/config 문제였다.
수정 범위의 실패는 없었다. n1의 기존 production job이 GPU 1–7을 사용 중이어서 전체
GPU 1-asset run은 충돌을 피하기 위해 실행하지 않았다.

## 2026-09-11 — n17→n1 production 전환 상태 점검

n17의 `youngwoo_diyscene` container는 실행 상태지만 내부에는 Pixal3D preprocessing
`supervisor`/`worker`/leaf process가 없었다. n17 GPU 1/2/3/7에서 보인 Python process는
모두 다른 container(`9c664d34...`)의 학습/evaluation 작업이었다. 따라서 중지할 활성
Pixal3D preprocessing process는 이미 없다.

공유 production queue manifest에는 756 units, 188,934 assets가 있다. 완료 marker는
151 units/37,675 assets이고, 미완료 범위는 605 units/151,259 assets다. failure marker는
없다. lease directory 4개는 2026-08-06 이후 갱신되지 않은 node11/node16 lease 또는 빈
directory로, 현재 실행 process와 대응하지 않는 stale 상태다. worker registry heartbeat도
같은 시점에 멈춰 있다.

중요하게 queue config hash는 `50d88c149217781d6daff8bc48edf37fe131c772dda6c4d8bd7f8ced021bcae4`,
현재 fork master의 canonical config hash는
`ff7dc7073940b869b001b8c8e60525390fe15091f518887814ddc1f3ca35e988`다. 현재 CLI로
`queue status`를 실행하면 이 차이를 fail-closed로 거부한다. 원인을 확인하지 않고 queue
manifest hash를 바꾸거나 새 worker를 production queue에 붙이지 않는다.

n1jh에서 host GPU 상태를 조회한 시점에는 RTX 3090 GPU 4–7이 모두 utilization 0%,
memory 3 MiB로 idle이었다. production container에는 `--gpus all`로 host GPU 0–7을
모두 노출하되, worker registry의 GPU allowlist를 `4,5,6,7`로 제한한다. worker runtime은
이 allowlist를 `PIXAL3D_GPU_INDICES`로 전달하고 각 Blender/encoder subprocess에
`CUDA_VISIBLE_DEVICES=4`, `5`, `6`, `7`을 배정한다. supervisor process 전체에 하나의
`CUDA_VISIBLE_DEVICES=4,5,6,7`을 설정해 ordinal을 재매핑하지 않는다. encoder는 GPU당
한 rank를 유지하고, Cycles condition render만 canonical adaptive
`render_workers_per_gpu_steps: [2, 3, 4]`를 사용한다. 초기값은 GPU당 3 process이며,
메모리/온도 상태가 안정적이면 4로 올리고 실패 또는 압력이 있으면 2로 낮춘다. 4 GPU에서
최대 render process는 16개로 코드의 28-process cap 이내다.

아직 production 전환은 실행하지 않았다. n1 host SSH(`10.20.22.128:55555`)가 key exchange
전에 반복적으로 reset되어 isolated container 재구성과 NFS 수정 후 full-asset QA를 할 수
없었다. 접속 복구 뒤에는 GPU 4–7 전용 container에서 isolated 1-asset completion과
32/64-asset의 process-per-GPU 2/3/4 throughput·peak VRAM 비교를 먼저 통과시키고, queue
config mismatch를 원인에 맞게 해소한 뒤에만 shared production queue를 재개한다.

후속 점검에서는 n17을 jump host로 사용해 n1 host에 접속했다. validation container를
`--gpus all`로 재생성했고 내부 `nvidia-smi`에서 GPU 0–7 전체가 보이는 것을 확인했다.
validation config는 GPU 4개와 adaptive render process `[2, 3, 4]`로 바꾸고
`PIXAL3D_GPU_INDICES=4,5,6,7`을 적용했다. helper probe에서 선택 index가 정확히
`(4, 5, 6, 7)`이었다.

이 구성에서 hardware preflight가 GPU 0–6 정확히 7개만 허용하는 기존 hardcode 때문에
실패했다. visible GPU inventory와 `PIXAL3D_GPU_INDICES` allowlist를 분리하고 report의
`gpu_count`/`required_gpus`를 config 기반으로 만드는 회귀 테스트를 먼저 추가했다. 기존
7-GPU case는 pass, 8개 visible + 4–7 selected case는 기존 코드에서 fail하는 red를 확인한
뒤 수정했으며 관련 local suite는 56 passed, container targeted suite는 2 passed였다.
실제 n1 preflight는 GPU 4–7 각각에 단일 `CUDA_VISIBLE_DEVICES`를 주어 OptiX cube render를
완료했고 71.01초에 통과했다. validated hardware report도 생성됐다.

1-asset full smoke는 아직 완료되지 않았다. fresh validation root에 registry가 없어 첫 시도가
4.57초에 선행 차단됐고, registry 168,307 assets를 55.55초에 생성했다. 다음에는 hardware
report 부재로 5.61초에 차단됐다. 이후 원본 metadata를 read-only mount한 구성이
`asset_stats/new_records` write를 막아 73.22초에 실패했다. 원본 metadata는 변경하지 않고
160MB 복사본을 새 validation root에 만들어 이 문제를 해소했다. 마지막 fresh-root 시도는
hardware report를 복사했지만 Blender tool cache가 새 scratch에 없어서 75.79초에 실패했다.
production 입력·출력과 원래 두 `youngwoo_diyscene` container는 변경하지 않았다.

현재 남은 validation setup은 검증된 Blender 4.5.1 tool cache를 fresh scratch에 배치하고,
registry/report/writable metadata를 함께 가진 하나의 clean root에서 smoke를 처음부터 한 번
실행하는 것이다. smoke가 pack/audit까지 성공한 뒤에만 32/64 assets로 GPU당 render process
2/3/4를 비교한다. shared production queue는 별도로 config hash mismatch를 해소하기 전까지
연결하지 않는다.

## 2026-09-11 — n1 GPU 4–7 full smoke 및 production 재개

앞 절의 미완료 validation을 fresh v3 root에서 마쳤다. `pixal3d_n1_validation`에는 host GPU
0–7 전체를 노출하고, 실행 환경에 `PIXAL3D_GPU_INDICES=4,5,6,7`만 전달했다. registry,
hardware report, writable metadata 복사본과 Blender tool cache만 새 root로 옮겼으며 이전 실패의
checkpoint/work/output은 재사용하지 않았다.

첫 full smoke는 geometry/latent 단계에서 `no kernel image is available`로 실패했다. n17은
RTX PRO 6000 Blackwell(sm_120), n1은 RTX 3090(sm_86)인데 전달된 image의 `flex_gemm`
extension에는 sm_120 cubin 6개만 들어 있었다. 동일 upstream commit
`6dd94a859c26ee8246888502eada3dd8ad85532e`을 validation container 안에서
`TORCH_CUDA_ARCH_LIST=8.6`으로 다시 빌드했고, `cuobjdump`로 sm_86 cubin 6개를 확인했다.
그 뒤 clean 재실행은 실제 렌더·geometry·latent를 모두 수행한 뒤 395.33초에 Git
safe-directory provenance 검사에서만 멈췄다. `/root/dev/Pixal3D` 한 경로만 Git safe-directory로
전달해 재개한 실행은 17.63초에 종료됐다.

최종 checkpoint에는 download, stage_raw, prepare_bundle, geometry_encode_bundle,
256/512/1024 cleanup, validate_outputs, build_packs, archive_raw, cleanup_local이 모두 완료로
기록됐고 asset SHA
`000045aad61c956b45fc468b2b2ec954636e5f647f1c1995854d46ecaa525e10`의 quality는
`completed`였다. common 1개, shape 3개, PBR 3개, SS 1개의 총 8개 prepared tar와 raw
archive가 생성됐으며 별도 `pipeline.cli audit`도 exit 0으로 통과했다. scratch output은 cleanup
후 0 files였다. clean 계산 run의 GPU telemetry 787 samples/GPU에서 선택 GPU의 최대 VRAM은
GPU 4/5/6/7 각각 7069/7256/3342/6338 MiB였다. asset 하나에 target view가 2개뿐이라
지속 부하는 주로 GPU 5와 7에 걸렸고 전체 wall의 대부분은 render 23.83초 이후의 latent
encode였다.

renderer process 비교용으로 같은 frozen 32-asset scope를 생성했다. instances SHA-256은
`12d192a65dc275d03b76df13ef35d8262f6b5a23794704ecb500dc5a84e8d2c3`이며, raw 준비는
90.12초, 비교에서 공통으로 재사용할 mesh/PBR dump는 70.55초였다. 다만 비교 직전 n1의
기존 `youngwoo_diyscene` production이 이미 2026-09-11 19:37 KST부터 GPU 1–7로 실행 중인
것을 발견했다. production과 GPU 4–7을 경쟁시키지 않기 위해 isolated 2/3/4 A/B/C는 중단했다.

기존 production process group은 batch 83의 resumable checkpoint를 보존한 채 SIGTERM으로
2초 안에 정상 종료했다. container의 Docker device request는 계속 `--gpus all`이다. 기존
local-container `runtime.py`가 hardware evidence와 요청 GPU 수를 모두 정확히 7개로
하드코딩해 4-GPU 재개를 거부했으므로, config hash와 7-GPU hardware evidence는 유지하되
evidence의 GPU 수를 `config.parallelism.gpu_count`로 검증하고 물리 index가 unique/nonnegative인지
검증하도록 최소 runtime patch를 적용했다. patch 원본은
`autoresearch-results/n1-production-runtime-gpu-subset.patch`에 남겼다. 4–7 환경에서 기존
hardware report CLI는 exit 0으로 통과했다. production container의 hardware test는 7 passed,
1 failed였으며 실패는 기존 local-container 코드가 오류 문자열을 `OptiX`에서 `OPTIX`로 바꾼
반면 테스트 기대 문자열은 갱신하지 않은 unrelated mismatch다.

production은 같은 config, checkpoint, input/output에서 batch 83부터
`PIXAL3D_GPU_INDICES=4,5,6,7`로 재개했다. parent의 `CUDA_VISIBLE_DEVICES`도 script가 같은
값으로 설정하지만 container 자체에는 GPU 0–7이 모두 보인다. batch 83 잔여 검증을 마친 뒤
batch 84 `prepare_bundle`이 `--render_workers_per_gpu 3`, `world_size 12`로 시작되어 물리 GPU
4개당 renderer 3개가 실제 적용됐다. 실행 중 20초/37 samples per GPU telemetry는 GPU 0–3의
utilization과 VRAM이 모두 0이었고, GPU 4–7만 최대 21–32% utilization 및
3562–3588 MiB를 사용했다. n17의 `youngwoo_diyscene`에는 여전히 활성 Pixal3D process가
없다.

추가 geometry 관찰에서 기존 local-container patch가 CPU 중심 geometry producer의 CUDA
import를 물리 GPU 0에 강제해 4 MiB context를 만드는 것을 발견했다. production process
group을 같은 resumable 방식으로 다시 종료하고, 해당 device를 `gpu_indices[0]`으로 선택하도록
`autoresearch-results/n1-production-geometry-gpu-subset.patch`를 적용한 뒤 batch 84부터
재개했다. 현재 geometry command의 encoder `world_size`는 4이며 GPU 4–7에 각각 약 961 MiB가
할당됐다. GPU 0–3은 utilization 0%, memory 0 MiB이고 최근 production log에 새 ERROR/Traceback은
없다. n1 production process는 현재 계속 실행 중이다.

## 2026-09-11 — n17 canonical queue를 n1에서 이어서 재개

n17 production의 canonical root와 frozen queue를 다시 조사했다. queue는 756 units,
188,934 assets이고 기존 완료 marker는 151 units/37,675 assets였다. 완료되지 않은
605 units/151,259 assets 가운데 prepared index와 raw archive가 모두 있는 후보는 2개뿐이었고,
둘 다 full published/archive validator를 통과하지 못해 추가 완료 marker는 만들지 않았다.
따라서 기존 151개 marker가 실제 완료 범위이며 queue claim은 이 marker들을 먼저 확인해
건너뛴다.

queue에 저장된 config hash `50d88c149217781d6daff8bc48edf37fe131c772dda6c4d8bd7f8ced021bcae4`는
현재 canonical hash
`ff7dc7073940b869b001b8c8e60525390fe15091f518887814ddc1f3ca35e988`와 달랐다.
두 실행 계획을 registry와 frozen shard manifest에서 재구성하자 unit 756개와 asset 순서가
완전히 동일했다. 옛 hash는 node11 host path
`/file2/youngwoo/pixal3d`, `/file3/youngwoo/pixal3d`,
`/file2/youngwoo/pixal3d_local`을 사용했을 때 정확히 재현됐으므로 차이는 처리 범위가 아니라
동일 storage의 host/container path identity뿐이었다. 원본 `units.json`을
`control/recovery/queue-config-path-identity-20260911T2154KST/`에 보존하고 queue hash만
현재 canonical hash로 원자적으로 이관했다. 이후 정식 `queue status`가 같은 config로
통과했다.

node1 worker는 canonical shared root `/root/data2/pixal3d`,
`/root/data3/pixal3d`와 local ext4 scratch `/root/node1/data/pixal3d`를 사용하도록
등록했다. GPU allowlist는 4,5,6,7이고 container에는 GPU 0–7 전체가 노출된다. runtime
검사와 hardware report가 CUDA 12.4/Torch 2.6 및 CUDA 12.8/Torch 2.8 두 지원 조합을 같은
규칙으로 허용하도록 맞췄고, hardware inventory/report의 GPU 수를 hard-coded 7이 아니라
registered execution config에서 검증하도록 수정했다. 회귀 테스트는 runtime 8 passed,
resources 36 passed, hardware 11 passed다. 관련 patch는
`autoresearch-results/n1-canonical-runtime-profile-*.patch`,
`n1-canonical-hardware-profile-*.patch`, `n1-hardware-gpu-subset-*.patch`,
`n1-accounting-vanished-file-*.patch`에 보존했다.

기존 n1 `youngwoo_diyscene`은 `NVIDIA_DRIVER_CAPABILITIES=compute,utility`라 OptiX
library가 주입되지 않았다. 기존 container와 내부 VS Code/Codex 세션은 중지하지 않고,
`docker commit --pause=false`로 conda 환경을 보존한
`youngwoo_diyscene_fast:n1-canonical-20260911` 이미지를 만든 뒤 동일 bind mount와
`NVIDIA_DRIVER_CAPABILITIES=all`인 별도 `youngwoo_diyscene_fast` container를 생성했다.
GPU 4에서 실제 OptiX cube PNG/metadata 생성과 CPU fallback 없음이 확인됐다.

n1 GPU 4–7 held config로 hardware evidence/report를 새로 생성하고 smoke/pilot evidence를
fresh timestamp로 재발행했다. 첫 시험 launch가 stale gate 때문에 남긴 batch001–004의
false terminal failure와 batch005 release 2개는 각각
`work_queue/remediated/n1-stale-gate-20260911` 및
`control/recovery/n1-stale-gate-20260911-batch005`에 보존하고 active queue에서 제거했다.
실제 leaf command는 실행되지 않았던 이력이므로 원래 attempt numbering으로 복구됐고,
재개 직전 failed 0, lease 0이었다.

canonical supervisor는 `/opt/conda/envs/pixal3d/bin/python`으로
`youngwoo_diyscene_fast`에서 시작했다. parent `CUDA_VISIBLE_DEVICES`는 설정하지 않고
`PIXAL3D_GPU_INDICES=4,5,6,7`만 전달했다. queue는 완료 151 units를 건너뛰고 첫 미완료
`ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00001/batch001`을 claim했다. 첫 leaf 이전에는
persisted project accounting을 data2 약 1.57TB/data3 약 2.69TB의 전체 canonical tree와
재조정했고, 과거 runtime/experiment 디렉터리까지 재귀 stat하면서 약 32분이 걸렸다. 이는
실제 preprocessing 외의 별도 startup 병목이다. 검증된 accounting snapshot을 첫 실행에서만
재사용하도록 opt-in launch flag를 추가했고, 이후 supervisor restart에는 해당 flag가 전달되지
않아 기본 fail-closed 동작을 유지한다.

n17에서 생성한 미완료 chunk checkpoint를 n1에서 이어받을 때 두 가지 cross-node resume
문제가 추가로 드러났다. 첫째, 기존 `prepare, render, geometry_256, ...` stage 이름을 현재
`prepare, render, encode, finalize`로 매핑하는 것만으로는 n17 local scratch에만 있던 중간
산출물까지 존재한다고 잘못 가정했다. non-promoted checkpoint의 완료 stage를 dependency
순서대로 모두 검증하고, 처음 유효하지 않은 dependency부터 다시 실행하도록 변경했다. 둘째,
이미 promoted된 chunk는 canonical final output은 남아 있지만 node-local raw metadata/file이
정리되어 publication 재개가 실패했다. 이 경우 canonical raw cache metadata를 읽어 checksum을
검증하며 필요한 completed raw만 n1 local scratch에 다시 stage하도록 수정했다. 두 회귀 테스트는
수정 전 각각 실패했고 수정 후 통과했으며, scheduler/runtime/resources/hardware와 관련 raw
metadata 테스트를 합친 최종 targeted suite는 `72 passed`였다.

호환성 확인 과정에서 node1이 잘못 terminal 처리한 batch002–005 checkpoint, quality outcome,
queue attempt는 원본을 삭제하지 않고
`control/recovery/n1-cross-node-resume-20260911T2330KST`와
`work_queue/remediated/n1-cross-node-resume-20260911T2330KST`에 보존한 뒤 재실행 가능한 상태로
되돌렸다. rolling quality ledger는 손상 전 logical batch outcome에서 최근 500개 entry를 다시
구성했다. 완료 marker 151개는 변경하지 않았고 failed marker는 0개다.

최종 재개는 canonical config hash
`ff7dc7073940b869b001b8c8e60525390fe15091f518887814ddc1f3ca35e988`와 canonical shared
output root를 그대로 사용한다. 현재 queue lease는
`ObjaverseXL_sketchfab/ObjaverseXL_sketchfab-00001/batch005`이고 heartbeat가 30초마다
갱신된다. `chunk000`의 16개 render rank가 실제 Cycles/OptiX Blender process를 생성했으며,
rank 0–15의 `CUDA_VISIBLE_DEVICES`는 정확히 `4,5,6,7`을 네 번 반복한다. 연속 GPU snapshot에서
GPU 4–7만 Pixal3D 부하를 보였고 local output 파일 생성도 확인했다. batch001–004의 잘못
소진된 attempt history도 recovery 위치로 이동했으므로 현재 unit 이후 queue에서 정상적으로
다시 claim할 수 있다. production 최종 출력은 기존과 동일한 `/root/data2/pixal3d`와
`/root/data3/pixal3d`에 기록되고 `/root/node1/data/pixal3d`는 node1 local scratch와 log에만
사용된다.

## 2026-09-12 — geometry producer wave 기반 encoder timeout 수정

n1 production의 `ObjaverseXL_sketchfab-00001/batch005`에서 geometry/latent stage가 반복
실패한 원인을 추적했다. CUDA OOM이나 memory leak이 아니라, geometry producer와 encoder를
동시에 시작하면서 단일 asset 제한인 900초를 64-asset chunk 전체의 geometry 입력 대기 및
loader inactivity 제한으로도 그대로 사용한 것이 원인이었다. 실제 `chunk000` 로그에서
`geometry input timed out` 1회와 `loader inactivity timed out after 900 seconds` 2회를
확인했고, 첫 timeout 대상 `.vxz`는 producer가 계속 실행됐다면 뒤늦게 정상 생성되는 파일이었다.

`geometry_encode_bundle`이 family manifest, view 수, geometry job 수, worker 배분을 이용해
producer wave 수를 계산하고 encoder 대기 제한에만 `single_asset_timeout × waves`를 적용하도록
수정했다. 개별 geometry asset의 900초 제한은 유지하므로 실제 leaf hang은 기존처럼 조기에
실패한다. production `chunk001`은 shape 63개/PBR 55개, 2 views, 12 jobs, job당 3 workers라
21 waves와 18,900초 encoder 대기시간이 계산됐다.

failing-first 단위 테스트는 수정 전 1초를 그대로 전달해 실패했고, 수정 후 일반 manifest 없는
4-job/6-wave 경우와 production 형태의 12-job/21-wave 경우를 모두 통과했다. command graph 관련
최종 targeted suite는 `49 passed in 0.50s`였다. 전체 data-toolkit suite는 `827 passed`였고,
이번 변경 밖의 실행환경/기존 dirty 변경 때문에 Blender fixture, 선택 dependency, preflight 관련
9개가 실패했다.

실제 n1 `youngwoo_diyscene_fast`의 동일 canonical queue/output에서 원래 실패하던
`chunk001`을 재개했다. 네 encoder rank 모두 `--timeout_seconds 18900` 및
`PIXAL3D_WAIT_FOR_GEOMETRY_SECONDS=18900`을 사용했고, 기존 900초 지점을 넘긴 약 19분 39초 후
새 timeout 없이 stage가 완료됐다. 최종 두-view 산출물은 shape 256/512/1024 각각 126개,
PBR 256/512/1024 각각 110개, SS-64 126개였으며 기존 partial output은 skip/reuse됐다.
worker는 곧바로 같은 batch의 `chunk002/prepare_bundle`로 진행했다. canonical output 경로와
산출물 형식은 변경하지 않았다.

## 2026-09-12 — n1 production GPU 저사용률 원인 확인

실행 중인 `youngwoo_diyscene_fast`의
`ObjaverseXL_sketchfab-00001/batch005/chunk002` geometry/latent stage를 중단하거나 수정하지
않고 GPU process, CPU/메모리/IO, telemetry, checkpoint와 scheduling 코드를 함께 조사했다.
20초 process-level `nvidia-smi pmon`에서 Pixal3D rank가 배치된 GPU 4, 5, 6은 전체 구간 동안
SM work가 없었고, GPU 7의 Pixal3D rank만 13%, 36%의 짧은 burst 두 번을 보였다. GPU 6/7의
나머지 device utilization은 별도 사용자 process에서 발생했다. 따라서 단일 시점 측정의 착시가
아니라 Pixal3D encoder가 실제로 충분히 공급받지 못하고 있었다.

직접 원인은 geometry CPU producer concurrency의 과도한 축소였다. 직전 `prepare_bundle`의
2026-09-11 17:34:16 UTC telemetry에서 CPU가 순간적으로 80.4%에 도달했고,
`choose_worker_profile`은 최근 60개 sample 중 하나라도 80% 이상이면 full geometry profile
`(44 workers, 1 thread)`을 reduced profile `(11 workers, 4 threads)`로 낮춘다. 실제 직전
prepare/render checkpoint는 44 workers였지만 현재 `geometry_encode_bundle` 명령은
`--max_workers 11`이었다. 더구나 reduced profile의 안정 복구는 설정된
`[(8,4), (10,4), (11,4)]` 안에서만 증가하므로 special full profile `(44,1)`로 돌아가는
전이가 없다.

geometry bundle은 shape/PBR × 256/512/1024 × 2 views의 12개 장기 job으로 나눈 뒤
`max_workers // job_count`를 각 job에 배정한다. 따라서 현재 11-worker profile은 각 job당
1 worker가 되어 geometry가 family/view마다 한 asset씩만 생성되고 네 GPU encoder를
지속적으로 공급하지 못한다. 동시에 encoder는 asset 목록을 rank별 contiguous slice로 고정
분할하므로 asset 복잡도 차이가 그대로 straggler가 된다. 실제 rank 완료 시점이 수분씩 벌어져
먼저 끝난 rank는 5–20 GiB의 model VRAM을 유지한 채 대기했다.

현재 시점의 약 310 GiB 가용 RAM, 0% I/O wait, 0 swap-in, 비어 있는 resource reason,
약 70% CPU idle을 확인했으므로 NFS, swap 또는 resource guard pause는 현재 저사용률의 원인이
아니다. 후속 수정 우선순위는 (1) reduced profile에서 full `(44,1)`로 복귀 경로 추가,
(2) 12개 job별 정적 worker 분할을 global dynamic geometry queue로 변경,
(3) encoder의 contiguous rank slice를 dynamic ready-task queue로 변경하는 순서다.

## 2026-09-12 — geometry 공급 및 encoder rank 균형 개선

GPU 저사용률 조사 결과에 따라 세 가지 scheduling 수정과 회귀 검증을 수행했다. resource
snapshot의 CPU pressure 판정은 설정의 hard limit인 90%를 사용하게 하여 순간 80.4%가 full
geometry `(44 workers, 1 thread)`를 불필요하게 낮추지 않도록 했다. 이미 reduced profile
`(11,4)`에 들어간 경우에도 안정된 snapshot 3개 뒤에는 full profile로 복귀하는 전이를
추가했다. geometry bundle의 동시 shape/PBR-view job에는 worker 수를 나머지까지 배분하고,
서로 다른 physical-core affinity offset을 주어 각 leaf가 같은 첫 CPU core들에 겹쳐 pin되던
문제를 제거했다. view/job 수가 동시 실행 slot 수보다 많을 때는 wave가 전부 끝난 뒤에만 같은
affinity slot을 재사용하도록 보강했으며, 해당 회귀를 포함한 집중 suite는 78 tests를 통과했다.

Shape/PBR/SS encoder는 정렬된 asset 목록의 연속 구간 대신 rank-strided asset을 받는다. 출력
경로와 resume 판정은 asset/view 파일 기반으로 그대로 유지하면서 metadata 순서에 비슷한
복잡도의 asset이 뭉쳤을 때 마지막 rank 하나만 오래 남는 tail을 줄이는 변경이다. 순수 partition
helper, profile 복구, affinity 전달/분리 회귀 테스트를 추가했다. 변경 집중 suite는 280 tests,
전체 `tests/data_toolkit`은 840 tests가 통과했다. 전체 suite의 7 failures는 noexec Blender
fixture, local environment의 missing `objaverse`, preflight path permission 및 기존 test/config
mismatch로 수정 범위 밖이다. 신규 partition module의 basedpyright는 0 errors/0 warnings,
선택 파일 ruff와 `git diff --check`도 통과했다.

n1의 production 입력을 읽기 전용으로 재사용하고
`/root/node1/data/pixal3d-ab-7z5wvucM` 아래 별도 source/output에서 8 assets, 2 views,
2 ranks 격리 A/B를 실행했다. Shape-256은 baseline 12초/optimized 13초, PBR-256은 12초/12초,
SS-64는 7초/8초였다. 이 작은 균등 workload의 목적은 throughput 측정이 아니라 rank 재배치의
출력 검증이다. 두 결과는 product 92 files, metadata 46 rows, 모든 sparse coordinates와 scale이
일치했다. feature PSNR(peak=1)은 Shape 55.93 dB, PBR 55.17 dB, SS는 byte-identical로 무한대여서
합의한 50 dB 기준을 통과했다.

production `youngwoo_diyscene_fast`의 대상 source는
`/root/node1/data/pixal3d/control/runtime/source-backup-20260912-gpu-balance`에 백업한 뒤 검증된
7개 파일만 교체했다. local/remote SHA-256과 container 내 py_compile이 모두 일치했다. 오래
걸리던 batch005/chunk002 geometry stage는 교체 직전에 자연 완료되어 중단하지 않았고,
worker는 기존 canonical queue/checkpoint/output을 유지한 채 chunk003 prepare를 시작했다.
container에는 계속 모든 GPU가 보이며 production allowlist는 물리 GPU 4–7이다. 새 geometry
profile/affinity/rank scheduling은 chunk003의 geometry/latent stage부터 적용된다.

## 2026-09-12 — GPU scheduling source reload 중 발견한 rolling quality gate

새 scheduling 코드를 worker process에도 반영하기 위해 node1을 draining으로 바꾸고 기존
batch005 lease를 정식 handoff한 뒤 worker를 재시작했다. 재시작된 worker는 leaf command를
실행하기 전에 세 번 연속 `end-to-end failures exceed 10% (75/500)`으로 중단됐다. canonical
manifest는 최근 handoff unit보다 가장 앞선 미완료 unit을 우선하므로 batch005가 아니라
`ObjaverseXL_sketchfab-00001/batch001`을 선택했다. 이 세 번은 실제 preprocessing 실패가
아니며 startup gate가 만든 queue release/terminal 기록이다.

live quality ledger를 직접 집계한 결과 500개 중 `completed=425`, `failure=75`였고 75건은 모두
batch001 asset validation 결과였다. chunk별 분포는 33/20/22/0이며 실패 위치는 batch001의
position 1–190에만 존재했다. 75개 모두 canonical raw metadata와 non-empty GLB가 남아 있었다.
실제 ledger를 `RollingQualityGate`에 복원하면 동일한 75/500 오류가 재현됐고, 메모리상의 복제본에서
실패 25개만 completed로 바꿔 50/500으로 만들면 violation이 사라졌다. 따라서 이번 production
중단 원인은 GPU scheduling이나 batch005 checkpoint 손상이 아니라 persisted quality window다.

과거 failure 중 첫 asset
`080e093167141cc8e5a4bcc3f72bb1d7dcaf4e5eab55f21aee884c245c1285e6`을 canonical과 분리된
`/root/data2/pixal3d-quality-gate-repro-20260912`,
`/root/data3/pixal3d-quality-gate-repro-20260912`,
`/root/node1/data/pixal3d-quality-gate-repro-20260912`에서 현재 fast 코드로 재처리했다. GPU 4
하나를 사용한 cold smoke run은 약 149초에 exit 0이었고 common/SS/shape 3종/PBR 3종의 8개
pack 모두 `completed=1`, `quarantined=0`이었다. 즉 적어도 대표 실패는 현재 코드에서 정상
복구 가능하다. 재현 config와 metadata는 `autoresearch-results/n1-quality-gate-repro*`에 남겼다.

재시작이 만든 leaf 미실행 failed marker 1개와 release history 3개는 삭제하지 않고
`work_queue/remediated/n1-gpu-balance-reload-quality-gate-20260912/`로 이동했다. failed marker
SHA-256은 `fee1e037...2722cfc`, 세 owner record는 각각
`8fd52344...f3961b8`, `7770323f...f9ab72`, `a297e8e2...b21b57`이다. 복구 뒤 active queue는
실제 completed marker 151개, failed 0개, lease 0개이며 node1은 의도적으로 draining 상태를
유지한다. 75개 quality outcome 자체는 변경하지 않았고 production 재개에는 (a) 해당 asset
선택 재처리 또는 (b) 15% 실패 window를 허용하는 명시적 quality 정책 변경 중 하나가 필요하다.

## 2026-09-12 — batch001 75개 실패의 격리 복구와 thermal-safe 재개

10% rolling quality gate를 완화하지 않고 과거 batch001 실패 75개를 현재 fast 코드로 다시
검증하기 위해 canonical과 분리된
`/root/data2/pixal3d-quality-recovery-20260912`,
`/root/data3/pixal3d-quality-recovery-20260912`,
`/root/node1/data/pixal3d-quality-recovery-20260912`를 만들었다. canonical metadata의 해당
75행과 GLB 75개(총 1,141,929,100 bytes)를 복사해 별도 registry를 구성했으며, smoke와 pilot은
각각 실제 end-to-end 처리, audit, evidence/report gate를 통과했다. canonical 입력과 기존
production 산출물은 읽거나 복사만 했고 수정하지 않았다.

75개 production recovery는 GPU 7 하나에서 시작해 chunk000의 64개 mesh/PBR 준비를 완료했고
18개 asset의 render `transforms.json`을 남겼다. 실패 asset 일부는 RGB render 한 건에
4–6분이 걸리는 고복잡도 mesh였다. 실행 중 CPU 97.7%, load1 200.35, 최고 92°C가 되어 설정된
CPU hard-temperature guard가 2026-09-11 20:27:26 UTC에 정상적으로 작업을 중단했다. asset
timeout이나 신규 quarantine에 의한 중단은 아니며 chunk checkpoint와 partial output은 보존됐다.

중단 후에도 n1의 `changwoo_audio_n1` 컨테이너에서 약 72개의 SoundSpaces render와 학습 작업이
CPU를 사용해 load1 116–136, 최고 92–95°C가 유지됐다. 해당 외부 작업은 수정하거나 중지하지
않았다. 복구는 canonical identity hash를 유지하는 `WorkerExecutionConfig`로 CPU 8, dump 8,
render worker/GPU 1, GPU 7, chunk 1개로 제한했고 child process까지 CPU 0–7에 affinity를
상속하도록 했다. thermal watcher는 최고 CPU 온도 <82°C와 load1 <100이 15초 간격 4회 연속
확인될 때만 체크포인트 resume를 시작하며, 기존 85/92°C soft/hard limit는 변경하지 않는다.
현재 production node1은 draining이며 canonical queue/quality ledger/packs는 그대로다. 75개가
모두 복구·감사된 뒤에만 별도 candidate에서 batch001 전체 256개 pack을 재구성하고 canonical
promotion 여부를 결정한다.

## 2026-09-12 — n17 GPU 0/1/4/5로 75개 quality recovery 이전

n1을 다른 사용자가 쓸 수 있도록 n1 recovery process가 남아 있지 않음을 확인하고, timeout 뒤
남아 있던 진단용 `find /` process와 자식만 종료했다. canonical node1은 계속 draining이고
lease가 없으므로 Pixal3D GPU 작업은 n1에서 실행되지 않는다.

n17에서는 요청한 GPU 0, 1, 4, 5가 각각 14 MiB/0%로 비어 있었고, load1 8.68, 최고 CPU
온도 약 57°C, 가용 RAM 387 GiB였다. 기존 `youngwoo_diyscene`은 중지하거나 수정하지 않고
`youngwoo_diyscene_fast_n17_recovery`를 별도로 생성했다. coworker의 canonical data2는
`/canonical/data2/pixal3d`에 read-only로 mount했고, writable output은
`/file2/youngwoo/pixal3d-quality-recovery-n17-20260912`와
`/file3/youngwoo/pixal3d-quality-recovery-n17-20260912`로 분리했다. 컨테이너에는 모든 GPU가
보이지만 `CUDA_VISIBLE_DEVICES`와 `PIXAL3D_GPU_INDICES`는 `0,1,4,5`로 제한했다.

canonical batch001의 75개 실패 목록과 GLB 75개(1,141,929,100 bytes)를 격리 root로
복사했고, source/destination 파일을 전부 SHA-256으로 비교해 75/75 일치를 확인했다. n17의
`903fe647...` checkout은 이후의 noncontiguous GPU preflight 및 scheduling 변경 전이므로
coworker checkout을 수정하지 않고 현재 production worktree의 `data_toolkit`을 recovery 전용
read-only overlay로 동기화했다. overlay 전체 파일 digest는 local/remote 모두
`0ec821da...03ca50b`이다.

n17 NVMe는 89 GiB만 남아 planner의 고정 120 GiB local reserve를 충족하지 못했다. 따라서
246 GiB가 비어 있는 `/dev/shm/pixal3d-quality-recovery-n17-20260912/local`을 재생성 가능한
scratch로 사용하고 pack/checkpoint는 NFS data2/data3에 지속 저장하도록 구성했다. GPU
0/1/4/5 각각의 Blender 4.5.1 OPTIX cube preflight가 CPU fallback 없이 통과했다. 과거 실패
asset을 대상으로 smoke 1개와 pilot 4개를 실제 end-to-end 처리해 각각 completed 1/1과
4/4, quarantine 0을 확인했고 8개 pack family의 manifest/member 검증도 통과했다.

75개 production recovery는 2026-09-12 01:06:08 UTC에 detached로 시작했다. 입력 75개가 모두
기존 raw로 인식되어 다운로드는 0건이며 첫 chunk는 64개다. 초기 관측에서 선택 GPU 네 장 모두
Pixal3D memory를 사용했고 traceback, library error, thermal stop은 없었다. 로그는
`/file2/youngwoo/pixal3d-quality-recovery-n17-20260912/production.log`에 기록된다. 이 실행이
완료·감사되기 전에는 canonical ledger, batch001 packs 또는 production queue를 변경하지 않는다.

첫 실행은 smoke/pilot의 `asset_stats/new_records/part_0.csv`와 production chunk의
`part_chunk000_0.csv`/`part_chunk001_0.csv`를 전역으로 함께 읽어 같은 SHA를 중복으로 판정하면서
75개 모두 `schema_failure`가 됐다. 실제 render/geometry/latent 계산의 실패가 아니었다.
chunk context에서는 해당 `record_prefix` 파일만, prefix가 없는 context에서는 숫자 rank 형태의
`part_<rank>.csv`만 읽도록 validator를 수정했다. 원래 코드에서 중복 SHA 오류를 재현하는 회귀
테스트를 먼저 추가했고 수정 후 통과했으며, n17 container에서 전체 orchestrator 테스트는 환경의
GPU override를 제거한 상태로 203 passed였다. 실패 실행의 pack/checkpoint/log는
`failed-run-20260912T012640Z` 아래에 rollback 자료로 보존했다.

수정본으로 75개를 clean end-to-end 재실행한 결과 2026-09-12 01:49:37–02:24:30 UTC,
총 2,093초(34분 53초), 시스템 처리량 기준 asset당 27.91초와 시간당 약 129개였다. 이 수치는
GPU 0/1/4/5 네 장이 병렬 처리한 전체 wall time을 75로 나눈 값이며, 단일 asset의 직렬 latency는
아니다. parent outcome은 completed 75/75, quarantine 0이었다. common/shape 3종/SS pack에는
75개가 모두 포함됐고, 원래 unsupported shader를 가진 11개만 PBR 3종에서 제외되어 PBR에는
64개가 포함됐다. 총 984개 `.npz`, 8개 family pack, raw archive를 생성했으며 별도 CLI audit도
exit 0으로 통과했다. 이 성공본은 아직 격리 recovery root에 있고 canonical batch001 및 전체
production queue에는 아직 승격하지 않았다.

## 2026-09-12 — recovery 75개 canonical promotion 및 n17 production 준비

canonical을 read-only로 mount한 별도 promotion container에서 기존 batch001의 181개 pack과
recovery 75개 pack을 모두 checksum 검증하고 `/dev/shm/pixal3d-canonical-promotion-n17`에
256개 병합본을 먼저 만들었다. dry run 결과 common/shape 3종/SS는 256개, PBR 3종은 기존 37개와
신규 11개의 unsupported-shader 제외를 합친 208개, raw archive는 256개였으며 8개 pack과 raw의
member/checksum, checkpoint와 quality schema 검증을 모두 통과했다.

canonical 대상 23개 파일의 입력 hash를 고정하고 target lease/terminal marker가 없음을 확인한
뒤, 같은 filesystem의 hard-link/copy rollback backup을
`/root/data2/pixal3d/control/recovery/n17-canonical-promotion-20260912T025429Z`와
`/root/data3/pixal3d/control/recovery/n17-canonical-promotion-20260912T025429Z`에 만들었다.
8개 pack+manifest, raw archive+manifest, prepared index, quality ledger, parent checkpoint의 총
21개 파일을 atomic copy로 교체하고 다시 전부 검증했다. 공식 target batch auditor 결과는
`published_valid=True`, `archive_valid=True`였다. batch001 quality/checkpoint는 completed
256/256, recovered quarantine 0이며 queue에 adoption한 뒤 상태는 completed 152, pending 604,
failed 0, running 0이 됐다. shard 전체 audit는 shard의 나머지 batch가 아직 미완료이므로 올바른
pre-resume gate가 아니어서, target 검증 후 우리가 시작한 audit process만 종료했다.

새 `youngwoo_diyscene_fast_n17` container에는 모든 GPU를 노출하고 canonical data2/data3을
그대로 mount했으며, local scratch만 246 GiB가 비어 있는
`/dev/shm/pixal3d-production-n17-4gpu`로 분리했다. node17 실행 profile은 canonical config hash를
유지하면서 CPU 44, GPU allowlist 0/1/4/5, renderer 2–4 processes/GPU로 preflight를 통과했다.
첫 worker가 batch002를 claim한 직후 coworker의 별도 `anchor-hybrid-prestudy` pilot이 GPU 0–5에
다음 wave를 올린 것을 확인해, GPU leaf 시작 전에 node17을 drain하고 lease를 정식 handoff했다.
다른 작업은 중지하거나 수정하지 않았다. `n17_gpu_watchdog.sh`는 요청 GPU 네 장이 memory 1 GiB
미만·utilization 5% 이하인 상태를 15초 간격 8회(2분) 연속 확인한 뒤에만 node17을 activate하고
canonical supervisor를 시작하도록 현재 실행 중이다.

## 2026-09-12 — production accounting NFS 전수 스캔 제거

watchdog가 처음 supervisor를 시작한 뒤 `batch002` worker는 lease heartbeat를 정상 갱신했지만
8분 이상 GPU leaf나 local scratch를 만들지 않았다. PID namespace를 공유하는 read-only 진단
container에서 syscall을 추적한 결과, worker는 멈춘 것이 아니라 4.26 TB 규모의 기존 data2/data3
프로젝트 전체를 `os.walk`하며 HSSD/ABO 등 현재 batch와 무관한 모든 `renders_cond/<sha>` 파일을
NFS `getdents/stat`하고 있었다. `initialize_project_accounting()`이 유효한 `accounting.json`을
읽고도 즉시 전체 reconcile을 반복했고, queue의 `run_batch()`도 매 batch 완료 시 같은 전체
reconcile을 수행하는 경로가 원인이었다.

유효한 persisted accounting이 있으면 startup에서는 해당 값을 사용하고, batch 경계에서는
publication delta를 atomic checkpoint만 하도록 변경했다. persisted accounting이 없는 최초
실행과 실제 shard 경계의 full reconciliation은 유지했다. 출력 생성·검증·pack schema는
변경하지 않았다. 기존 코드에서 실패하는 회귀 테스트를 추가한 뒤 수정했고, local 관련 suite는
241 passed, n17 production container의 mounted overlay에서는 234 passed였다. 실제 canonical
config에서 accounting 초기화는 8분 이상에서 0.026초로 감소했다.

배포 전 `node17`을 draining으로 바꾸고 출력이 하나도 생성되지 않은 `batch002` lease를 token
`b2940c455b484788ba9566b3f2cab4ff`로 정식 handoff했다. 원격 이전 소스는
`/home/youngwoo/data/pixal3d-quality-recovery-n17-20260912/deploy-backups/accounting-cache-20260912T0404Z`
에 보존했다. queue는 completed 152, pending 604, running 0이다. 배포 후 watchdog은 다시
실행 중이며, coworker의 별도 Blender wave가 GPU 0/1/4/5 일부를 점유하는 동안에는 자동으로
대기하고 네 GPU가 모두 2분 연속 비면 production supervisor를 재개한다.

## 2026-09-12 — n17 중단 및 n7 local-ns3 production 이전

n17에서 재개된 production은 `batch002`부터 `batch005`까지 각 세 번씩 GPU stage 전에
`hardware report does not match held evidence`로 실패했다. canonical hardware input/report의
SHA와 측정 필드는 일치했지만 report는 CUDA 12.4/PyTorch 2.6 evidence를 `passed`로 기록한 반면,
현재 production validator는 CUDA 12.8 및 PyTorch 2.8 이상을 요구해 같은 evidence를 `failed`로
재계산했다. n17의 fast worker를 drain하고 종료해 queue lease가 없음을 확인했으며,
`youngwoo_diyscene_fast_n17`만 exited 상태로 만들었다. coworker의 `youngwoo_diyscene`은 계속
running이며 수정하거나 중지하지 않았다.

n7에서는 GPU 0–5의 RTX 4090 여섯 장이 모두 비어 있고 `/file3`가 `/dev/md0` local ext4,
`/file2`는 NFS임을 확인했다. 새 `youngwoo_diyscene_fast_n7` container는 n7의 기존
`f1.unist.info:443/jhenv6:cu128-260813` 이미지를 기반으로 만들고 모든 GPU를 노출했다. 실제
worker profile은 node7 registration의 GPU 0–5, CPU 44로 제한한다. source와 scratch는 각각
`/file3/youngwoo/pixal3d-n7-runtime/source`와
`/file3/youngwoo/pixal3d-n7-runtime/local`, canonical data2/data3은 기존
`/file2/youngwoo/pixal3d`와 `/file3/youngwoo/pixal3d`를 그대로 mount했다. config hash는
`ff7dc7073940b869b001b8c8e60525390fe15091f518887814ddc1f3ca35e988`로 유지된다.

새 환경은 CUDA 12.8/PyTorch 2.11.0+cu128이며 기존 conda 환경에 없던 `objaverse==0.1.7`만
추가했다. n17 검증에 사용한 동일 Blender 4.5.1을 n7 local scratch로 복사했다. node7의 실제
6-GPU OptiX cube 검증과 local/data2/data3 1 GiB checksum I/O 검사를 수행해 hardware report가
`passed`임을 확인했다. 기존 control artifact는
`control/recovery/n7-hardware-refresh-20260912T045051Z`에 백업했다. 이어 공식
`refresh_gate_evidence()` API로 기존 품질 측정과 output pack은 유지한 채 hardware 참조와
timestamp만 갱신했으며 smoke와 pilot 모두 다시 `passed`였다. 잘못 소진된 batch002–005는
`retry_failed()` API의 incident `n17-hardware-report-mismatch-20260912`로 복구해 queue를
completed 152, pending 604, failed 0, running 0으로 되돌렸다.

첫 node7 batch002 attempt 1은 기본 환경에 설치된 Hugging Face `datasets` package가
namespace-only `data_toolkit/datasets`를 가리면서 `datasets.ObjaverseXL`을 찾지 못해 prepare
단계에서 종료됐다. `data_toolkit/datasets/__init__.py` package marker를 추가해 로컬 adapter가
항상 우선하도록 했다. 외부 동명 package를 둔 회귀 테스트는 수정 전 실패하고 수정 후 관련
52 tests 및 no-excuse 검사를 통과했으며, n7 실제 interpreter에서도 로컬 ObjaverseXL adapter
경로를 확인했다.

batch002 attempt 2는 mesh/PBR dump와 512 해상도 8-view Cycles/OptiX render에 정상 진입했고
신규 traceback은 없었다. 최초 node policy의 동시 chunk 3개와 GPU당 render worker 2–4개는
Blender process를 40개 이상 만들고 load average를 400 이상, CPU idle을 거의 0%로 만들어
GPU가 준비 작업을 기다리게 했다. lease token을 보존한 공식 handoff와 container stop/start로
중간 결과를 보존하면서 해당 실행을 두 번 안전하게 재조정했다. 중단 도중 남은 atomic temp
pickle은 재개 시 validator가 자동으로 찾아 제거하고 해당 asset만 다시 생성했다.

최종 node7 runtime policy는 canonical config hash나 출력 설정을 바꾸지 않고 동시 chunk 1개,
GPU당 render worker 1개(총 6개), dump worker budget 16으로 제한한다. 실제 child command에서
mesh/PBR worker는 각각 5개가 되어 CPU와 GPU 준비 작업을 나눠 사용한다. 이 설정의 새 chunk
초기 관측은 mesh 11개/45초, PBR 6개/38초였고, 과다 병렬 설정에서 남은 asset의 첫 mesh/PBR이
각각 약 136/138초 걸린 것보다 빠르다. 현재 batch002 attempt 2는 node7에서 running이며 queue는
completed 152, pending 603, running 1, failed 0이다. worker PID는 1695563, 로그는
`/file3/youngwoo/pixal3d-n7-runtime/local/logs/node7-first-batch-policy6-20260912.log`다. watcher
PID 1772476은 batch002 completion marker가 생성된 경우에만 같은 container의 canonical
supervisor를 시작하고, 실패하면 node7을 drain한다. supervisor 로그는 성공 전환 후
`/file3/youngwoo/pixal3d-n7-runtime/logs/node7-production-20260912.log`에 기록된다.

05:52 UTC 상태 점검에서 container restart는 0회, node7 health는 true였고 lease heartbeat는
점검 시각까지 갱신되고 있었다. queue는 completed 152, pending 603, running 1, failed 0을
유지하며 traceback, artifact mismatch, OOM, disk-full 오류는 없었다. chunk001은 mesh 64/64와
PBR 64/64를 완료하고 6개 render rank가 각각 약 55–70%까지 진행했다. 시작 약 18분 시점에
64개 중 38개 render directory가 atomic commit됐으며, 실제 표본에는 512 px PNG 8개와
`transforms.json`이 모두 존재했다. 따라서 tqdm의 `processed=0` 표시는 postfix 갱신 문제일 뿐
산출물 누락을 의미하지 않는다. 이 구간의 aggregate render 처리량은 약 28초/asset이다.
60초 GPU 관측 평균은 장치별 0.3–5.2%(peak 7–28%)로 낮았지만 여섯 render process가 병렬로
계속 결과를 commit했다. 개별 asset latency는 약 135–190초이며 CPU 장면 준비와 짧은 OptiX
burst가 번갈아 발생하는 workload 특성 때문에 순간 GPU utilization과 aggregate 처리량을
구분해야 한다.

추가 CPU/GPU 진단에서 node7은 물리 GPU 0–5를 모두 사용하며 각 GPU에 Blender compute
context가 정확히 하나씩 배치된 것을 확인했다. live child의 `CUDA_VISIBLE_DEVICES`도 각 물리
GPU 하나로 제한되고 Cycles는 CPU device를 선택하지 않았다. 반면 container는 약 2,951% CPU와
1,265 threads를 사용했고, 개별 Blender는 134–200 threads와 약 4.5–7 CPU cores를 사용했다.
이 시점에는 geometry/latent PyTorch process가 없었으므로 현재 CPU 상승의 직접 원인은 tensor
fallback이 아니라 Blender의 GLB import, scene/BVH 준비, boundary fitting 및 render 전후
처리다. 다른 host process는 개별 18% 미만으로 주원인이 아니었다.

다만 이후 geometry 단계에 대해서는 CPU tensor 사용이 명시적으로 존재한다.
`dual_grid_view.py`는 CPU tensor를 만든 뒤 o_voxel의
`mesh_to_flexible_dual_grid_cpu`를 호출하고, PBR voxelization도
`textured_mesh_to_volumetric_attr_cpu`를 호출한다. latent encoder의 model/input은 `.cuda()`로
GPU에 올라간다. 따라서 geometry를 GPU화하려면 단순 device 이동이 아니라 o_voxel CUDA backend
구현과 결과 동등성 검증이 필요하다.

또한 node7에 배포한 `master-nfs-fix` renderer가 이전에 선택한 최종 optimized renderer와
다름을 발견했다. 배포본은 모든 boundary retry를 최종 512 해상도 Cycles로 수행한다. 로컬
`master`는 낮은 `boundary_resolution`에서 fitting하고 최종본만 512로 render하지만 production
supervisor가 없으며, `master-nfs-fix`는 supervisor/NFS/package fix를 포함하지만 이 render
최적화가 빠져 있다. 현재 작업은 출력 호환성은 유지하지만 이전에 보고한 53.55초 optimized
implementation이 아니므로, 두 변경 집합을 통합·재검증하기 전에는 active batch 중간에 source를
교체하지 않는다.

## 2026-09-12 — node7 작업 중지 및 Blender/native bpy 조사

사용자 요청에 따라 node7을 `draining`으로 바꾸고 first-batch watcher를 종료한 뒤 container를
stop/start하여 모든 전처리 자식 프로세스를 정리했다. 실행 중이던 batch002 attempt 2 lease는
token을 확인한 공식 queue handoff로 반환했다. 최종 queue는 completed 152, pending 604,
running 0, failed 0이며 `youngwoo_diyscene_fast_n7`에는 entrypoint와 보조 서비스만 남아 있다.

현재 production renderer는 conda Python에서 `bpy`를 직접 import하는 방식이 아니다.
`data_toolkit/render_cond.py`가 asset마다 Blender 4.5.1 binary를 `-b -P` 옵션으로
`subprocess.run`하고, 그 Blender 내장 Python이 `blender_script/render_cond.py`에서 `bpy`를
import한다. 반면 n7 conda 환경에는 별도의 native `bpy==4.4.0`이 이미 설치돼 있다. 이 모듈을
`CUDA_VISIBLE_DEVICES=3`으로 실행하자 물리 GPU 3(PCI 9c:00)의 RTX 4090 하나만 열거했고,
128px/16-sample Cycles OptiX probe를 2.90초에 정상 완료했다. 따라서 이 환경에서는 native
`bpy`도 GPU를 사용할 수 있고 `CUDA_VISIBLE_DEVICES`도 작동하므로 GPU마다 container를 분리할
필요는 없다. 다만 production Blender 4.5.1과 native bpy 4.4.0의 버전이 달라 지금 즉시
교체하면 출력 동등성을 보장할 수 없다. 동일 4.5.1 module을 준비한 뒤 GPU별 장수명 worker에서
여러 asset을 처리하여 process/kernel startup을 재사용하고, Blender datablock 정리와 memory
증가 및 PNG/transforms 동등성을 검증하는 방식이 적절하다.

중지 직전 batch는 1024 geometry 단계에 진입해 있었고, 이 시점의 CPU 사용은 실제로
`dual_grid_view`와 PBR voxelizer의 CPU backend에서 발생했다. 설치된 `o_voxel==0.0.1` extension은
`mesh_to_flexible_dual_grid_cpu`와 `textured_mesh_to_volumetric_attr_cpu`만 export하며 대응 CUDA
함수가 없다. 관련 CUDA 심볼은 별도 용도의 `rasterize_voxels_cuda`뿐이고 wheel에는 backend를
수정할 C++/CUDA source도 포함돼 있지 않다. 따라서 입력에 `.cuda()`를 붙이는 수정은 동작하지
않으며, geometry GPU화는 extension-level CUDA 구현/재빌드와 sparse coordinate, attribute,
latent 동등성 검증이 필요한 별도 작업이다. 그 전에도 view별 mesh concatenate, normalization,
deep-copy 및 동일 geometry 재계산을 공유하는 CPU 최적화는 가능하다.

## 2026-09-12 — production renderer 통합 및 100-asset native bpy 검증

production 호환 branch에 128px boundary fitting과 마지막 512px Cycles render를 통합했다.
128px에서도 Cycles를 그대로 사용하면 10개 기준 916초로 기존 512px fitting 898초보다 오히려
느렸다. fitting engine만 EEVEE로 바꾸면 external Blender가 415초로 2.16배 빨라졌다. 이어
`bpy==4.5.1`을 optional dependency로 고정하고 asset마다 Blender process를 시작하지 않는 native
worker를 구현했다. 한 worker를 무제한 재사용하면 10개에서 메모리가 1.08GB에서 3.75GB까지
증가했으므로, 8개 asset마다 worker를 종료하고 새 process를 만드는 bounded lifecycle을
적용했다. Python 3.11의 `max_tasks_per_child`는 8개 뒤 남은 task를 재시작하지 못한 실제
deadlock이 있어, metadata를 명시적인 8개 batch로 나누는 방식으로 수정했다. 10개 단일 GPU
결과는 기존 external/full-Cycles 898초에서 native/EEVEE 184초, 즉 89.8초/asset에서
18.4초/asset으로 4.88배 향상됐다.

100개/800 view에서는 legacy가 retry 10회에 도달한 세 view에서 저해상도 loop와 최종 validation의
retry budget이 중복 적용되는 문제를 발견했다. 저해상도 fitting이 budget을 소진하면 해당 view만
기존 full-resolution loop를 정확히 다시 실행하도록 수정했다. 수정 후 기준본 대비 100개
`transforms.json` 전체와 800개 radius, camera matrix, alpha가 모두 exact였다. raw RGB PSNR은
min/p50/p95/p99 24.81/44.85/55.09/60.25dB였다. 기존 코드는 Cycles RNG를 고정하지 않고
retry마다 32-sample render를 반복하므로, 최종 render 한 번만 수행하는 수정본과 Monte-Carlo noise
sequence가 달라진다. 동일 수정본 반복 실행의 median PSNR은 99.31dB였다. 따라서 PNG 검증은
raw PSNR 단독 threshold 대신 exact transform/radius/alpha를 필수 gate로 두고 RGB PSNR 분포를
보조 지표로 사용한다.

native EEVEE는 한 container에 GPU를 모두 노출하면 `CUDA_VISIBLE_DEVICES`와 관계없이 graphics
context가 물리 GPU 0에 집중됐다. 이 구성은 100개에 535초가 걸리고 GPU 0 VRAM이 8.16GB까지
올랐다. GPU마다 하나의 GPU만 노출한 6개 격리 container에서는 EEVEE가 각 GPU로 분산돼 424초,
즉 시스템 처리량 4.24초/asset이었고 all-visible 구성보다 1.26배 빨랐다. native full-resolution
기준본 765초 대비로는 1.80배 향상이다. rank별 worker는 8개마다 정상 recycle됐고 peak RSS는
container당 2.73–4.10GB, peak VRAM은 GPU당 3.45–4.54GB였다. 따라서 native EEVEE production
배포에는 GPU별 container 격리가 필요하며, 기존 단일 production container는 아직 교체하지 않았다.

## 2026-09-12 — 100-asset geometry/latent 동등성 및 OPTIX/CUDA 비교

동일 view에서 256/512/1024가 반복하던 mesh transform과 PBR transformed deep-copy를 view당 한 번만
수행하도록 변경했다. shape는 100개, PBR은 shader 지원 대상 85개를 views 0–1과 세 해상도에서
기존/수정 코드로 각각 실행했다.

| 항목 | 기존 | 수정 | 결과 |
|---|---:|---:|---|
| shape dual-grid wall | 608초 | 607초 | 속도 차이 없음 |
| shape peak RSS | 20.40GB | 22.15GB | 개선 없음 |
| PBR voxel wall | 548초 | 598초 | 이번 run에서 9.1% 느림 |
| PBR peak RSS | 23.74GB | 19.42GB | 18.2% 감소 |
| geometry 동등성 | 2,220 files | 2,220 files | `.vxz`/scale 전부 byte-exact |

shape transform 재사용은 호출 횟수만 3회에서 1회로 줄였고 end-to-end 시간에는 영향이 없었다.
PBR deep-copy 재사용은 속도 이득은 확인하지 못했지만 peak RSS를 줄였다. 두 경로 모두 실제
병목은 `o_voxel` CPU kernel이다.

동일 geometry에서 6 GPU latent A/B를 실행했다. 기존은 최대 rank wall 587초였고 PBR-1024
micro-batch 4에서 rank 0이 fragmentation OOM을 일으켜 누락 5개를 micro-batch 2로 49초에
재개했다. 수정본은 600 shape + 200 SS + 510 PBR `.npz`를 177초에 OOM 없이 완료해 최초
기준 wall 대비 3.32배 빨랐다. 기준 run에는 최초 model cache download가 포함됐다. 양쪽의
2,620개 latent/scale 파일 수, schema, shape, dtype와 sparse coordinates는 모두 exact였다.
최대 relative L2 차이는 shape 0.0420%, SS 0.0769%, PBR 0.1417%였다. 수정본 peak RSS는
28.35GB로 기준 22.02GB보다 높았고, peak VRAM은 양쪽 모두 24GB 한계에 가까웠다. 수정본은
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 및 adaptive micro-batch로 OOM을 피했다.

같은 100개에서 최종 Cycles backend만 비교하면 CUDA는 357초, OPTIX는 424초로 CUDA가 15.8%
빨랐고 메모리도 조금 적었다. 그러나 radius와 matrix는 800/800 exact인 반면 alpha byte-exact는
439/800, binary mask exact는 766/800이었고 최저 mask IoU는 0.99520이었다. 결과 연속성을
우선하므로 production 기본값은 OPTIX를 유지한다. CUDA는 IoU 기반 동등성을 허용할 때만 opt-in
후보로 남긴다.

최적화 후 CPU geometry가 약 6초/asset으로 가장 큰 남은 stage가 됐으므로 o_voxel CUDA backend는
성능상 구현 가치가 있다. 다만 설치된 wheel에는 대응 CUDA symbol이나 C++/CUDA source가 없어
현재 production branch에서 즉석 구현하지 않는다. 별도 extension project에서 exact sparse
coordinates, 제한된 attribute/latent 오차, bounded VRAM, 실제 end-to-end speedup을 acceptance
조건으로 개발하는 것이 결론이다. 이 모든 실험 동안 node7 production은 draining 상태,
queue는 completed 152/pending 604/running 0/failed 0으로 유지했고 coworker container와 canonical
입출력은 수정하지 않았다.

fragmentation OOM 재발 방지를 위해 위 실험에서 검증한
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 production encoder command 환경에도
반영하고 회귀 테스트를 추가했다. 최종 관련 suite는 실행 가능한 worktree 임시 경로에서
113 passed였고 `git diff --check`도 통과했다. `/tmp`가 `noexec`인 현재 host에서만 Blender
installer fixture 3건이 permission error를 냈으며 같은 테스트를 executable 경로로 옮기면
모두 통과했다. 실험 종료 후 n7의 benchmark 전용 container 7개는 중지했으며 production
`youngwoo_diyscene_fast_n7`과 n17의 coworker `youngwoo_diyscene`은 수정하거나 재시작하지 않았다.

## 2026-09-13 — 격리된 100-asset ObjaverseXL production-path qualification

exact code snapshot `a9114d82588e019e529204affdc97f08d46ca47f`를 n7의 완전히 분리된
input/output/scratch에 배포하고, 실제 Objaverse GLB 100개를 6개 one-GPU container로 나눠
download, stage, prepare/render, geometry/latent, validation, 8개 family pack, raw archive,
cleanup까지 전체 11-command graph로 실행했다. 모든 GLB의 SHA-256이 metadata와 일치했고,
quality ledger는 completed 100, quarantine 0이었다. 6개 checkpoint는 active attempt 없이
11개 command가 모두 완료됐고, 48개 prepared pack과 6개 raw archive manifest는 전부 exact
tool commit 및 qualification config hash를 기록했다. 지원되지 않는 shader를 가진 15개 asset은
기존 정책대로 PBR 세 family에서만 제외됐으며 다른 family에는 100개가 모두 포함됐다.

frozen full-resolution OPTIX 기준과 exact snapshot의 100 assets/800 views를 비교해 asset/file
set, retry, camera matrix/radius/angle, AABB, scale/offset 및 alpha 800/800이 exact임을 확인했다.
unseeded lighting 때문에 독립 run RGB PSNR은 hard gate로 사용하지 않았고, camera/alpha와
artifact schema/membership을 production gate로 유지했다. 별도의 clean benchmark에서는
renderer 전체 경로가 10 assets 기준 898초에서 184초로 4.88배, 100 assets/6 GPU 기준
765초에서 424초로 1.80배 빨라졌다. latent 기준의 587초 run은 model cache download와
OOM 뒤 별도 49초 재개를 포함하므로, 177초 수정 run과의 비율은 진단값일 뿐 clean speedup
판정으로 사용하지 않는다.

E2E 도중 다른 사용자가 GPU 0--3에서 학습을 시작해 해당 GPU의 세 qualification container만
중지하고 빈 GPU 4/5에서 checkpoint resume했다. 세 shard 모두 완료됐고 이미 완료된 shard의
두 번째 resume는 10.10초 no-op이었다. 이 때문에 이번 E2E wall time은 speed benchmark가
아니라 isolation/resume/completeness 증거로만 사용한다. 중단된 shard 로그에서 SIGKILL 뒤
0.5초 안에 worker reap이 완료되지 않으면 불필요한 `could not kill worker process` 예외가
발생하는 race를 발견했다. 두 geometry worker의 post-SIGKILL grace를 bounded 5초로 늘리고
delayed-reap regression을 추가했다(`756f5d1`). 수정 후 전체 suite는 892 passed, warning 1개였다.

canonical production은 completed 152, pending 604, running 0, failed 0으로 그대로이며,
qualification container 9개는 제거했다. 상세 수치와 증거 경로는
`docs/benchmarks/2026-09-13-preprocessing-qualification.md`에 기록했다.

최종 review에서 native 배포 runbook의 multi-GPU 등록 예시가 one-visible-GPU 계약과
충돌하는 것을 발견해, GPU마다 고유 container/node-id/local-root를 사용하는 6-worker
예시로 수정했다. `RENAME_EXCHANGE`를 지원하지 않는 NFS에는 previous-output rollback
fallback을 추가했고, Objaverse direct input은 descriptor-relative `O_NOFOLLOW`로 열어
고정한 inode를 private alias와 inherited FD로 Blender에 전달하도록 보강했다. 같은 경로가
검증 직후 symlink로 교체되어도 원래 inode를 읽는 회귀 테스트를 포함한다.

`data_toolkit.benchmark_quality`에는 unseeded RGB diagnostic와 seeded 50 dB gate, exact
alpha/camera/coordinate, latent relative L2 0.2% gate를 구현했다. frozen 800 PNG 비교는
failure 0/alpha IoU 1.0, frozen 1,310 latent NPZ 비교는 failure 0/max relative L2
0.1417104%로 통과했다. qualification config, 100개 asset list, 비교 report, evidence hash,
test log를 저장소에 포함했다. 전체 suite 최종 결과는 900 passed, 기존 `torch.cross`
deprecation warning 1개였다.

마지막 code review에서 저해상도 fitting은 수렴했지만 첫 512px 검증이 불일치하는 경우,
기존 구현이 저해상도 retry를 차감한 남은 budget만 쓰는 corner case를 발견했다. target
resolution 검증이 한 번이라도 실패하면 저해상도 radius를 버리고 원래 radius에서 legacy
full-resolution 10회 loop를 그대로 재실행하도록 수정했다. 기존 100-asset exact-camera
경로는 최초 target 검증을 통과하므로 추가 연산이 없고, 불일치 mesh만 보수적 fallback을
사용한다. 이 수정까지 포함한 전체 suite는 900 passed, warning 1개였다.

최종 handoff review에서 추가로 세 가지 fail-closed 조건을 보강했다. comparator는 reference나
candidate root가 없거나 comparable artifact가 0개이면 즉시 실패한다. NFS render publication은
asset별 `flock`으로 직렬화하고, 새 output 게시와 기존 output rollback이 모두 실패한 경우에도
`.previous`를 남겨 다음 resume에서 복구한다. target-resolution disagreement fallback은 문자열
검사가 아니라 replay 조건과 원래 radius/10회 budget 초기화 함수를 실행하는 regression으로
검증한다. 관련 전체 suite는 runtime commit
`c1002e990a74187f5b01f2290238f7a4eb698262`에서 906 passed, warning 1개였다.
추가 실패 주입에서는 이미 `.previous`가 존재하는 상태의 publish/rollback 이중 실패도
last-known-good을 보존하고 다음 resume에서 복구함을 확인했다. comparator는 reference와
candidate 양쪽의 non-finite latent를 모두 fail-closed 처리한다.

성능 수치의 원본 log digest와 rank별 wall time을 `benchmark-timings.json`에, geometry 2,220개
파일의 reference/candidate hash-manifest digest를 `geometry-hash-summary.json`에 기록했다.
운영 container는 mutable host source bind 대신 검증된 runtime tree를 COPY한 overlay image를
사용하도록 runbook을 바꿨다. image 내부 commit marker를 supervisor 시작 전에 exact 비교하며,
host checkout이 이후 바뀌어도 실행 중 코드는 변하지 않는다. 신뢰 경계, descriptor-pinned input,
archive path 검증, NFS rollback/recovery 동작도 qualification 문서에 명시했다.

n7에서 overlay image를 실제 build해 base의 `bpy 4.4.0`을 pinned `bpy 4.5.1`로 교체하고,
GPU 0 하나를 노출한 임시 container에서 commit marker, Blender 4.5.1, Torch 2.11.0+cu128,
RTX 4090 인식을 확인했다. 임시 image/container/build root는 검사 후 제거했다.
운영 runbook은 기존 non-Git source directory에 의존하지 않고 fork의 exact commit을 새 release
directory에 detached checkout하며, dirty/untracked 상태를 거부하고 base image도 digest로 고정한다.

최종 runtime `87b58b49a07f91d0d7c069b11044a5881fade64f`
(`data_toolkit` tree `b7b315125600bbfa4d134a6f6e2987f12e9d6103`)에서는 추가 handoff
검토 결과를 반영했다. comparator는 JSON/NPZ 양쪽의 non-finite numeric 값을 거부하고,
asset SHA-256은 경로 생성 전에 엄격한 lowercase 64-hex로 검증한다. Objaverse direct/zip
input은 processor에 넘기기 직전 열린 descriptor/member의 digest를 다시 확인해 앞선 검사 뒤의
파일 교체도 탐지한다. overlay image build는 build context 내부 Git commit, `data_toolkit` tree,
clean status를 직접 검증하고 `bpy==4.5.1` wheel SHA-256을 고정한다. n7에서 이 exact runtime을
새로 build해 GPU 0 smoke를 통과했으며 전체 test suite는 911 passed, warning 1개였다.

최종 handoff 검토에서 양쪽에 동일한 일부 파일만 존재해도 comparator가 통과할 수 있는
불완전-result case를 발견했다. frozen manifest에서 가져온 `.json/.npz/.png/.vxz` expected
count를 모두 필수로 만들고, 누락·초과 산출물은 fail-closed 처리했다. 보존된 n7 결과에 새
계약을 적용해 render 800 PNG+100 JSON과 latent 1,310 NPZ+1,310 JSON을 다시 통과시켰다.
NFS fallback publish에는 candidate 파일/디렉터리 및 각 rename metadata의 `fsync` barrier를
추가했고, 선택된 ZIP member에는 8 GiB 압축 해제 상한을 추가했다. 운영 runbook은 공용 Blender
tools를 각 GPU-local root에 read-only 중첩 mount하고 data2의 원본 `raw` subtree 역시
read-only로 중첩 mount하도록 수정했다. 이 최종 runtime은
`b67c6b371028952fe423c1379b408871e1941274`이며 `data_toolkit` tree는
`489c3a256ff083872ff4fd1b18837e0544b766c2`이다.
전체 suite는 916 passed, warning 1개였다. 같은 commit으로 n7 overlay image를 새로 build해
commit/tree marker, Git metadata 제거, `bpy`/external Blender 4.5.1, Torch 2.11.0+cu128,
RTX 4090 한 장 노출 및 원본 raw mount의 read-only 상태를 확인하고 임시 image/container/build
root를 제거했다.

## 2026-09-13 — n7 canonical production 재개

기존 `youngwoo_diyscene_fast_n7`과 다른 사용자 작업은 변경하지 않고, 비어 있던 n7 물리 GPU 5만
노출하는 `youngwoo_diyscene_fast_node7-gpu5` container를 생성했다. CPU quota는 7 cores로
제한했으며 canonical data2/data3는 기존 경로를 사용하고, `/root/data2/pixal3d/raw`와 공용
Blender tools는 중첩 read-only mount했다. worker별 scratch는
`/file3/youngwoo/pixal3d-n7-runtime/local/gpu5`로 분리했다.

최초 immutable image 실행에서는 세 가지 production blocker를 발견했다. raw가 read-only인데
Objaverse download record를 raw 아래에 기록하던 동작은 writable metadata root로 분리했고
(`d35a48e`), mutable 기존 container에만 수동 설치돼 있던 `objaverse==0.1.7`과
`GPUtil==1.4.0`을 hash-pinned image dependency로 명시했다(`2ef2aa4`). 그 뒤 worker-local
`gpu_count=1`을 공유 hardware report의 수집 topology에도 적용해 6-GPU n7 evidence를 거부하는
문제를 수정했다(`ab00bba`). 최초 report 생성은 configured GPU count를 계속 엄격히 검사하고,
이미 고정된 report를 읽을 때만 evidence 자체의 GPU inventory로 재검증한다. worker가 실제로
한 GPU만 볼 수 있는지는 별도 environment preflight가 그대로 검사한다.

GPU-count 오류가 attempt 3까지 진행된 batch002는 실패 marker와 history를
`control/runtime/work_queue/remediated/n7-hardware-report-worker-mismatch-20260913`에 보존한 뒤
retry했고, 다음으로 claim된 batch003은 preprocessing 시작 전에 정식 operator handoff했다.
복구 직후 queue는 completed 152, pending 604, running/failed/stale 0이었다. 새 image
`pixal3d-fast:ab00bba`는 commit `ab00bba6362588cdcc0596cfe074b3cb89c6e12e`, data_toolkit tree
`bdae9d59c25fd3f672beda3709cdba5055e145aa`로 고정됐다. 전체 test suite는 executable basetemp에서
920 passed, warning 1개였고, claim 없는 n7 preflight에서 worker GPU 1, held-report GPU 6,
decision passed와 canonical config hash 일치를 확인했다.

04:37 UTC에 canonical queue를 재개했다. batch002가 attempt 1로 claim되어 prepare 단계와
native OPTIX render에 정상 진입했다. 초기 관측에서 container CPU는 quota 범위인 약 6.5--6.9
cores, RSS는 약 5.3 GiB, GPU 5 VRAM은 약 1.0--3.4 GiB였으며 Cycles render 시 GPU activity를
확인했다. canonical queue와 최종 output은 기존 `/file2/youngwoo/pixal3d` 및
`/file3/youngwoo/pixal3d` 계약을 그대로 사용하고 이미 completed인 152개 unit은 건너뛴다.

05:07 UTC에 타 사용자 작업이 끝나 GPU 0/1이 비었지만, 신규 worker의 claim 전 검증에서 기존
n7 hardware report가 생성 후 87,036초 경과해 24시간 freshness 한도를 넘은 것을 탐지했다.
host/container clock skew나 report/evidence 불일치는 없었다. 신규 node는 즉시 draining으로
되돌렸고 lease는 생성되지 않았다. 이후 GPU 2/3도 비어 active production GPU가 0/1/2/3/5가
됐다. 기존 evidence와 smoke/pilot report는
`control/recovery/n7-hardware-refresh-20260913T0511Z` 및
`control/recovery/n7-hardware-refresh-expand-20260913T0517Z`에 순서대로 보존했다.

GPU 0/1/2/3/5만 노출한 임시 container에서 다섯 RTX 4090의 isolated OPTIX cube와 local/data2/
data3 1 GiB checksum I/O를 다시 측정했다. 새 hardware report는 GPU 5개, decision `passed`,
canonical config hash를 기록하며 smoke와 pilot evidence도 공식 `refresh_gate_evidence()` API로
재발행한 뒤 모두 read-back `passed`를 확인했다. 임시 container와 storage fixture는 제거했다.
각 GPU별 no-claim 검증에서 visible GPU 1개, held report GPU 5개, environment/hardware decision과
config hash 일치를 확인한 뒤 GPU 0/1/2/3 worker를 추가했다. 05:28 UTC queue는 completed 152,
running 5(batch002--006), pending 599, failed/stale 0이며 각 container CPU quota는 7 cores라
총 상한은 35 physical cores다. GPU 4의 타 사용자 process는 건드리지 않았다. 모든 production
container는 host 재시작 뒤 GPU 점유를 재확인하지 않고 자동 실행되지 않도록 restart policy를
`no`로 유지한다.

기존 production checkpoint 중 548개가 최적화 이전 stage 이름과 node-local 경로를 담고 있어,
새 worker가 이미 완료된 chunk를 검증하는 과정에서 legacy stage를 현재 4-stage 계약과 비교하지
못하거나 사라진 이전 node의 raw metadata를 요구하는 문제가 드러났다. former 9-stage 이름을
현재 stage로 정규화하고 dependency 순서대로 완료 상태를 검증하며 parent metadata를 fallback으로
사용하도록 수정했다(`32bbbf2`). parent와 child local metadata가 모두 없을 때에는 checksum이
검증된 canonical raw를 worker-local source로 다시 stage하도록 보강했다(`6acb806`). 각 수정 후
전체 suite는 924 passed, 기존 `torch.cross` warning 1개였고, 관련 targeted suite는 219 passed였다.
최신 image `pixal3d-fast:6acb806`은 runtime commit
`6acb8060e2388498ba0768ddc33379f70f50cf4a`, data_toolkit tree
`e89b402a34f7a255f2c42a68df8fb0a491c786e7`로 고정했다.

legacy checkpoint 전체 26 MiB는 변경 전에
`control/recovery/n7-legacy-checkpoint-schema-20260913T0550Z/checkpoints-before-resume.tar.gz`
및 SHA-256 sidecar로 보존하고 검증했다. 현재 shard에서 네 chunk가 promoted됐지만 durable
publication은 없고 parent quality failure가 10%를 넘은 batch006/007/008/012/013/015는 재사용할
수 없는 cross-node partial state로 판정했다. 같은 shard lease가 모두 없는 상태에서 parent/chunk
checkpoint, 이전 quality ledger와 queue history를
`control/recovery/n7-invalid-promoted-resume-20260913T0615Z`로 이동해 보존하고, completed unit만
남도록 quality ledger를 재구성한 뒤 terminal batch006/007을 공식 retry했다. GPU3의 batch006/008
local scratch도 `/file3/youngwoo/pixal3d-n7-runtime/local/gpu3/control/remediations/`
아래 같은 incident 이름으로 이동했다. 복구 전 dry-run과 합성 회귀 테스트를 통과했고, 적용 후
queue는 completed 152, pending 604, running/failed/stale 0이었다.

06:35 UTC에 GPU 0/1/2/3/5별 container를 모두 최신 image로 교체했다. 각 container는 RTX 4090
한 장만 보며 `bpy 4.5.1 LTS`, 7-core quota, 32 GiB shared memory, raw/tools read-only mount 및
restart policy `no`를 확인했고, claim 없는 `validate_worker_environment()`를 모두 통과했다.
이전 worker의 batch002--005 lease는 정식 handoff한 뒤 같은 unit을 다시 claim했으며 batch002는
완료된 chunk를 건너뛰고 chunk003부터 재개했다. reset된 batch006은 새 chunk000--003 checkpoint를
만들어 canonical raw에서 다시 시작했다. 재개 직후 queue는 completed 152, running 5, pending 599,
failed/stale 0이고 초기 오류는 없었다. GPU4는 production에 사용하지 않았다.

06:44 UTC 재확인에서 이전에 GPU4를 사용하던 타 사용자 compute process가 종료된 상태가 연속
관측됐고 VRAM 사용량도 0 MiB였다. 다른 container를 변경하지 않고 GPU4만 노출한
`youngwoo_diyscene_fast_node7-gpu4`를 같은 immutable image와 read-only raw/tools mount로
추가했다. draining 상태에서 `bpy 4.5.1 LTS`, visible RTX 4090 한 장 및 전체 worker environment
검증을 통과한 뒤에만 활성화했다. GPU4 worker는 reset된 batch007을 attempt 1로 claim해 running
상태에 진입했고 초기 오류는 없었다. 여섯 container의 CPU quota 합계는 42 cores로 physical
44-core 제한 이내이며 restart policy는 모두 `no`다.

06:51 UTC GPU process ownership 감사에서 타 사용자 `inha`의 별도 작업이 GPU0에 새로 진입해
우리 renderer와 같은 물리 GPU를 공유하는 것을 발견했다. 타 사용자 process/container는 변경하지
않고 `node7-gpu0`만 registry draining으로 전환하고 전용 container를 중지한 뒤, 실행 중이던
batch002 lease를 `gpu0-released-for-other-user` 사유로 공식 handoff했다. GPU0에는 타 사용자
process만 남았음을 확인했고 우리 container는 restart policy `no` 상태로 stopped다. batch002의
기존 chunk003 scratch/checkpoint는 삭제하지 않아 다음 worker가 이어받을 수 있다. 철수 직후 queue는
completed 152, running 5, pending 599, failed/stale 0이며 활성 production CPU quota 합계는
35 cores다.

자원 기준선은 여섯 worker가 동시에 prepare/render하던 20초 구간에서 GPU별 평균 utilization
0.3--5.8%, 최대 4--39%, 평균 VRAM 2.2--3.0 GiB, 최대 VRAM 3.3--4.1 GiB였다. container CPU는
합계 약 41.4 cores로 42-core quota 안에 있었다. 이는 end-to-end asset 완료 시간이 아니라 CPU
mesh/PBR dump와 native Cycles가 겹치는 prepare 구간의 표본이다. GPU0 철수 후 GPU 1--5의 동일
active chunk에서 완성된 8-view render 디렉터리는 약 5분 동안 34개에서 48개로 증가했다. 이 render
단계 관측 처리율은 aggregate 약 21.4초/asset, GPU당 환산 약 106.8초/asset이며 초기 raw staging은
제외되지만 이후 geometry/latent 및 canonical publish 시간은 포함하지 않는다. 각 renderer는
`native_worker_max_assets=8` 경계에서 child를 정상 교체했고 supervisor/lease heartbeat와 산출물
갱신이 계속됐으며 오류는 0건이었다.

07:03 UTC 장기 실행 resource 감사에서 data2는 28 TiB, data3와 worker-local scratch filesystem은
18 TiB가 남아 있었고, 다섯 active scratch 합계는 약 15.3 GiB였다. available RAM은 150 GiB,
I/O PSI는 `avg10=0.01`, vmstat iowait는 0%라 storage/RAM gate는 안전했다. 우리 다섯 container의
CPU 사용량 합계는 약 34.3 cores로 35-core quota 안이었다. 반면 host load average는 약 215,
CPU PSI `some avg60`은 약 93%로 타 사용자 CPU workload와 강한 경합이 관측됐다. 이는 낮은 GPU
utilization과 production render 처리율에 영향을 주는 외부 요인이지만, 다른 사용자 작업은
변경하지 않고 현재 worker 수와 quota를 유지했다.

초기 staging 영향이 줄어든 07:08--07:12 UTC의 4분 20초 연속 관측에서는 GPU 1--5 active
chunk의 완성된 8-view render 디렉터리가 82개에서 101개로 19개 증가했다. steady render 단계
처리율은 aggregate 약 13.7초/asset, GPU당 환산 약 68.4초/asset이었다. 같은 구간 내내 다섯
supervisor/worker process와 lease heartbeat가 유지됐고 queue는 failed/stale 0, 오류 로그도
0건이었다. GPU0의 타 사용자 process는 계속 유지되어 재투입하지 않았다.

07:30 UTC GPU3의 batch005/chunk003이 64개 asset render를 모두 완료하고 geometry/latent 단계로
전환했다. `renders_cond` 아래 65번째 디렉터리는 asset이 아니라 renderer metadata용
`new_records`였고, instances 64개와 asset render 이름은 extra/missing 없이 일치했다. authoritative
chunk checkpoint의 `render.elapsed_seconds`는 3,196.48초로 49.95초/asset이었다. 이후
`geometry_encode_bundle`이 view 0/1의 shape/PBR 256·512·1024를 생성하면서 latent encoder와
producer/consumer 방식으로 겹쳐 실행됐다. 초기 4분 관측에서 geometry `.vxz`는 106개에서
269개, latent는 0개에서 48개로 증가했다. container RSS는 9.45--12.51 GiB, encoder model 로드 후
VRAM은 약 5.02 GiB였고 OOM/error는 없었다.

07:54--07:58 UTC에 GPU0의 타 사용자 compute process가 종료되고 VRAM 사용량이 0 MiB인 상태를
30초 간격으로 연속 확인했다. 기존 우리 `youngwoo_diyscene_fast_node7-gpu0` container의 image
ID, GPU0 단독 노출, 7-core quota, 32 GiB shared memory, restart policy `no`, canonical raw와
Blender tools의 nested read-only mount를 다시 확인했다. registry를 `draining`으로 유지한 채
container를 시작하고 claim 없는 `worker --once` 환경 검증이 exit 0인 경우에만 node를
활성화했다. supervisor는 동일한 immutable image에서 별도 log
`control/logs/supervisor-6acb806-gpu0-resume-20260913T0756Z.log`로 시작했다. GPU0 worker는 공식
queue에서 기존 batch002 attempt 1을 다시 claim했고, 완료된 chunk000--002를 재실행하지 않고
보존된 batch002/chunk003의 `prepare_bundle`로 재개했다. queue는 completed 152, running 6,
pending 598, failed/stale 0이며 여섯 container의 CPU quota 합계는 42 cores로 44-core 제한
이내다.

08:10 UTC 전후 병렬 `prepare_bundle` 감사에서 batch004/chunk000이 64개 render와 mesh/PBR
dump를 모두 생성한 뒤에도 stage 완료로 인정되지 않고 chunk001로 넘어간 현상을 조사했다.
chunk별 `ShardContext`가 parent의 전역 prefix를 버리고 `chunk000_`만 사용해, 서로 다른 batch의
`asset_stats/new_records/part_chunk000_0.csv`가 같은 source metadata 경로에서 경쟁적으로
교체되고 있었다. 실제 shared CSV의 63개 SHA는 batch003/chunk000과만 일치하고 동시에 실행된
batch002/004/005/006/007과의 교집합은 0개여서 cross-batch overwrite를 확인했다.

chunk record prefix가 parent의 `<shard>_<batch>_` namespace를 보존한 뒤 chunk id를 붙이도록
수정하고, 같은 chunk id를 가진 두 parallel batch가 서로 다른 prefix를 생성하는 회귀 테스트를
추가했다. 수정 전 테스트는 두 값이 모두 `chunk000_`여서 실패했고 수정 후에는
`ABO-00000_batch003_chunk000_`와 `ABO-00000_batch004_chunk000_`를 확인했다. scheduler,
commands, pipeline integration, orchestrator 관련 272개 테스트와 전체 925개 테스트가 통과했다.
전체 suite의 최초 4개 Blender fixture 실패는 host `/tmp`가 `noexec`인 환경 때문임을 독립 실행
fixture로 확인했으며, executable worktree-local `TMPDIR`에서는 모두 통과했다. 기존 in-flight
chunk는 새 asset-stats part를 재생성하되 이미 검증 가능한 render/mesh/PBR 결과는 재사용할 수
있어 canonical 산출물을 삭제하거나 덮어쓸 필요가 없다.

수정 commit `7869c38a4fd2b25c88443a0380872014e7e90d71`과 data_toolkit tree
`a0f854d7808b8b99b1d114d6f551209fd688ca31`로 n7 image `pixal3d-fast:7869c38`
(`sha256:1f66d062420264356c80f70bd9f434189d447c1647ef5bc33b432577ff7ee73a`)을
build했다. 격리된 두 개의 실제 `asset_stats` process가 같은 `chunk000`을 병렬 실행해도
`batch003`과 `batch004`의 part 파일과 SHA 집합이 분리되는 것을 확인했다. GPU 0--5별 새
container는 각각 RTX 4090 한 장만 보며 7-core quota, 32 GiB shared memory, `bpy 4.5.1`,
raw/tools read-only mount, restart policy `no` 및 claim 없는 worker preflight를 통과했다. 이전
image container는 삭제하지 않고 stopped backup으로 보존했다.

기존 여섯 lease는 registry drain 후 공식 handoff했고, batch005의 실패/history는
`control/runtime/work_queue/remediated/n7-parallel-record-prefix-race-20260913T0830Z`에
보존한 뒤 공식 retry했다. queue가 completed 152, pending 604, running/failed/stale 0인 것을
확인한 후에만 새 image supervisor를 활성화했다. 새 실행 argv의 record prefix가 실제로
`ObjaverseXL_sketchfab-00001_batch004_chunk000_`처럼 batch까지 포함하는 것도 확인했다.
batch002는 완료되어 08:54 UTC 상태가 completed 153, running 6, pending 596, failed 1,
stale 0이 됐다.

batch005는 수정 image에서도 세 번 재시도한 뒤 chunk001/002 finalize의 기존 attempt budget
소진으로 다시 failed가 됐다. 정확한 checkpoint를 붙여 validator를 직접 실행한 결과 각 chunk의
기존 성공 asset 하나에서 `renders_cond/<sha>/transforms.json`이 실제 누락돼 있었다. 그러나
완료 command의 사전 검증이 durable quality checkpoint/ledger를 수정할 수 있었고, 모든 terminal
quality outcome을 후보에서 제외해 기존 `completed` asset의 누락 파일을 검사하거나 재생성하지
못하는 별도 결함이 확인됐다.

완료 출력 사전 검증을 read-only context로 만들고 이때 quality outcome을 기록하려 하면
재생성을 요구하도록 변경했다. 또한 완료 command를 재검증하거나 재실행하는 동안에는 이전
`completed` asset만 candidate/instances에 다시 포함하고, `failure`와 `schema_failure` asset은
계속 제외한다. 수정 전 두 회귀 테스트는 각각 command를 잘못 skip해 실패했고, 수정 후 기존
terminal outcome을 바꾸지 않은 채 성공 asset만 재생성 대상으로 전달함을 확인했다. 관련
orchestrator/scheduler/commands/integration 302개와 전체 927개 테스트가 통과했으며 기존
`torch.cross` warning 1개 외에는 경고나 실패가 없었다.

복구 수정은 commit `671d7a29e919bec4b1d63238a4cc0e3022363d0f`, data_toolkit tree
`c3161c66d50b6dc6b15cbb856c212996020e46ea`로 master와 production branch에 push했다. n7에서
exact detached source로 image `pixal3d-fast:671d7a2`
(`sha256:d21f470b543913ead0881679e1437fc8bcf568ff9b3f1d2c9ce10fac1229ac36`)을 build하고,
image marker/label, `bpy 4.5.1`, 두 복구 회귀와 prefix 회귀를 통과시켰다. 여섯 node를 모두
draining으로 만든 후 active lease 6개를 `runtime-upgrade-671d7a2` 사유로 공식 handoff해
completed 153, pending 602, failed 1, running/stale 0 상태를 확인했다. 이전 image container는
삭제하지 않고 `_7869c38_backup` 이름의 stopped backup으로 보존했다.

새 image는 GPU 0--5 각각에 one-visible-GPU container로 배치했다. 각 container의 7-core quota,
32 GiB shared memory, restart policy `no`, exact runtime marker/tree, Torch GPU 1개,
native/external Blender 4.5.1, raw/tools read-only mount 및 draining 상태의 claim 없는 실제 worker를
모두 검증했다. batch005 실패 이력은
`control/runtime/work_queue/remediated/n7-completed-output-repair-20260913T0910Z`에 보존하고
retry했다. supervisor 재개 후 batch003--008이 GPU 0--5에 attempt 1로 claim됐고 queue는
completed 153, running 6, pending 597, failed/stale 0이었다.

GPU2의 batch005/chunk001은 기존 failure 1개를 제외하고 이전 성공 63개를 정확히 담은
`control/eligible/prepare_bundle.txt`로 복구 실행에 진입했다. 이전 실패 뒤 local render output은
보존돼 있지 않아 chunk001/002의 transforms와 8-view PNG set이 모두 0개였으므로 이 두 chunk는
전량 재생성이 필요하다. dump_mesh 3개, dump_pbr 3개와 native OPTIX renderer 1개가 실제로
병렬 실행되는 것을 확인했다. 재개 직후 여섯 container CPU는 각각 약 6.36--7.04 cores,
RSS는 4.64--5.68 GiB였고 GPU2 render 표본은 utilization 36%, VRAM 3.36 GiB였다. 모든 lease
heartbeat가 유지됐고 supervisor error log는 비어 있었다.

계속된 checkpoint/output 대조에서 batch005/chunk003도 64개 `completed` outcome과
`prepare_bundle` attempt 3을 기록했지만 local render transforms는 57개만 남아 있었다. 따라서
완료 stage의 불완전 output을 찾아도 과거 attempt budget이 이미 3이면 복구 command를 실행하기
전에 중단되는 경로를 production 실패 전에 재현했다. 일반 미완료 command의 3회 retry 제한은
유지하면서, 과거 완료 stage의 output이 나중에 불완전하다고 검증된 경우에만 현재 resume에서
1회의 post-budget repair attempt를 허용했다. 이 복구 attempt가 실패하면 즉시 중단하며 반복
retry하지 않는다.

수정 전 회귀 테스트는 `command attempt budget already exhausted`로 command 실행 없이 실패했다.
수정 후 post-budget 복구 성공과 단일 복구 실패의 fail-closed 동작을 모두 확인했고 관련 suite
304개와 전체 929개 테스트가 통과했다. full suite 최초 실행의 Blender fixture 4개 실패는 삭제된
worktree-local `TMPDIR` 때문에 Python이 host의 noexec `/tmp`로 fallback한 환경 문제였으며,
실행 가능한 temp directory를 다시 만든 뒤 해당 4개와 전체 suite가 통과했다. 같은 시점의
production queue는 completed 153, running 6, pending 597, failed/stale 0으로 유지됐고
09:19--09:26 UTC의 render transforms는 여섯 batch 합계 254개에서 274개로 증가했다.

수정 commit `456251c443270614e02bfb1920bd2d6680cc6aa0`, data_toolkit tree
`3bf6c5f7bc8365ce63ff9a4db7cef64c36178f56`를 master와 production branch에 push하고 n7에서
image `pixal3d-fast:456251c`
(`sha256:e954f68ed50f0d6946697210bfe4578832e053a3758a955dd720d244e040d2e9`)로 build했다.
image label/marker, Git metadata 제거, `bpy 4.5.1`과 완료-output 복구/prefix 회귀 5개를 격리
code smoke로 확인했다. 모든 active GPU process가 기존 Pixal3D container cgroup 소유인 것을
확인한 뒤 여섯 node를 drain하고 active lease를 `runtime-upgrade-456251c` 사유로 공식 handoff했다.
queue가 completed 153, pending 603, running/failed/stale 0인 상태에서만 기존 container를
`_671d7a2_backup` 이름의 stopped backup으로 보존했다.

GPU0 실제 one-visible-GPU smoke와 GPU 0--5별 새 container의 exact runtime/tree, 7-core quota,
32 GiB shared memory, restart policy `no`, Torch GPU 1개, native/external Blender 4.5.1,
raw/tools read-only mount 및 claim 없는 worker를 모두 통과시켰다. worker-local partial state를
다른 GPU에 배치하지 않도록 supervisor를 순차 시작해 GPU0--5가 각각 기존 batch003--008을
다시 claim했는지 lease owner를 하나씩 확인했다. 재개 queue는 completed 153, running 6,
pending 597, failed/stale 0이었고 supervisor 오류는 없었다. batch005/chunk001은 handoff 전
산출물을 보존해 재개 직후 mesh 63, PBR 42, transforms 14, PNG 124개로 계속 증가했다.

09:48 UTC GPU process ownership 재감사에서 타 사용자 `bang_vlm` container의 feature-cache
process가 GPU5에 진입해 우리 batch008 renderer와 겹친 것을 발견했다. 다른 사용자 process와
container는 변경하지 않고 node7-gpu5만 즉시 drain하고 batch008 lease를
`gpu5-released-for-other-user` 사유로 공식 handoff한 뒤 전용 container를 중지했다. queue는
completed 153, running 5, pending 598, failed/stale 0, active CPU quota 합계 35 cores가 됐다.
해당 외부 compute process가 종료되고 GPU5가 비어 있는 상태를 30초 간격으로 연속 확인한 뒤,
stopped container의 exact image와 claim 없는 preflight를 다시 확인하고 node를 활성화했다.
batch008을 동일 GPU5 local scratch로 다시 claim한 후 queue는 running 6, pending 597로 복구됐고,
추가 타 사용자 GPU5 process는 관측되지 않았다.

### 2026-09-13 n7 canonical production 실측

10:14 UTC에 GPU4의 `ObjaverseXL_sketchfab-00001/batch007/chunk001`이 새 runtime에서
prepare, render, encode, finalize 전 단계를 완료하고 chunk002로 진행했다. checkpoint의 64개
quality outcome은 모두 `completed`였고 failure/schema_failure는 없었다. stage elapsed는
prepare 28.39초, render 1,720.67초, encode 1,793.67초, finalize 18.83초로 합계
3,561.57초, 즉 55.65초/asset이었다. 기존 최종 100-asset 벤치마크 53.55초/asset과 3.9%
차이로 production에서도 동일한 처리량 수준이 재현됐다.

해당 checkpoint에는 이전 실행에서 누적된 `prepare_bundle` 3회와
`geometry_encode_bundle` 2회의 attempt 이력이 남아 있지만, 이번 resume의 최종 output
validation은 통과했다. 같은 시각 queue는 completed 153, running 6, pending 597,
failed/stale 0이고 모든 lease heartbeat가 정상이었다. 여섯 container의 순간 CPU 사용 합계는
약 28 cores로 42-core quota 이내였고 RSS는 container당 약 4--10 GiB였다.

GPU0 batch003/chunk001에서는 한 ObjaverseXL asset
`08e87a669e4ffac0d0607a21d94086e300152b55848f2878bd0e0adf3f71a555`의 external Blender
`dump_mesh`가 900초 제한으로 timeout되어 mesh pickle이 생성되지 않았다. supervisor와
chunk command는 계속 살아 있고 다음 asset의 PNG가 증가하므로 worker 장애나 memory leak이
아닌 개별 입력 품질 문제로 분류했다. worker를 재시작하거나 canonical output을 수정하지 않고
기존 validator가 종료 결과를 판정하도록 두었다.

10:36:47 UTC prepare 종료 후 해당 timeout asset 하나만 `asset_validation` terminal
`failure`로 quarantine됐고 worker는 즉시 `geometry_encode_bundle`로 진행했다. PBR pickle이
없던 나머지 10개 asset은 모두 `unsupported_shader` 사유로 PBR-256/512/1024 family만
제외됐으며 terminal asset failure로 확장되지 않았다. 즉 64개 입력 중 geometry/RGB candidate
63개를 유지하면서 지원하지 않는 shader의 PBR family만 정확히 분리했고, timeout 하나가 전체
chunk나 queue failure를 일으키지 않는 것을 production ledger로 확인했다.

11:09 UTC batch003/chunk001의 geometry, 세 해상도 voxel cleanup과 output validation까지
완료됐다. 최종 quality outcome은 63개 `completed`, timeout asset 1개 `failure`였고 worker는
같은 lease에서 batch003/chunk002 prepare를 자동 시작했다. checkpoint stage elapsed는 prepare
42.10초, render 3,154.35초, encode 1,927.22초, finalize 21.30초로 합계 5,144.97초,
64개 입력 기준 80.39초/asset이었다. 이 합계에서 단일 불량 mesh의 강제 timeout 900초를
제외한 정상 처리 경로는 4,244.97초, 66.33초/asset이다. geometry 중간 관측에서 voxel은
shape 63개와 PBR 53개의 3해상도×2-view 합계 696개를 모두 생성했고, shape/SS/PBR latent도
family별 검증을 거쳐 output validation을 통과했다. geometry 구간 container RSS는 약
11.8 GiB까지 올라갔다가 6 GiB대로 회수돼 누적 memory leak 징후가 없었다.

10:23 UTC에 GPU2의 batch005/chunk001 post-budget repair가 production에서 완료됐다.
`pipeline.json`은 `stage_raw`, `prepare_bundle`, `geometry_encode_bundle`, 세 해상도 voxel cleanup,
`validate_outputs`를 모두 completed command로 기록했고, quality outcome은 복구 대상 성공 asset
63개 `completed`와 기존 입력 불량 asset 1개 `failure`였다. worker는 lease를 유지한 채 즉시
batch005/chunk002의 eligible repair set으로 넘어가 3개 mesh dump, 3개 PBR dump와 native
renderer를 시작했다. 따라서 완료 출력 누락을 복원하면서 기존 terminal failure는 재처리하지
않는 `456251c`의 production 계약이 실제 canonical resume에서도 확인됐다.

12:43:51 UTC에 GPU3의 batch006/chunk002가 중단됐던 `prepare_bundle`을 attempt 2로 자동
재개했다. attempt 1에서 asset
`0a32e4b3cfd356ab5232440603bd76baed377f9cd981867e990a1d62af17cc7a`의 external Blender
`dump_mesh`가 900초 제한으로 timeout됐지만, 재시작이나 수동 checkpoint 변경 없이 scheduler가
다른 active chunk를 진행한 뒤 해당 chunk로 돌아왔다. 재시도 eligibility는 64개 중 63개였고
timeout asset은 제외돼 같은 900초 작업을 반복하지 않았다.

12:52:45 UTC `prepare_bundle` attempt 2가 완료돼 worker는 `geometry_encode_bundle` attempt 1로
진행했다. checkpoint의 render elapsed는 535.50초였다. 최종 geometry 입력은 shape/SS 62개와
PBR 53개였으며, quality ledger에는 기존 schema failure 1개와 Blender timeout 1개만 terminal
`asset_validation` quarantine으로 남았다. PBR이 없는 나머지 9개 asset은
`unsupported_shader` family exclusion으로만 기록돼 shape/SS 처리 대상은 유지됐다. 이로써
장시간 개별 Blender 실패가 queue failure나 무한 재시도로 번지지 않고, terminal asset과
지원하지 않는 PBR family를 분리한 채 자동 복구되는 것을 canonical production에서 확인했다.

같은 실행 중 GPU2 batch005가 한 차례 unit을 release한 뒤 같은 batch를 다시 claim해
`geometry_encode_bundle` attempt 3으로 진행한 사실을 확인했다. supervisor 원문은
chunk001/encode와 chunk003/render에서 `cannot persist attempt before launch: invalid checkpoint
attempts`를 보고했다. completed-output 복구는 기본 3회 budget 소진 후 정확히 한 번의 post-budget
attempt 4를 의도하고 테스트도 그 값을 요구했지만, durable checkpoint parser는 여전히 attempt를
3 이하로 제한하고 있었다. 따라서 attempt 4를 저장하는 순간 infrastructure failure로 unit이
release되고, 재claim 후 다른 chunk부터 반복하는 불일치였다.

completed command에 한해서만 attempt 4 checkpoint를 허용하고, 이미 attempt 4가 기록된
completed-output 복구는 attempt 5를 launch하지 않도록 수정했다. non-completed command의 attempt
4는 save/load 시 계속 fail-closed한다. post-budget repair와 checkpoint round-trip 회귀 테스트
5개, 전체 orchestrator 213개, 전체 suite 932개가 통과했다. Ruff 전체-file 검사는 기존 파일의
baseline 위반 47개를 보고했으나 이번 변경 라인에서 새 위반은 없었다.

수정 commit `090b409fa6f8f0e8fe2f215ec0f1284dade826c9`, data_toolkit tree
`15bd61a9aa0d27d6e1bc142f47aae2c4bfb299c6`를 master에 push하고 n7에서 exact detached source로
image `pixal3d-fast:090b409`
(`sha256:cbfecca685cc38b1047019f24f8dc2748d122a931a0ac74b6b74862385ccda6d`)를 build했다.
claim 없는 image smoke의 marker/tree, Git metadata 제거, `bpy 4.5.1`과 post-budget 회귀 5개,
GPU2 한 장만 노출한 smoke의 RTX 4090/Torch, native 및 external Blender 4.5.1, tools read-only
mount를 모두 통과했다.

GPU2부터 GPU0, GPU1, GPU3, GPU4, GPU5 순서로 각 node만 drain하고 active lease를
`runtime-upgrade-090b409` 사유로 공식 handoff했다. 매번 해당 물리 GPU의 compute PID가 기존
container cgroup에만 속하는지와 handoff 후 GPU가 비었는지 확인했으며 다른 사용자 process는
변경하지 않았다. 기존 `456251c` container는 GPU별 `_456251c_backup` 이름의 stopped backup으로
보존했다. 새 container는 모두 7-core quota, 32 GiB shm, restart policy `no`, one-visible GPU,
raw/tools read-only mount와 claim 없는 실제 worker preflight를 통과했고, GPU0--5가 각각 기존
batch003--008을 같은 local scratch로 다시 claim했다.

rollout 직후 GPU2 batch005/chunk001의 `geometry_encode_bundle` attempt 4가 새 parser를 통해
durable checkpoint에 정상 저장돼 실행 중인 것을 확인했다. 같은 supervisor log에는
`invalid checkpoint`, batch release, traceback이 없었다. 따라서 합성 round-trip뿐 아니라 실제
canonical post-budget repair에서도 attempt 4 persistence 결함이 제거됐으며, queue는 completed
153, running 6, pending 597, failed/stale 0으로 복구됐다.

14:03 UTC GPU3 batch006/chunk002는 최종 output validation까지 완료했다. 64개 입력 중 62개가
`completed`, Blender 900초 timeout 1개가 `failure`, 별도 입력 1개가 `schema_failure`였으며,
worker는 같은 lease에서 chunk003으로 진행했다. 이어 batch006/chunk001의
`prepare_bundle` attempt 3회가 소진됐지만 quality outcome이 전혀 없는 현상을 조사했다.
GPU5, GPU4, GPU3의 세 worker-local scratch에는 각각 handoff 직전까지 mesh/PBR/render 파일이
계속 생성된 흔적이 있었고, chunk001 관련 Blender timeout이나 Python exception은 없었다.

세 차례 runtime image 교체 시 생성된 operator-handoff 기록은 모두 `refunded_attempts=[]`였다.
원인은 handoff CLI가 scheduler의 canonical checkpoint인
`<data2>/control/checkpoints/<source>/<shard>/chunks/<batch>/<chunk>/pipeline.json`이 아니라 존재하지
않는 `<local>/preprocess/active/<shard>/<batch>/chunk_checkpoints`를 조회한 것이었다. 따라서 정상
작업 중 image 교체로 중단된 prepare가 실패가 아닌데도 매번 attempt를 소비했고, 세 번째 이후
command budget이 고갈됐다.

handoff CLI가 lease의 source/shard/batch와 worker registry의 `data2_root`로 canonical checkpoint
root를 구성하고, refund helper가 shard identity를 확인한 뒤 그 경로만 no-follow 방식으로
읽고 쓰도록 수정했다. remediation evidence는 실제 `checkpoint_root`를 기록하는 schema version
2로 갱신했다. production topology를 사용하는 failing-first CLI 테스트는 수정 전 attempt가
`2`로 남아 실패했고, 수정 후 `2 -> 1` 환급을 확인했다. 관련 targeted 테스트 3개와 전체
suite `932 passed, 1 warning in 119.72s`를 통과했으며 warning은 기존 `torch.cross` deprecation
한 건이다. 별도 `/dev/shm` fixture에서 실제 `python -m data_toolkit.pipeline.cli queue --action
handoff` 명령을 실행한 manual smoke도 running 0/pending 1, active attempt 제거, remediation
evidence 생성까지 확인했다. production GPU3 node는 현재 batch006 lease가 끝날 때까지
`draining`으로 두어 새 unit claim과 checkpoint 복구가 경쟁하지 않도록 했다.

14:27 UTC GPU ownership watcher가 다른 사용자의 `/gs_anigauss` container가 GPU 0, 3, 4, 5에서
학습을 시작한 것을 감지했다. 외부 process/container는 변경하지 않고 해당 네 Pixal3D node만
drain한 뒤 우리 container를 중지했다. 방금 수정한 exact source를 기존 image에 read-only
mount한 claim 없는 CLI container로 각 lease를 공식 handoff했다. schema 2 evidence는
batch003의 active command 2개, batch006의 1개, batch007의 2개, batch008의 2개 등 총 7개의
중단 attempt를 각각 한 번 환급했다. 이후 queue는 completed 153, running 2(GPU 1·2), pending
601, failed/stale 0이었고 GPU 0·3·4·5에는 외부 학습 process만 남았다.

모든 worker를 잠시 draining으로 만들어 새 claim을 차단하고, lease가 없는 batch006/chunk001의
`prepare_bundle` attempt만 `3 -> 0`으로 복구했다. 세 attempt가 각각 GPU5/GPU4/GPU3에서 정상
파일 생성 중 runtime cutover로 중단됐다는 앞선 mtime/handoff 증거와 checkpoint의
`active_attempt=null`, `completed_commands=[stage_raw]`, 빈 quality outcome을 모두 precondition으로
검사했다. 원본과 전후 SHA-256은
`control/runtime/work_queue/remediated/n7-handoff-attempt-recovery-20260913T1440Z`에 보존했으며,
복구 후에도 batch006 lease가 생기지 않은 것을 확인한 뒤 GPU 1·2를 다시 활성화했다.

수정 commit `60de7fe5`, documentation commit `54bc0fce340a7a4ea5ff04215bf3ed1b9db0f666`,
data_toolkit tree `c48c1d22f16cef729ee592f86246997c9d9c7cd6`를 master에 push했다. n7의 clean detached
source에서 image `pixal3d-fast:54bc0fc`
(`sha256:7d6eb6aa3505b13627dfd1ca2c8e6be97b1ae1283a3f1f89a45402e784a9c3ca`)를 build했다. image
label/marker, Git metadata 제거, `bpy 4.5.1 LTS`, canonical handoff 회귀 3개를 claim 없는 smoke로
확인했다. GPU smoke와 stopped node 재개는 GPU 0·3·4·5의 외부 학습이 끝나 빈 상태가 두 번
연속 관측된 뒤에만 수행하도록 monitor를 교체했다.

15:32--15:41 UTC GPU 0·3·4·5가 compute process 없이 비어 있는 상태를 30초 이상 간격으로
두 번 확인했다. 중지된 `090b409` container는 삭제하지 않고 GPU별 `_090b409_backup`으로
보존하고, 같은 local scratch와 canonical data2/data3 mount를 사용하는
`pixal3d-fast:54bc0fc` one-visible-GPU container를 새로 생성했다. 네 container 모두 exact
image revision/tree, Git metadata 제거, Torch가 인식하는 단일 RTX 4090, native/external
Blender 4.5.1 LTS, raw/tools read-only mount, 7-core quota, 32 GiB shm와 draining 상태의 claim
없는 실제 `worker --once`를 통과했다. node를 하나씩 활성화해 GPU0/3/4/5가 각각 기존
batch003/006/007/008을 원래 worker-local scratch에서 재claim한 것을 확인했고, queue는
completed 153, running 6, pending 597, failed/stale 0으로 복구됐다.

15:42 UTC cgroup ownership monitor가 다른 사용자의 `gs_anigauss` container가 GPU 0에 다시
진입한 것을 감지했고, 수동 교차 확인 중 GPU 3 진입도 이어서 확인했다. 외부 process와
container는 변경하지 않고 node7-gpu0/3만 drain하고 해당 Pixal3D container를 중지했다.
`54bc0fc`의 수정된 canonical handoff 명령으로 batch003/006 lease를 각각 반환해 중단된
command attempt를 환급했다. GPU 1·2·4·5의 네 worker는 계속 실행 중이며 15:45 UTC queue는
completed 153, running 4, pending 599, failed/stale 0, active CPU quota 합계 28 cores였다.
모니터는 이 네 GPU의 container cgroup 소유권과 queue failure/staleness를 검사하고 GPU 0·3의
연속 free 상태를 기다리도록 갱신했다.

15:47--15:50 UTC GPU 3의 외부 compute process가 사라진 상태를 30초 이상 간격으로 다시
확인했다. 기존 `54bc0fc` GPU3 container를 시작해 exact marker, 단일 RTX 4090,
native Blender 4.5.1과 claim 없는 worker smoke를 재확인한 뒤 node를 활성화했다. scheduler는
반환된 unit 중 queue 선두인 batch003을 node7-gpu3에 배정했다. canonical completed checkpoint는
계속 skip되지만 GPU0 local scratch의 미완료 중간물은 GPU3에서 보이지 않으므로 그 부분만 다시
실행될 수 있다. 결과 계약에는 영향이 없고 GPU를 유휴 상태로 두는 것보다 완료 시간이 짧아지는
방향이라 lease를 유지했다. 16:06 UTC queue는 completed 153, running 5, pending 598,
failed/stale 0이었다. 10분 자원 표본의 container CPU 합계는 약 11.8 cores(설정 상한 35), RSS
합계는 약 28.7 GiB였고 worker 최대 VRAM은 GPU3 약 7.6 GiB였다. GPU0은 외부 작업에 계속
양보한 채 draining/stopped 상태로 유지했다.

16:30--16:38 UTC GPU0의 외부 compute process가 사라진 상태를 monitor와 30초 간격 독립
검사로 확인했다. pending queue 선두인 batch006의 기존 partial은 GPU3 local scratch에 있었고
GPU3는 현재 batch003만 처리해 해당 디렉터리를 쓰지 않았다. canonical output은 변경하지 않고
GPU3의 batch006 scratch를 read-only source로 GPU0의 존재하지 않던 target에 복제했다. 복제는
7.4 GiB/3,916 files, 111초가 걸렸고 기존 target overwrite는 없었다. 첫 GPU smoke shell은
검증문의 quoting 오류로 `SyntaxError`를 냈지만 node가 draining인 상태여서 worker/lease는
시작되지 않았다. quoting만 수정해 native `bpy 4.5.1 LTS`, 단일 RTX 4090과 claim 없는
`worker --once`를 통과시킨 후 node를 활성화했다. node7-gpu0이 batch006 attempt 2를 정확히
claim했으며 queue는 completed 153, running 6, pending 597, failed/stale 0으로 복구됐다.
6-GPU cgroup ownership monitor도 다시 시작했고 직후 container CPU quota 합계 42 cores,
실측 약 13.8 cores, RSS 약 31.2 GiB였으며 worker 최대 VRAM은 GPU2 약 23.2 GiB였다.

## 2026-09-13 — rolling quality gate 재발 격리와 batch002/005 복구 시작

17:13:50 UTC node7-gpu2의 batch005가 attempt 3에서
`end-to-end failures exceed 10% (309/500)`으로 terminal failed가 됐다. 새 unit claim을 막기 위해
node7-gpu0--5를 모두 draining으로 바꾸고 우리 six production container만 중지한 뒤, 당시 active
lease인 batch003/004/006/007/008/012를 `quality-gate-recovery-20260913` 사유로 공식 handoff했다.
각 token과 중단 attempt 환급 evidence를 보존했으며 다른 사용자의 process/container는 변경하지
않았다. containment 후 canonical queue는 completed 153, pending 602, failed 1(batch005),
running/stale 0이었다.

실패는 최근 leaf retry 수가 아니라 source별 500-asset rolling ledger였다. 해당 window는 batch002
failure 244건과 batch005 completed 191/failure 65건으로 구성됐다. canonical parent/chunk를 다시
대조하면 batch002는 256/256이 모두 generic `asset output failed validation`으로 격리된 채 빈 pack을
publish했고, batch005는 chunk000 64건과 chunk001 1건이 같은 generic failure, 나머지 191건이
completed였다. 의심 asset 321개 모두 canonical metadata와 non-empty GLB가 존재했고 원본 합계는
4,470,424,448 bytes였다. 따라서 quality threshold를 완화하지 않고 실제 재처리로 false failure를
판별하기로 했다.

canonical과 분리된
`/file2/youngwoo/pixal3d-quality-recovery-n7-20260913`,
`/file3/youngwoo/pixal3d-quality-recovery-n7-20260913`,
`/file3/youngwoo/pixal3d-n7-runtime/quality-recovery-20260913`을 만들었다. metadata/checkpoint/output은
이 root에만 쓰고 canonical ObjaverseXL GLB와 Blender tools는 read-only bind했다. registry는
training 321개와 evaluation 1개로 생성했다. 일반 preflight는 configured source와 무관하게 모든
dataset을 검사하는 기존 동작 때문에 HSSD token 및 3D-FUTURE/Toys4k manual archive에서 blocked였으나,
실제 one-visible-GPU hardware preflight는 Blender 4.5.1, CUDA 12.8, Torch 2.11.0+cu128,
OPTIX, CPU fallback 없음으로 passed였다. 1 GiB fixture 기준 data2 쓰기 50.82 MiB/s,
data3/local 쓰기 216.17/210.74 MiB/s였고 모든 fixture는 제거됐다.

batch002, batch005/chunk000, batch005/chunk001에서 각각 1개를 end-to-end smoke로 실행했다.
앞의 두 asset은 canonical failure였지만 현재 코드에서 모든 output/pack을 정상 생성했고 final
8-view render는 각각 237.97초와 414.05초였다. 마지막 `09b4609f...`는 mesh dump 662.41초,
PBR dump 446.38초 뒤 final render가 900초 제한에 도달해 재현 가능하게 timeout됐다. 이 1건은
genuine quarantine으로 유지하고 나머지 320건만 복구 대상으로 확정했다. 실패-only smoke pack은
삭제하지 않고 sibling diagnostics root로 18개 경로/24개 파일과 move manifest를 보존했다.
성공한 두 smoke의 registry/frozen scope/pack/raw archive/checksum report는 config hash
`578e495f...d08d911`로 passed했다. 별도 batch002 pilot 1개도 render 164.47초를 포함한 전체
pipeline과 같은 hash의 report를 통과했다.

320개를 64개씩 다섯 recovery shard로 처음 병렬 실행할 때 shared checkpoint parent를 동시에
생성한 shard2 하나가 `FileExistsError`로 leaf launch 전에 종료됐다. no-follow directory walker가
`open(ENOENT)` 직후 다른 worker가 directory를 만든 정상 race를 unsafe destination으로 오판한
것이 원인이었다. concurrent winner를 허용한 뒤 같은 `O_DIRECTORY|O_NOFOLLOW` open으로 최종
검증하도록 수정했고, failing-first race test와 기존 symlink rejection tests를 포함한 focused
225 tests가 통과했다. 전체 suite는 929 passed였고 `/tmp` noexec 때문에 실패한 Blender fixture
4개는 executable workspace basetemp에서 4/4 통과했다. 수정 commit은 `836563b`이며 fork의
master와 production branch에 push했다.

초기 five-shard 준비 단계는 cgroup CPU 합계가 35 cores로 44-core 상한 이내였지만 Blender와
filesystem worker가 동시에 늘며 host load1이 156까지 올라갔다. shard2--4를 중지해 scratch와
checkpoint를 보존하고 shard0--1부터 처리했으며, 이후 load가 내려간 뒤 shard2를 config identity를
유지하는 CPU 5/render 1/dump 5 profile로 resume했다. 강제 중지 때 생긴 비어 있는 임시 mesh
pickle 2개는 resume validator가 제거했고 유효 mesh 7개부터 재개했다. 세 active container의
cgroup quota는 각각 4 cores, 합계 12 cores로 낮춰 load1을 72 아래에서 유지하고 있다. canonical
queue와 pack/quality ledger는 격리 복구·감사가 끝날 때까지 변경하지 않는다.

## 2026-09-14 — read-only raw cleanup 복구 및 shard 00001 완료

격리 복구 shard 00001은 64개 asset의 render, geometry, latent, pack publication까지 모두
완료한 뒤 `archive_raw` cleanup에서 `OSError: [Errno 18] Invalid cross-device link`로 종료됐다.
raw GLB는 data2의 canonical raw subtree를 read-only nested bind mount한 것이고 quarantine은
그 상위 writable mount에 만들어져, 같은 backing filesystem이어도 Linux mount boundary를
가로지르는 `rename`이 `EXDEV`를 반환했다. 따라서 이 오류는 NFS 처리 속도나 산출물 손상이
아니라 read-only 원본을 삭제하려던 publication 이후 정리 단계의 mount 계약 불일치였다.

`_unlink_regular_beneath`가 quarantine rename에서 정확히 `EXDEV`가 발생한 경우에만 externally
mounted raw를 그대로 보존하고 성공으로 처리하도록 최소 수정했다. 다른 rename 오류와 unsafe
path 오류는 계속 전파한다. 수정 전 실제 예외를 재현하는
`test_raw_delete_retains_source_when_quarantine_crosses_mount_boundary`를 추가했고, 수정 후 cleanup
안전 테스트 4개와 data_toolkit 전체 928개 테스트가 통과했다. host `/tmp`가 `noexec`라 실패한
Blender fixture 4개는 executable `/dev/shm` TMPDIR에서 4/4 통과했다. 수정 commit은
`5271d8e76025b66c9843d90d768f97f035a66e6f`, data_toolkit tree는
`cf10b656e581f9a2ea3f47277fd31356400a54ee`이며 fork의 production branch에 push했다. n7 image
`pixal3d-fast:5271d8e`는
`sha256:98668aa74ec52fdec6c438712d0b8acf6709713b949ba1b2d7a6ef0f2e073fef`로 고정했다.

첫 새-image resume은 기존 published pack이 `tool_commit=294521a...`를 기록한 반면 isolated
recovery config에는 compatibility attestation이 없어 현재 commit `5271d8e...`만 허용되면서,
이미 완료된 geometry와 pack을 다시 검증·생성한 뒤 `different valid published pack`으로
fail-closed했다. 기존 pack hash와 member set은 변하지 않았고 checkpoint도 `build_packs`
완료를 보존하고 있었다. `5271d8e`의 제품 변경은 publication 이후 raw cleanup뿐이므로, 이
격리 recovery 실행에만 `PIXAL3D_TOOL_COMMIT=294521a9310c4401be2c988fabd3e406d2a4b72e`를 적용해
기존 pack provenance를 유지했다. canonical config나 production output contract는 바꾸지 않았다.

GPU 3이 memory 100 MiB 미만, utilization 0%, compute PID 없음, host load1 50 미만 상태를
30초 간격 10회 연속 통과한 뒤 shard 00001을 one-visible-GPU container로 재개했다. 실제
container는 CPU 10 cores, RAM 40 GiB, shm 32 GiB로 제한했고 raw와 Blender tools를 read-only로
mount했다. GPU 재연산 없이 기존 publication 검증과 cleanup만 수행해 91초 만에 exit 0,
`OOMKilled=false`로 끝났다. 최종 checkpoint는 11개 command가 모두 완료되고
`active_attempt=None`, quality ledger는 completed 64/quarantine 0이다. production pack 8개와
raw archive 1개의 manifest/checksum을 공식 `pipeline.cli audit`로 다시 검증해 exit 0을
확인했다.

나머지 recovery shard 00000/00002/00003/00004는 n7의 다른 사용자 compute process가 GPU를
점유하는 동안 durable coordinator에서 대기한다. launcher는 동일한 5분 안정성 gate를
통과한 GPU만 선택하며 foreign process가 실행 중인 GPU에는 container를 만들지 않는다.
이 시점 canonical queue는 completed 153, pending 602, failed 1(batch005), running/stale 0이고
active lease는 없다. 격리 shard 전체가 완료·감사되기 전에는 canonical pack, raw archive,
checkpoint, quality ledger 또는 queue marker를 변경하지 않는다.

## 2026-09-15 — n7 burst GPU 경합에 맞춘 recovery scheduler 조정

GPU 3이 기존 30초 간격 10회 안정성 gate를 통과한 뒤 recovery shard 00002를
`pixal3d-fast:5271d8e` exact image로 재개했다. container는 GPU 3 한 장만 노출하고 CPU 10 cores,
RAM 40 GiB, shm 32 GiB로 제한했으며 canonical raw와 Blender tools mount는 read-only였다.
약 7분 동안 유효 mesh pickle은 18개에서 43개, PBR pickle은 10개에서 22개, 완성 render는
0개에서 2개로 증가했다. 이후 타 사용자 PID 3361978이 GPU 3에 진입하자 15초 ownership monitor가
우리 container만 중지했다. Docker exit code는 stop에 따른 137이었고 `OOMKilled=false`였으며,
부분 산출물과 격리 checkpoint는 그대로 보존됐다.

n7의 타 작업은 GPU 3/4에 약 2--3분 간격으로 진입·이탈해 5분 gate가 반복 리셋됐다. 외부
process/container에는 손대지 않고, launch 직전 memory/utilization/compute-PID 재검사와 launch 후
15초 foreign-PID 감시는 그대로 유지한 채 안정성 표본만 10회에서 5회로 줄였다. 첫 표본부터
launch까지 2분이며 launcher 원본은
`wait-shard-5271d8e.sh.before-2m-gate-20260915T0414KST`에 보존했다. 수정본은 `bash -n`을
통과했고 SHA-256은 `f9075eaa630e05f5cb211898b445703698a9c474304d81c1874e0e247b294899`다.
변경 당시 recovery worker가 0개임을 확인한 뒤 우리 coordinator process group만 종료하고 PID
3667738로 다시 시작했다. canonical queue/output에는 변경이 없으며 네 미완료 recovery shard는
같은 exact image와 40-core 합계 상한으로 계속 대기한다.

같은 recovery config의 검증된 `load_soft=72`와 달리 임시 launcher만 load1 50을 hard limit으로
사용해 안정 표본이 불필요하게 리셋되는 것도 확인했다. launch 직전 GPU memory/utilization/PID
검사는 유지하고 load gate만 72로 일치시켰으며, 수정 launcher SHA-256은
`6867c7ec104f59fdd4086fd5d47df2ed33c7d994bdf6cd9f29088999eedb7a02`다. 이 gate를 통과한
shard 00002 resume은 7.2초 뒤 `command attempt budget already exhausted`로 종료됐다. Docker
상태는 exit 1, `OOMKilled=false`였고 GPU context 생성 전 Python traceback이 발생했으므로 GPU
충돌, OOM, mount 실패는 배제했다.

직전 shard 00002 worker는 약 7분간 정상 산출물을 늘리다가 foreign GPU PID 감지로 의도적으로
중지됐지만, durable checkpoint에는 `prepare_bundle`의 active attempt 3이 남았다. 다음 resume은
abandoned attempt를 실패로 확정해 `active_attempt=null`, attempts 3으로 저장한 뒤 budget에서
거부했다. shard 00000/00003/00004에도 같은 이유로 active attempt 2가 남아 있어 coordinator를
우리 process group에 한해 중지했다. canonical queue/output과 타 사용자 process/container에는
변경이 없다.

production queue handoff가 이미 사용하는 원칙과 같이, isolated recovery를 의도적으로 중지할
때 active attempt 하나만 환급하고 원본 checkpoint, before/after SHA-256, 사유와 시각을 별도
remediation evidence에 원자적으로 기록하는 `preserve_isolated_handoff_attempt`를 추가했다.
failing-first test는 helper가 없어서 import 단계에서 실패했고, 구현 후 operator handoff tests
3개와 executable `/dev/shm` TMPDIR의 data_toolkit 전체 933개 테스트가 통과했다. Ruff 오류는
0개, basedpyright 오류는 0개이며 기존 strict-mode 경고 35개만 남았다. n7의 임시 overlay image와
canonical checkpoint 복사본을 사용한 실사용 QA에서도 `prepare_bundle 2→1`, active attempt
clear, 원본 backup과 일치하는 remediation hash를 확인했다. 실제 recovery checkpoint 복구와
재개는 새 exact image 및 launcher 통합 후에만 수행한다.

helper commit `8754e9c84b2282aebc77908c0a15903ee6925bb5`와 data_toolkit tree
`c201f1926bda41bfe8e0708c192453a0514dc4c6`를 fork의 production branch에 push하고, 기존
검증 image를 수정하지 않은 채 n7 image `pixal3d-fast:8754e9c`
(`sha256:bf77b70ce1abc76c34753dae61c2be9a7135b036996c4d2f5eed61870e6f57fa`)를 새로
생성했다. image revision/tree label과 helper import를 검증했다. active attempt가 남은 recovery
shard 00000/00003/00004는 각각 `prepare_bundle 2→1`로 환급했다. shard 00002는 거부된 resume이
active attempt를 이미 null로 확정했으므로 현재 checkpoint SHA-256
`4e2b53fa...c1e59ef7`, attempts 3, completed download/stage_raw를 모두 precondition으로 고정한
1회성 복구로 3→2로 환급했다. 네 원본과 before/after hash 및 사유는
`control/remediations/isolated-recovery-handoffs` 아래에 보존했다.

새 worker가 CPU-only mesh/PBR 준비 중에는 CUDA context가 없어 GPU memory/utilization만으로는
이미 예약된 GPU를 식별할 수 없었다. 실제 GPU3 container가 실행 중인데 다음 launcher가
`candidate=3 stable=4/5`까지 선택하는 것을 확인하고 아직 container를 만들지 않은 waiter만
중지했다. running recovery container의 `io.pixal3d.physical-gpu` label을 후보에서 제외하고
launch 직전에도 같은 reservation을 재검사하도록 launcher를 보강했다. 배포 SHA-256은
`9e51542ccf1be98ee7962af9845035d4248c68ca4f89e0c4f3f918735e9c56c2`다. 수정 후 GPU3은
shard 00004에 예약된 채 shard 00003이 GPU4로 시작했고, 다음 waiter는 `candidate=none`을
기록했다.

GPU4에 타 사용자 PID 140675가 들어온 실제 경합에서는 monitor가 우리 shard 00003 container만
중지했다. 이어 GPU3에 PID 169381이 들어왔을 때도 shard 00004만 중지됐다. 두 Docker 상태는
exit 137, `OOMKilled=false`였고 새 helper가 각각 `prepare_bundle 2→1`을 환급해 원본 checkpoint와
remediation hash를 저장했다. 두 coordinator는 exit 75를 받아 즉시 안전 대기로 복귀했다.
shard 00004는 중단 전 mesh 29, PBR 10까지 부분 결과를 늘렸으며 다음 resume에서 이를 재사용한다.
continuation coordinator PID 102221과 선행 coordinator PID 4189627은 reservation-aware launcher로
네 미완료 shard를 자동 재시도한다. 실행 중 recovery CPU quota 합계는 GPU당 10 cores이고 최대
두 GPU가 가용할 때도 20 cores로 44-core 상한 이하다.

## 2026-09-15 — recovery 재개 handoff 및 canonical 승격 사전 확인

새 image worker는 shard 00000을 GPU3, shard 00002를 GPU4에서 재개해 각각 CPU 10 cores와
RAM 40 GiB 상한으로 `dump_mesh`/`dump_pbr` prepare를 진행했다. 2026-09-14 20:11 UTC에
타 사용자 compute PID가 GPU3과 GPU4에 들어오자 ownership monitor가 우리 두 container만
중지했다. 두 container 모두 Docker exit 137, `OOMKilled=false`였고 helper는 shard 00000의
`prepare_bundle` attempt를 2→1, shard 00002를 3→2로 환급한 뒤 remediation evidence를
보존했다. canonical에는 쓰기가 없었고 coordinator는 다시 2분 안정성 gate를 기다린다.

승격 대상은 recovery selection이 고정한 canonical
`ObjaverseXL_sketchfab-00001/batch002`와 `batch005`임을 read-only로 재확인했다. batch002의
기존 checkpoint는 failure 256개, batch005는 completed 191개와 failure 65개다. batch002의
256개 failure는 recovery shard 00000--00003의 64개씩과 정확히 일치한다. batch005의 65개
failure 중 앞 64개는 shard 00004와 일치하며 마지막
`09b4609f7082aac37e7e5a125511e767cdc9787d096ed78e1accb7ad7b7efd44`는 기존 genuine timeout으로
recovery 입력에서 제외돼 있다. 따라서 최종 승격 계약은 batch002 completed 256개,
batch005 completed 255개와 quarantined timeout 1개이며, 모든 격리 pack/raw manifest 감사가
끝난 뒤에만 기존 191개와 recovery 64개를 병합한다.

최신 `pixal3d-fast:8754e9c` image를 사용한 CPU-only `pipeline.cli audit`로 완료 shard 00001의
64개 output, 8개 family pack 및 raw archive를 다시 검사했고 약 74초 후 exit 0을 확인했다.
canonical batch005에서 재사용할 191개 성공 output의 실제 scratch는
`/file3/youngwoo/pixal3d-n7-runtime/local/gpu2/preprocess/active/ObjaverseXL_sketchfab-00001/batch005`
이며 5.6 GiB로 보존돼 있다. 이후 shard 00003/00004가 GPU3/4에서 재개됐으나 타 사용자 compute
PID가 들어와 우리 container만 중지됐다. 두 실행도 exit 137, `OOMKilled=false`였고 각각
`prepare_bundle 2→1`, `active_attempt=None`으로 환급됐다. 재확인한 canonical queue는
completed 153, pending 602, failed 1, running/stale 0이며 active lease가 없다.

## 2026-09-15 — canonical 승격 staging 검증 및 recovery monitor 보강

canonical batch005의 기존 성공 output을 canonical mount 없이 read-only source에서
`/dev/shm/pixal3d-canonical-promotion-n7-20260915/batch005-output-base`로 복사했다. 원본과
staging은 각각 6,743 files, 2,695,153,979 bytes였고 파일별 SHA-256 manifest를 비교해 완전
일치했다. 완료 recovery shard 00001의 8개 pack도 `verify_pack` 후 별도 staging에 안전하게
추출했다. manifest와 추출본의 2,272 members, 941,803,633 bytes 및 모든 파일 SHA-256이
일치했다. 해당 shard의 64개 asset은 recovery selection의 canonical batch002 positions
64--127과 순서까지 같고 checkpoint, quality ledger 및 pack scope도 일치한다.

GPU ownership monitor에는 `nvidia-smi`에서 PID를 읽은 직후 짧은 Blender child가 종료되면
뒤이어 실행한 `docker top`에서 사라져 외부 PID로 오인할 수 있는 sampling race가 있었다.
격리된 PID lifetime test에서 기존 판정이 종료된 PID를 foreign으로 분류하는 것을 재현했고,
처음 unknown이었던 PID는 GPU presence를 다시 측정하고 container PID snapshot도 새로 읽은 뒤
두 번째 검사에서도 GPU에 남아 있고 container에 없을 때만 foreign으로 확정하도록 임시 n7
recovery launcher를 보강했다. `bash -n`과 vanished-PID green test를 통과했으며 launcher
SHA-256은 `723f531da1cb43e7d4ca46da27599a74383cb82652ba0e1dbcd083e315e50b2a`다. 이 변경은
canonical code/output이 아니라 recovery 전용 launcher에만 적용했다.

20:49--20:52 UTC에 GPU3과 GPU4가 2분 안정성 gate를 통과해 shard 00000/00002가 각각
one-visible-GPU container로 재개됐다. 각 container는 CPU 10 cores, RAM 40 GiB, shm 32 GiB로
제한되고 raw dataset과 Blender tools는 read-only다. 21:00 UTC 기준 두 worker 모두 살아 있고
shard 00000은 mesh 64/64, PBR 33/64, shard 00002는 mesh 64/64, PBR 53/64까지 누적했다.
동시 CPU quota는 20 cores로 44-core 상한 이내이며, GPU0/1/2/5의 타 사용자 process와
canonical queue/output은 변경하지 않았다.

## 2026-09-15 — recovery renderer concurrency 복원

실제 recovery worker argv가 `--render_workers_per_gpu 1`로 생성돼, 이미 검증한 GPU당 renderer
3개 정책이 임시 resume helper에서만 비활성화된 것을 확인했다. shard 00000/00002의 실행 중
Blender를 강제로 끊지 않고 Docker의 90초 graceful stop으로 종료했으며 둘 다 `OOMKilled=false`였다.
전용 operator-handoff helper로 active `prepare_bundle` 시도만 각각 2→1, 3→2로 환급하고 원본
checkpoint, before/after SHA-256, 사유와 시각을
`control/remediations/isolated-recovery-handoffs/manual-renderers3-20260914T2108Z-*`에 보존했다.
canonical checkpoint/output에는 쓰지 않았다.

resume helper의 runtime policy만 `(1,)`에서 `(3,)`으로 바꿨다. config hash는
`578e495f060141ff215711d7e5229871c96f38471aac47db5b765a3f3d08d911`로 그대로이고 helper는
Ruff format/check, no-excuse 검사와 `py_compile`을 통과했다. n7 배포본 SHA-256은
`1cf8e4e556564878cdcb54d22476eef37ae0eaf8e4fc5499f67b2f0d410697b6`이며 이전 파일은
`/tmp/pixal3d_quality_recovery_resume_20260913.py.before-renderers3-20260915`에 보존했다.
우리 r3 coordinator process group만 종료·재시작했고 선행 coordinator와 타 사용자 작업은
변경하지 않았다.

2분 GPU 안정성 gate 뒤 선행 coordinator가 shard 00004를 physical GPU3에, r3 coordinator가
shard 00003을 physical GPU4에 시작했다. 두 container 모두 exact image
`pixal3d-fast:8754e9c`, GPU 하나만 노출, CPU 10 cores, RAM 40 GiB, shm 32 GiB이며 raw/tools는
read-only다. 실제 `prepare_bundle` argv는 `--render_workers_per_gpu 3`, 세 자식은 각각
`world_size=3`의 rank 0/1/2로 확인했다. 초기 장기-tail cohort에서 shard 00003은 약 12.4분에
신규 render 3개, shard 00004는 약 14.5분에 신규 render 3개를 원자적으로 게시했다. 이는
약 4.0--4.8분/asset/GPU의 초기 처리율로, 직전 single-renderer 첫 자산 442.62초보다는 빠르지만
일반 100-asset benchmark의 53.55초/asset과 직접 비교할 수 없는 무거운 recovery 자산이다.
준비 단계에는 mesh/PBR Blender와 renderer가 겹쳐 각 container가 약 9.8--10.1 CPU cores로
cgroup quota를 소진했으나 두 worker 합계 quota 20 cores는 44-core 상한 이내였고 host CPU는
샘플 구간에 56--62% idle, I/O wait 0--1%였다. VRAM은 약 0.1--10.3 GiB 범위였으며 남은
shard 00000/00002는 GPU가 비는 동안 예약-aware waiter에서 대기한다.
