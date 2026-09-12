# Pixal3D ObjaverseXL production-path qualification

## 결론

최적화된 production pipeline의 ObjaverseXL 경로는 실제 Objaverse GLB 100개를 대상으로 download부터
raw archive/cleanup까지 전체 11개 command graph를 완료했다. 100개 모두 quality
ledger에서 `completed`이며 quarantine은 0이다. 기존 canonical queue와 output은 사용하지
않았고, 검증 전용 input/output/scratch와 GPU별 container를 사용했다.

이 결과는 ObjaverseXL production 경로에 한정된다. ABO, HSSD, 3D-FUTURE는 native
renderer를 사용하지 않고 기존 external Blender 경로로 자동 전환되며, 이번 100개
E2E 범위에는 포함되지 않았다.

검증한 코드 snapshot은 `a9114d82588e019e529204affdc97f08d46ca47f`이다. 검증 중
발견한 worker 종료 후 reap race는 후속 commit `756f5d1`에서 수정했으며, 수정 후 전체
test suite는 892 passed였다. 이후 NFS fallback, descriptor-pinned Objaverse input과
reproducible comparator까지 포함한 최종 suite는 900 passed였다. 이 보강은 정상 산출물
계산 알고리즘을 바꾸지 않는다.
최종 suite가 검증한 runtime commit은
`87b58b49a07f91d0d7c069b11044a5881fade64f`이고 `data_toolkit` runtime tree는
`b7b315125600bbfa4d134a6f6e2987f12e9d6103`이다. 저해상도 fitting과 최초 target
resolution 검증이 불일치하면 원래 radius와 10회 budget으로 legacy full-resolution
loop를 완전히 재실행한다. 이 동작은 source-string 검사가 아니라 replay 조건과 초기화
상태를 실행하는 regression test로 고정했다. comparator는 입력 root가 없거나 비교 가능한
artifact가 0개이면 실패하도록 fail-closed로 보강했다. JSON/NPZ의 non-finite 값과 경로에
사용되는 비정상 asset SHA-256도 거부하며, Objaverse input은 실제 processor에 전달하기
직전 열린 descriptor의 digest를 다시 검증한다.

## 성능 결과

동일 100개 asset의 깨끗한 renderer benchmark와 동일 10개 asset의 combined renderer-path
benchmark 결과는 다음과 같다.

| 범위 | 기준 | 최적화 | 향상 |
|---|---:|---:|---:|
| 10 assets, external Blender + full-resolution Cycles fitting | 898초 (89.8초/asset) | 184초 (18.4초/asset) | 4.88배 |
| 100 assets, 6 GPU, native full-resolution OPTIX | 765초 (7.65초/asset) | 424초 (4.24초/asset) | 1.80배 |
| 100 assets, final Cycles backend | OPTIX 424초 | CUDA 357초 | CUDA 15.8% 단축 |
| latent 전체, 6 GPU | 587초 후 OOM, 5개는 별도 49초 재개 | 177초, OOM 없음 | cache/restart 조건이 다른 진단값 |

각 clean renderer run의 topology, backend, boundary-fit 설정, rank별 wall time, exit code와
원본 log SHA-256은 `benchmark-timings.json`에 기록했다. 100개 수치는 여섯 rank 중 가장
늦게 끝난 wall time이며, 서로 동일한 100-asset manifest를 사용했다.

production 기본 backend는 출력 연속성을 위해 OPTIX로 유지한다. CUDA는 더 빠르지만
800개 frame 중 alpha byte-exact 439개, binary mask exact 766개, 최저 mask IoU
0.99520이어서 opt-in 후보로만 남긴다.

이번 100-asset E2E wall time은 성능 기준으로 사용하지 않는다. 중간에 다른 사용자가
GPU 0--3에서 학습을 시작했고, 해당 작업을 보호하기 위해 세 shard를 중지한 뒤 빈 GPU에서
재개했기 때문이다. 이 실행은 완전성·재개·메모리 안정성 검증이며 위 표의 오염되지 않은
실행을 속도 기준으로 사용한다.

## 출력 동등성

frozen full-resolution OPTIX 기준과 최적화 renderer의 100 assets/800 views 비교 결과:

- asset/file set, retry count, camera matrix, radius, angle, AABB, scale, offset: exact
- alpha: 800/800 byte-exact, 최저 IoU 1.0
- raw RGB PSNR min/p50/p95/p99: 24.81/44.85/55.09/60.25 dB

exact production snapshot으로 다시 생성한 800 views도 모든 구조·camera·alpha가 exact였다.
이 독립 실행의 raw RGB PSNR min/p50/p95/p99는 11.89/24.33/49.50/59.55 dB였다.
기존 renderer가 lighting seed를 고정하지 않으므로 독립 production run 사이의 RGB는
원래 deterministic하지 않다. 따라서 production hard gate는 exact camera/alpha와 artifact
schema/membership이다. 동일 seed를 주는 별도 evaluation에서는 view별 RGB PSNR 50 dB를
요구한다. latent acceptance는 exact schema/shape/dtype/coordinate와 family별 relative L2
0.2% 이하이다.

geometry 재사용 변경은 2,220개 VXZ/scale 파일이 모두 byte-exact였다. reference/candidate
hash manifest는 각각 동일한 SHA-256을 가지며 파일 수와 digest는
`geometry-hash-summary.json`에 고정했다. latent는 2,620개
파일에서 schema, shape, dtype, sparse coordinate가 exact였고, 최대 relative L2 오차는
shape 0.0420%, SS 0.0769%, PBR 0.1417%였다.

## 100-asset E2E 완전성

| 검사 | 결과 |
|---|---:|
| 입력 GLB SHA-256 | 100/100 일치 |
| quality ledger | completed 100, quarantine 0 |
| checkpoint | 6/6 shard, 11/11 commands 완료, active attempt 0 |
| prepared pack | 48 tar + 48 manifest |
| raw archive | 6 tar + 6 manifest |
| manifest provenance | 54/54 exact tool commit/config hash |
| resume | 중단된 3 shard 완료; 완료 shard 재실행 10.10초 no-op |
| audit | 6/6 shard exit 0 |

15개 asset은 지원되지 않는 Blender shader 때문에 PBR-256/512/1024 family에서만 기존
정책대로 제외됐다. shape, SS, common, raw family에는 100개가 모두 포함됐고 PBR 각 family는
85개를 포함한다. asset 전체 실패나 quarantine은 없다.

qualification config hash는
`19184d5a1c0d2a6d817f9f44a8d35af392caaa8dd7e4cbff1d585c0685cb11d7`이다.
canonical production config hash
`ff7dc7073940b869b001b8c8e60525390fe15091f518887814ddc1f3ca35e988`와는 격리된 별도
queue이므로 canonical state를 변경하지 않았다. 최종 canonical queue는 completed 152,
pending 604, running 0, failed 0이다.

## 자원 및 운영 조건

telemetry에서 GPU utilization 최대/p95는 모두 100%였고 latent 구간 VRAM은 최대
24,071 MiB였다. 이는 utilization 0% 상태의 유휴 메모리가 아니라 encoder가 실제 연산에
사용한 메모리다. process RSS 합은 shared page를 중복 계산하므로 container cgroup memory를
기준으로 봐야 하며, 관측 최대는 약 17.75 GiB였다.

native `bpy==4.5.1` renderer는 worker당 8 assets 후 재시작하고 hard timeout을 적용한다.
한 container에 GPU를 모두 노출하면 EEVEE graphics context가 GPU 0에 집중될 수 있으므로,
production에서는 GPU 하나만 노출한 container를 GPU당 하나씩 실행해야 한다. 각 container
내부에서는 보이는 GPU가 ordinal 0이다. input dataset은 read-only, scratch와 output은
container별로 분리하고, 검증 뒤에만 canonical output을 연결한다.

최종 overlay Dockerfile은 n7에서 실제 build했다. 기존 image의 `bpy 4.4.0`은 pinned
`bpy 4.5.1`로 교체됐고, GPU 하나를 노출한 임시 container에서 exact commit/runtime-tree marker,
Blender 4.5.1, Torch 2.11.0+cu128, RTX 4090 인식을 확인했다. build 단계 자체도 Git commit,
`data_toolkit` tree와 clean source를 검증하고, `bpy` wheel은 SHA-256 고정 설치한다. 결과와 base image ID는
`overlay-image-smoke.json`에 기록했으며 임시 container/image/build root는 제거했다.

## 운영 threat model과 장애 복구

신뢰 경계 안에는 고정된 container image/runtime, worker가 소유한 scratch/output, read-only
dataset mount가 있다. archive member 경로, asset metadata/content는 digest와 경로 검사를
통과하기 전까지 신뢰하지 않는다. 같은 Unix uid 또는 host root가 의도적으로 파일을 바꾸는
공격과 손상된 kernel/filesystem은 범위 밖이다.

Objaverse input은 `openat(..., O_NOFOLLOW)`로 연 뒤 검증된 descriptor를 Blender child까지
상속하므로 검증 직후 경로가 symlink로 바뀌어도 다른 inode를 읽지 않는다. archive member는
절대 경로와 traversal을 거부한다. render publication은 asset별 `flock`으로 직렬화한다.
NFS가 atomic exchange를 지원하지 않으면 기존 output을 `.previous`에 보존하고 새 output을
게시한다. 게시와 rollback이 모두 실패해도 `.previous`를 삭제하지 않으며, 다음 resume/filter가
잠금을 획득한 뒤 final이 없을 때 이를 복원하고 검증한다. final과 previous가 함께 있으면 final
검증 성공 후에만 previous를 정리한다.

## 재현 가능한 품질 검사

검증 manifest, qualification config, 비교 report 및 test log는 이 문서 옆의
`evidence/2026-09-13/`에 고정했다. output comparator는 다음처럼 실행한다.
6개 checkpoint와 quality ledger, 54개 publication manifest 원문도
`qualification-raw/`에 포함했으며, 전체 tree digest와 재검산 결과는
`qualification-raw-summary.json`에 기록했다.

```bash
python -m data_toolkit.benchmark_quality REFERENCE_ROOT CANDIDATE_ROOT \
  --rgb-policy diagnostic

# 두 render를 모두 --render_seed 444로 생성한 evaluation에서는
python -m data_toolkit.benchmark_quality REFERENCE_ROOT CANDIDATE_ROOT \
  --rgb-policy required --min-rgb-psnr 50
```

기본 production run은 unseeded이므로 첫 명령이 camera/alpha, artifact set, exact sparse
coordinates와 latent relative-L2 threshold를 hard gate로 검사하고 RGB 분포는 report만 한다.
실제 frozen output에 이 command를 다시 실행한 결과 render 800 PNG는 failure 0,
alpha IoU 1.0으로 통과했고, latent 1,310 NPZ도 failure 0, 최대 relative L2
0.1417104%로 0.2% 기준을 통과했다. exact command와 machine-readable 결과는
`quality-comparator-result.json`과 `latent-comparator-result.json`에 있다.
빈 root, 누락된 root, non-finite reference/candidate가 성공으로 판정되지 않는 regression도
최종 911-test suite에 포함한다.

## 보존된 증거

n7의 검증 root:

```text
/file3/youngwoo/pixal3d-e2e-qualification-final-a9114d8
```

주요 report:

```text
results/reference-vs-optimized-render.json
results/reference-vs-final-render.json
results/resource-telemetry.csv
data2/control/qualification/smoke/{quality,checkpoints}/
data2/prepared/**/batch000.tar.manifest.json
data3/archive/**/batch000.tar.manifest.json
```

저장소에 포함한 evidence index와 frozen input/config는 다음 경로에 있다.

```text
docs/benchmarks/evidence/2026-09-13/evidence-index.json
docs/benchmarks/evidence/2026-09-13/objaverse-assets.txt
docs/benchmarks/evidence/2026-09-13/qualification.yaml
```

검증용 container 9개는 완료 후 모두 제거했다. `youngwoo_diyscene_fast_n7`은 worker 없이
대기 중이고, coworker container와 canonical 입출력은 수정하지 않았다.
