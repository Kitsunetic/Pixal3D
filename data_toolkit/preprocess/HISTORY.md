# Pixal3D 전처리 실행 이력

이 문서는 Pixal3D v2 전처리를 수행하며 대화에서 확정한 서버·GPU 배치를
기록한다. 시점 기준은 **2026-09-22**이다. 운영 상태의 최종 근거는 각
batch의 `stage.json`과 실행 로그이며, 이 문서는 재시작·재배치 판단을 위한
사람 친화적인 이력이다.

## 공통 규칙

- `n7jh`는 local 작업 환경이다.
- GPU stage는 `CUDA_VISIBLE_DEVICES`로 GPU 하나만 노출하고, process 하나가
  그 논리 GPU 하나를 사용한다.
- CPU stage는 GPU를 사용하지 않는다. `n5`, `n8`, `n10`은 fleet에서 제외했고,
  `n12`도 사용하지 않았다.
- batch 소유권은 해당 실행의 `batch_index % world_size == rank`다. world size가
  달라진 실행은 이미 atomic publish된 asset/stage 결과를 skip하며 재개했다.
- 중간 결과와 최종 결과는 NS3 shared root를 사용했다. 05의 asset별 latent는
  atomic 저장이므로 중단된 batch도 이미 저장된 asset은 재실행 시 skip된다.

## 01_manifest

| 구분 | 서버 / GPU | 실행 내용 |
| --- | --- | --- |
| 초기 확인 | n7 local / GPU 미사용 | raw ObjaverseXL Sketchfab manifest와 batch index를 생성·검증했다. |
| fleet 실행 | n1, n2, n3, n4, n6, n7, n9, n11, n13, n14, n15, n16, n17 / GPU 미사용 | CPU-only manifest 실행. 한 시점에는 위 12개 노드에 각 8 rank를 배치해 world size 96으로 운용했고, n7 local rank를 별도 병행했다. |

`01_manifest`는 CUDA가 필요 없도록 유지했다. 기존 prepared 완료 batch와 control
batch index를 기준으로 skip한다.

## 02_dump

| 구분 | 서버 / GPU | 실행 내용 |
| --- | --- | --- |
| raw 접근 방식 변경 | n7 local / GPU 미사용 | Sketchfab GLB를 7z에서 개별 추출하는 방식 대신, n7에서 NS3의 Sketchfab raw GLB를 미리 압축 해제한 뒤 direct GLB를 읽도록 전환했다. |
| fleet 실행 | n1–n18 및 n7 local / GPU 미사용 | CPU-only dump를 modulo world-size partition으로 실행했다. node availability와 남은 rank에 따라 재분배했다. |

02는 embedded `bpy`/native import와 PBR·mesh dump를 수행하지만 CUDA를 사용하지
않는다. 7z를 asset마다 재읽는 경로는 제거했다.

## 03_render

03은 OptiX renderer GPU stage이며, canonical `world_size=32` 배치는 아래와
같다. 이 표의 GPU 번호는 각 서버의 물리 GPU 번호이며 실행 시 해당 번호만
`CUDA_VISIBLE_DEVICES`로 노출했다.

| 서버 | rank | GPU | 비고 |
| --- | --- | --- | --- |
| n1 | 0–4 | 5 | renderer worker 5개 |
| n7 local | 5–9 | 5 | renderer worker 5개 |
| n14 | 10–14 | 5 | renderer worker 5개 |
| n15 | 15–19 | 5 | renderer worker 5개 |
| n16 | 20–25 | 6 | renderer worker 6개 |
| w20 | 26–28 | 3 | renderer worker 3개 |
| w24 | 29–31 | 3 | renderer worker 3개 |

n11은 03 renderer fleet에 넣지 않았다. 각 서버는 GPU 하나만 사용했고,
서버 내 renderer process가 그 GPU를 공유했다.

## 04_voxelize

| 구분 | 서버 / GPU | 실행 내용 |
| --- | --- | --- |
| fleet 실행·재분배 | n1–n18, w20, w24 / GPU 미사용 | mesh/PBR dump와 render 결과를 dual-grid/PBR voxel VXZ로 생성했다. CPU 및 native voxel backend만 사용했다. |
| canonical 재개 | 동일 / GPU 미사용 | 메모리·통신 상황을 반영해 `world_size=166` 실행을 기준으로 중지·재개 및 남은 rank 재분배를 수행했다. |

중간에 더 큰 worker 배치를 실험했지만, 운영 기준은 W166 재개와 이후의 잔여
rank 재분배다. 이 단계의 GPU VRAM은 전처리 점유가 아니다.

## 05_encode

05는 shape → SS → PBR latent encoder GPU stage다. 1024 shape의 raw sparse voxel은
매우 큰 VRAM peak를 만들지만, 저장되는 latent token과는 다른 크기다.

| 시기 / 상태 | 서버 | GPU / rank | world size | 결과 |
| --- | --- | --- | --- | --- |
| 초기 local probe | n7 local | GPU 0–5 / rank 0–5 | 6 | resolution별 micro-batch를 조정하며 실행했다. |
| A6000 probe (중지) | n14 | GPU 0–7 / rank 0–7 | 8 | 48 GiB A6000에서 일부 1024 asset OOM 및 non-finite output을 관측해 전체 중지했다. |
| 기존 n17 fleet (중지) | n17 | GPU 0–7 / rank 0–7 | 8 | Blackwell 98 GiB GPU에서 실행했으나 input 구조 refactor를 위해 중지했다. Partial latent은 보존됐다. |
| DataLoader GPU0 probe | n17 | GPU 0 / rank 0 | 8 | `23649a1` 이후 Dataset/DataLoader input과 saver queue/thread를 검증했다. |
| 4-GPU DataLoader probe (중지) | n17 | GPU 0–3 / rank 0–3 | 8 | NFS VXZ 동시 read가 과도해지는 것으로 판단해 중지했다. 원자 publish된 partial latent은 보존됐다. |
| GPU7 single-asset probe (완료) | n17 | GPU 7 | 1 | 3,009,356 voxel asset을 `loader_workers=16`으로 shape/SS/PBR 전 family·두 view 완료했다. |
| GPU7 large-asset probe (완료) | n17 | GPU 7 | 1 | 7,919,826 voxel asset을 같은 설정으로 완료했다. 관측 시점 VRAM 14.5 / 95.6 GiB, GPU utilization 95–98%였다. |
| 다음 실행 구성 | n17 | GPU 7 / rank 0 | 1 | 1024 shape voxel 수가 큰 asset만 담당하며 `loader_workers=16`으로 실행한다. |
| 다음 실행 구성 | n7 local | GPU 0–5 / rank 0–5 | 6 | n17 범위와 겹치지 않는 VRAM-safe asset만 담당한다. |

다음 n17 05 실행 세부:

- CUDA: `CUDA_VISIBLE_DEVICES=7` 하나만 노출한다.
- input: PyTorch `Dataset/DataLoader(num_workers=16, pin_memory=True)`
- output: bounded saver `Queue` + 2 saver thread
- scheduling: 큰/작은 asset을 1024 shape VXZ header의 최대 `num_voxel` 범위로 disjoint하게 나눈다. asset latent output은 asset/view/resolution별 atomic publish다.

## 06_finalize

2026-09-23에 `ObjaverseXL_sketchfab-00000/batch010` 한 배치의 8개 tar를
NS2 `prepared-v2`에 시험 게시하여 checksum과 실제 학습 Dataset 로드를 확인했다.
이후 06은 로컬 임시 tar 생성·학습 loader 검사 뒤 기존 NS2 `prepared`에 합치는
방식으로 변경했다. 새 방식의 production `prepared` 게시 또는 fleet 실행은 아직 하지 않았다.

## 재현 시 확인 순서

1. 대상 `nXjh`에서 `git pull --ff-only` 후 commit을 확인한다.
2. GPU stage는 `CUDA_VISIBLE_DEVICES=<physical GPU>`로 GPU 하나만 노출한다.
3. 실행 중인 tmux session, 각 rank log, 해당 batch의 `stage.json`을 확인한다.
4. 이 문서의 historical GPU 배치보다 현재 GPU 점유·VRAM과 완료 marker를 우선한다.
