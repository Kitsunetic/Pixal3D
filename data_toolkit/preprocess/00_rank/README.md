# 00_rank

각 process는 `--world-size`, `--rank`를 반드시 받아 control batch 전체의 안정된 정렬 index로
자기 소유 batch만 처리한다.

```text
batch_index % world_size == rank
```

기본 stage 목록은 `01,02,03,04,05,06`이다. 각 stage script는 한 batch씩 종료하고, 이 launcher는
같은 batch에 대해 그 script들을 순서대로 호출할 뿐 queue/scheduler를 만들지 않는다. 모든 중간 파일은
NS3 shared `work_root`에 남는다.

```bash
CUDA_VISIBLE_DEVICES=5 /home/rvi/conda/envs/torch/bin/python \
  data_toolkit/preprocess/00_rank/run.py \
  --world-size 16 --rank 3 --publish
```

CPU-only 01/02만 수행하려면 다음처럼 명시한다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/00_rank/run.py \
  --world-size 16 --rank 3 --stages 01,02
```

동일한 `(world_size, rank)`는 한 번만 실행해야 한다. world size를 바꾼 재실행은 legacy `prepared`와
NS2 prepared index에 완료된 batch를 자동으로 건너뛰며, shared work root의 완료된
stage 결과를 재사용한다.
