# 06_finalize

05 latent tree를 검증하고 `--publish`일 때 `prepared-v2`로 atomic publish한다. publish된 tree에는
`completion.json` marker가 포함되며 이후 어느 rank/node에서 실행해도 이 batch는 skip된다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/06_finalize/run.py \
  --world-size 1 --rank 0 \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010 --publish
```

`prepared-v2/<source>/<shard>/<batch>`는 완료 marker가 없으면 완료로 취급하지 않는다.
