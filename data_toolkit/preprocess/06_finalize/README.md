# 06_finalize

04의 성공 asset 중 `stages.encode.exclude_asset_ids`에 없는 asset마다 03 렌더 8장과 transform, 05의 shape/PBR/SS latent 및 scale JSON을 확인한다.
`--publish`를 주면 8개 family의 비압축 PAX tar와 각 manifest를 로컬
`./data/preprocess_finalize`에서 만든다. SHA-256 검사 후 tar에서 학습 샘플을 꺼내
SS·shape·PBR의 실제 Dataset/DataLoader로 7개 학습 설정을 확인한다. 통과하면 NS2
`prepared` 내부 임시 경로에 복사·검증하고 기존 shard index의 batch를 보존하며 새 batch를
마지막에 추가한다. 기존 batch나 unindexed 파일과 충돌하면 덮어쓰지 않고 실패한다.

검증만 하려면 `--publish`를 빼고 실행한다. 이 경우 `06_finalize/report.json`과
`manifest.jsonl`만 NS3 work root에 저장된다.

```bash
/home/rvi/conda/envs/torch/bin/python data_toolkit/preprocess/06_finalize/run.py \
  --config data_toolkit/preprocess/config/default.yaml \
  --world-size 1 --rank 0 \
  --shard ObjaverseXL_sketchfab-00000 --batch batch010 --publish
```

출력 예: `/home/rvi/ns2/youngwoo/pixal3d/prepared/{common,ss,shape,pbr,index}/...`.
이전에 `--publish` 없이 06을 실행한 batch도 위 단독 명령으로 publish할 수 있다.
임시 tar 경로는 YAML `paths.local_temp_root` 또는 `--local-temp-root`로 바꿀 수 있다.
`--publish` 없는 실행은 입력 파일만 확인하며 tar를 만들거나 `prepared`를 수정하지 않는다.
