# Tokenized Dataset
## Description
- `train`

    전체 90% 데이터 포함. 학습에 사용할 데이터

- `train_baseline`

    `train` 데이터 중 baseline 모델 학습에 사용할 데이터. `train` 데이터 내의 50%, 전체 데이터의 45% 비중을 차지하고 있음.

- `train_distil`
    
    `train` 데이터 중 distil될 모델에 사용할 데이터. \
    구성)
    - 50% (전체 데이터의 22.5$): `train` 데이터 중 `train_baseline`에 포함된 데이터 
    - 50% (전체 데이터의 22.5$): `train_baseline`에 포함되지 않은 데이터

- `valid`

    validation에 사용할 데이터. 전체 데이터 중 5% 비중을 차지하고 있음. 모델 학습 중, 학습이 완료된 후 간단한 검증이 필요할 때 사용

- `test`

    최종 테스트 데이터

## Python Output
```python
DatasetDict({
    train: Dataset({
        features: ['__index_level_0__', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 1016426
    })
    train_baseline: Dataset({
        features: ['__index_level_0__', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 508213
    })
    train_distil: Dataset({
        features: ['__index_level_0__', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 508212
    })
    valid: Dataset({
        features: ['__index_level_0__', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 56468
    })
    test: Dataset({
        features: ['__index_level_0__', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 56469
    })
})
```
