# Заметки по репозиторию

Репозиторий оформлен как исследовательский проект. Основная реализация пока сосредоточена в ноутбуках, а дополнительные папки используются для конфигураций, результатов и краткой документации.

## Основные ноутбуки

```text
notebooks/
├── 00_dataset_preparation.ipynb
├── 01_dataset_eda.ipynb
├── 02_baselines_mf_two_tower_gcmc.ipynb
├── 03_ngcf_rating_weights.ipynb
└── 04_hybrid_two_tower_gcmc.ipynb
```

## Что не хранится в репозитории

- сырые данные;
- большие parquet-файлы;
- checkpoint-файлы моделей;
- токены Kaggle, Comet, WandB и другие секреты;
- временные notebook outputs.
