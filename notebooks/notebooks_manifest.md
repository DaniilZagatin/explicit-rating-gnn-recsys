# Систематизация ноутбуков

Основная линия ноутбуков для GitHub:

| Файл | Роль |
|---|---|
| `00_dataset_preparation.ipynb` | Подготовка датасета, генерация дополнительных признаков, сплит по времени |
| `01_dataset_eda.ipynb` | EDA подготовленного датасета |
| `02_baselines_mf_gcmc.ipynb` | Базовые модели: MF, Two-Tower scaffold, GCMC / rating-as-relation |
| `03_ngcf_rating_weights.ipynb` | NGCF-подобная модель: рейтинг как вес ребра |
| `04_hybrid_two_tower_gcmc.ipynb` | Гибридные модели: Two-Tower + GCMC |

Что не стоит выносить в основной список:
- `experiments-v2` — ранний вариант baseline-ноутбука;
- `notebookdd8941a5d5` — черновой вариант гибридных моделей;
- старые версии с промежуточными patch-ячейками лучше оставить локально или положить в `notebooks/archive/`, если нужно сохранить историю.

Для публичного GitHub лучше показывать только одну понятную экспериментальную ветку:
данные → EDA → baselines → NGCF → hybrid ensemble.
