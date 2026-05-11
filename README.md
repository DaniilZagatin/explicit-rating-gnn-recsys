# Применение графовых нейронных сетей в рекомендательных системах

Репозиторий содержит материалы выпускной квалификационной работы, посвящённой top-K рекомендации фильмов по истории явных пользовательских оценок. Основной акцент сделан на графовых рекомендательных моделях, способах использования численного рейтинга и гибридных архитектурах, объединяющих графовые представления с двубашенными моделями.

## Аннотация

В большинстве базовых постановок рекомендательных систем взаимодействие пользователя с объектом сводится к бинарному сигналу: пользователь либо взаимодействовал с объектом, либо нет. В задачах с явными оценками такая постановка теряет часть информации, поскольку оценки `5`, `3` и `1` отражают разную силу пользовательского предпочтения.

В работе исследуется, как использовать explicit ratings в задаче top-K рекомендации фильмов. Рассматриваются несколько вариантов учёта рейтинга: бинаризация положительного отклика, использование рейтинга как типа ребра в GCMC-подобной модели, использование рейтинга как веса связи в NGCF-подобной модели, а также объединение двубашенной и графовой моделей в гибридном ансамбле.

Эксперименты проводятся на подготовленном датасете на основе MovieLens 25M с дополнительными признаками фильмов. Для оценки используются warm-start, cold-user, cold-item и both-cold сценарии. Качество моделей сравнивается по метрикам Precision@K, Recall@K, HitRate@K, MRR@K, MAP@K и NDCG@K.

## Основная идея

Ключевая гипотеза работы состоит в том, что user-item граф с явными рейтингами содержит полезный структурный сигнал, который может дополнять двубашенную retrieval-модель.

В проекте сравниваются три группы моделей:

1. **Базовые модели**
   - матричная факторизация;
   - двубашенная модель SASRec-вида.

2. **Графовые модели с учётом рейтинга**
   - GCMC-подобная модель, где значение рейтинга задаёт тип ребра;
   - NGCF-подобная модель, где рейтинг влияет на вес распространения сигнала в графе.

3. **Гибридные модели**
   - ансамбль предобученной двубашенной модели и GCMC;
   - вариант с замороженными компонентами;
   - вариант с совместным дообучением;
   - объединение пользовательских и объектных embedding-векторов из разных архитектур.

## Материалы ВКР

Финальные материалы работы рекомендуется хранить в `docs/`:

```text
docs/
├── thesis/
│   ├── vkr_zagatin_daniil.pdf
│   └── source/
│       ├── main.tex
│       └── MachLearn.bib
│
└── presentation/
    ├── defense_presentation.pdf
    └── source/
        └── presentation.tex
```

После добавления файлов основные ссылки будут такими:

- текст ВКР: `docs/thesis/vkr_zagatin_daniil.pdf`;
- презентация к защите: `docs/presentation/defense_presentation.pdf`;
- описание датасета: `docs/dataset_description.md`;
- описание методов: `docs/methods.md`;
- описание экспериментов: `docs/experiments.md`.

## Структура репозитория

```text
explicit-rating-gnn-recsys/
├── architectures/
│   ├── GCMC.png
│   ├── NGCF.png
│   ├── SASRec_like.png
│   ├── message_pass.png
│   └── README.md
│
├── configs/
│   ├── mf.yaml
│   ├── two_tower_sasrec.yaml
│   ├── gcmc_rating_relation.yaml
│   ├── ngcf_rating_weighted.yaml
│   └── ensemble_two_tower_gcmc.yaml
│
├── docs/
│   ├── dataset.png
│   ├── dataset_description.md
│   ├── methods.md
│   ├── experiments.md
│   ├── repository_notes.md
│   ├── thesis/
│   │   └── vkr_zagatin_daniil.pdf
│   └── presentation/
│       └── defense_presentation.pdf
│
├── notebooks/
│   ├── 00_dataset_preparation.ipynb
│   ├── 01_dataset_eda.ipynb
│   ├── 02_baselines_mf_two_tower_gcmc.ipynb
│   ├── 03_ngcf_rating_weights.ipynb
│   ├── 04_hybrid_two_tower_gcmc.ipynb
│   └── notebooks_info.md
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── README.md
│
├── LICENSE
├── README.md
└── requirements.txt
```

Назначение основных папок:

- `architectures/` — схемы моделей и message passing;
- `configs/` — конфигурации запусков основных моделей;
- `docs/` — текстовое описание датасета, методов, экспериментов и материалов ВКР;
- `notebooks/` — воспроизводимые ноутбуки с подготовкой данных, обучением и анализом;
- `results/` — экспортированные таблицы и графики экспериментов.

## Датасет

В экспериментах используется подготовленный датасет на основе MovieLens 25M с дополнительными признаками фильмов. Датасет предназначен для исследования top-K рекомендаций по explicit feedback.

В датасете сохранены:

- взаимодействия пользователей и фильмов;
- исходные численные рейтинги;
- временное разбиение на train, validation и test;
- warm-start, cold-user, cold-item и both-cold сценарии;
- признаки фильмов;
- метаданные пользователей и фильмов;
- mapping-файлы для индексов моделей.

Рейтинги сохраняются в исходном численном виде и не бинаризуются на этапе подготовки данных. Это позволяет в разных экспериментах задавать собственную интерпретацию рейтинга.

Ожидаемый локальный путь к датасету:

```text
data/movielens20-withfeatures-split/
```

В Kaggle Notebook датасет может быть подключён по пути:

```text
/kaggle/input/datasets/daniilzagatin/movielens20-withfeatures-split
```

Основные interaction-файлы:

```text
train_warm_interactions.parquet
warm_val_interactions.parquet
warm_test_interactions.parquet
cold_user_support.parquet
cold_user_support_all.parquet
cold_user_val_interactions.parquet
cold_user_test_interactions.parquet
cold_item_val_interactions.parquet
cold_item_test_interactions.parquet
both_cold_val_interactions.parquet
both_cold_test_interactions.parquet
```

Файлы с признаками и метаданными:

```text
item_features_all.parquet
item_features_warm.parquet
user_meta.parquet
item_meta.parquet
feature_cols.json
split_config.json
```

Raw parquet-файлы датасета не хранятся в репозитории из-за размера.

## Ноутбуки

Основной код экспериментов находится в папке `notebooks/`.

| Ноутбук | Назначение |
|---|---|
| `00_dataset_preparation.ipynb` | Подготовка датасета, построение признаков и temporal split |
| `01_dataset_eda.ipynb` | Анализ распределений, статистик и warm/cold-сценариев |
| `02_baselines_mf_two_tower_gcmc.ipynb` | Базовые модели: MF, Two-Tower, GCMC |
| `03_ngcf_rating_weights.ipynb` | NGCF-подобная модель и функции учёта рейтинга |
| `04_hybrid_two_tower_gcmc.ipynb` | Гибридные ансамбли Two-Tower + GCMC |
| `notebooks_info.md` | Краткое описание ноутбуков |

Рекомендуемый порядок запуска соответствует нумерации ноутбуков.

## Модели

В проекте реализуются и сравниваются следующие подходы.

| Модель | Описание |
|---|---|
| Matrix Factorization | Базовая collaborative filtering модель |
| Two-Tower SASRec-like | Retrieval-модель, где представление пользователя строится по истории взаимодействий |
| GCMC / Rating-as-Relation | Графовая модель, где значение рейтинга задаёт тип ребра |
| NGCF-like Rating-Weighted Model | Графовая модель, где рейтинг влияет на вес распространения сигнала |
| Two-Tower + GCMC Ensemble | Гибридная модель, объединяющая двубашенные и графовые представления |
| Fine-tuned Hybrid Ensemble | Совместно дообучаемый ансамбль предобученных компонентов |

## Использование явных рейтингов

В работе рассматриваются несколько способов использования рейтинга.

### Бинарная релевантность

Фильм считается релевантным, если:

```text
rating >= 4
```

Этот вариант используется для стандартных top-K метрик.

### Рейтинг как численный сигнал для NDCG@K

Рейтинг также используется как численный сигнал при расчёте NDCG@K. В экспериментах рассматриваются схемы, где gain строится на основе отклонения рейтинга от нейтрального значения или через нелинейное преобразование рейтинга.

Примеры схем:

```text
centered_3: gain зависит от rating - 3
power:      нелинейное преобразование рейтинга
```

### Рейтинг как тип ребра

В GCMC-подобной модели взаимодействие задаётся как тройка:

```text
(user, item, rating)
```

Например, оценки `5` и `2` соответствуют разным типам отношений в user-item графе.

### Рейтинг как вес ребра

В NGCF-подобной модели рейтинг управляет силой распространения сигнала между пользователем и фильмом. Высокая оценка усиливает связь, а низкая оценка ослабляет её вклад.

## Оценка качества

Модели оцениваются в задаче top-K рекомендации.

Используемые метрики:

```text
Precision@K
Recall@K
HitRate@K
MRR@K
MAP@K
NDCG@K
```

Сценарии оценки:

| Сценарий | Описание |
|---|---|
| Warm-start | Пользователи и фильмы присутствуют в обучающем графе |
| Cold-user | Новый пользователь представлен через support-историю |
| Cold-item | Новый фильм представлен через признаки объекта |
| Both-cold | Одновременно новый пользователь и новый фильм |

## Результаты

Итоговые таблицы и графики сохраняются в папке `results/`:

```text
results/
├── tables/
└── figures/
```

Рекомендуемые итоговые таблицы:

```text
results/tables/warm_results.csv
results/tables/cold_user_results.csv
results/tables/cold_item_results.csv
results/tables/both_cold_results.csv
results/tables/final_comparison.csv
```

Рекомендуемые итоговые графики:

```text
results/figures/final_quality_comparison.png
results/figures/warm_cold_comparison.png
results/figures/validation_curves.png
```

В итоговом сравнении анализируются:

- качество отдельных baseline-моделей;
- влияние разных способов учёта рейтинга;
- качество графовых моделей;
- качество гибридных graph-retrieval ансамблей;
- различия между warm-start и cold-start сценариями.

## Установка

Клонировать репозиторий:

```bash
git clone https://github.com/DaniilZagatin/explicit-rating-gnn-recsys.git
cd explicit-rating-gnn-recsys
```

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать окружение:

```bash
# Linux / macOS
source .venv/bin/activate
```

```bash
# Windows
.venv\Scripts\activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

## Воспроизводимость

Репозиторий организован так, чтобы эксперименты можно было воспроизвести через ноутбуки и конфигурации:

- подготовка датасета описана в `notebooks/00_dataset_preparation.ipynb`;
- EDA вынесен в `notebooks/01_dataset_eda.ipynb`;
- настройки моделей хранятся в `configs/`;
- архитектурные схемы хранятся в `architectures/`;
- итоговые таблицы и графики сохраняются в `results/`;
- описание экспериментов хранится в `docs/experiments.md`.

Для полного воспроизведения необходимо использовать тот же датасет, те же split-файлы и соответствующие конфигурации моделей.

## Ограничения

Проект является исследовательским прототипом, а не production-ready рекомендательной системой.

Основные ограничения:

- эксперименты проводятся на датасете на основе MovieLens;
- большие checkpoint-файлы не хранятся в репозитории;
- raw parquet-файлы датасета не входят в репозиторий из-за размера;
- cold-user и cold-item сценарии требуют отдельной поддержки со стороны модели;
- итоговое качество зависит от подбора гиперпараметров и доступных вычислительных ресурсов.

## Краткое описание для резюме

**Рекомендательная система фильмов с учётом явных рейтингов.**

Разработан исследовательский pipeline для top-K рекомендации фильмов с explicit feedback. Реализованы и сравнены матричная факторизация, двубашенная модель SASRec-вида, GCMC-подобная графовая модель, NGCF-подобная модель с rating-aware propagation, а также гибридные graph-retrieval ансамбли. Модели оценивались в warm-start, cold-user и cold-item сценариях по метрикам Precision@K, Recall@K, HitRate@K, MRR@K, MAP@K и NDCG@K.

## Стек технологий

```text
Python
PyTorch
PyTorch Geometric
Pandas
NumPy
Scikit-learn
Jupyter Notebook
Parquet
Graph Neural Networks
Recommender Systems
Ranking Metrics
```

## License

Код и материалы репозитория распространяются согласно лицензии, указанной в файле `LICENSE`.
