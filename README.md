# IonoAutoML

## Current foF2 24h results and next run

The completed fast experiment is stored locally in:

```text
artifacts/fof2_fast_ml_24h
```

This artifact directory contains the full per-station outputs and is intentionally ignored by git because the unpacked results are large. A lightweight tracked summary is stored in:

```text
reports/fof2_fast_ml_24h_results_summary.md
reports/fof2_fast_ml_24h_model_quality.csv
reports/fof2_fast_ml_24h_latitude_zone_quality.csv
reports/fof2_fast_ml_24h_station_model_quality.csv
reports/fof2_fast_ml_24h_station_metadata.csv
reports/fof2_fast_ml_24h_station_status.csv
reports/fof2_fast_ml_24h_best_model_counts.csv
```

Current completed result status:

- 41 station folders are present in the result artifact.
- 35 stations have final metrics.
- `TV51R` has partial metrics and can be completed later.
- `BC840`, `EA036`, `JB57N`, `TR170`, and `WP937` currently have no usable final metrics in this artifact.
- The fast run uses `foF2`, a `24h` forecast horizon, 15-minute features, daily walk-forward testing over 2025, and training windows of `7`, `21`, and `28` days.
- The fast run includes `LinearRegression`, `ElasticNet`, `RandomForest`, `XGBoost`, and `CatBoost`.
- The fast run used fixed model parameters; Optuna/AutoML was not enabled for those existing results.

The next full experiment configuration is:

```text
configs/experiments/fof2_fast_ml_24h_automl_shap.json
```

It writes to a separate output directory so the current results are not overwritten:

```text
artifacts/fof2_fast_ml_24h_automl_shap
```

The new run enables:

- Optuna hyperparameter tuning for `ElasticNet`, `RandomForest`, `XGBoost`, and `CatBoost`;
- saved fitted models;
- SHAP summaries for all configured models and all training windows;
- exported metrics, predictions, Optuna trials, best parameters, feature importance, and SHAP summaries.

Run the new experiment on Windows with:

```powershell
.\run_fof2_automl_shap.ps1
```

Dry-run check:

```powershell
.\.venv\Scripts\python.exe .\run_ml_baselines.py `
  --config .\configs\experiments\fof2_fast_ml_24h_automl_shap.json `
  --station EB040 `
  --dry-run
```

Рабочий проект для подготовки парсера, проверки источников данных и отладки структуры хранения.

Сейчас разработка ведется здесь, в `IonoAutoML`. Подготовленные компоненты, конфигурации, результаты проверок и документация в дальнейшем пойдут в основной проект **ИОНОСКОП**.


Здесь настраивается сбор и первичная нормализация данных из GIRO DIDBase/FastChar, GFZ Kp API и NASA OMNIWeb.

README фиксирует текущее состояние рабочего парсера: что уже подключено, как запускать сбор, какие файлы создаются и какие ограничения источников уже учтены.

## Что делает парсер

`collect_hf_data.py` собирает несколько групп данных.

- GIRO DIDBase/FastChar: ионосферные параметры по станциям.
- GFZ Kp API: исторические индексы `Kp`, `ap`, `Ap`, `Fobs`, `Fadj`.
- NASA OMNIWeb: `Bz`, `Dst`, скорость солнечного ветра и плотность протонов.
- NOAA/SWPC: оставлен как дополнительный источник live-продуктов, но для исторических индексов основной упор сейчас на GFZ и OMNI.

Парсер сохраняет сырые ответы источников, нормализованные таблицы, список станций, журнал загрузки и манифест запуска.

## Текущее состояние

Уже добавлено:

- автоматическое получение списка GIRO-станций из формы DIDBase/FastChar;
- запрос всех доступных FastChar-признаков GIRO;
- основной рабочий период в конфиге: `2024-01-01T00:00:00Z` - `2026-01-01T00:00:00Z`;
- нарезка GIRO-запросов на окна `chunk_days`;
- защита от запроса GIRO через границу календарного года;
- сохранение сырых GIRO-файлов по станциям и годам;
- отдельный сбор GFZ-индексов сразу за заданный период;
- отдельный сбор OMNI-данных сразу за заданный период;
- выбор источников через параметр `--sources`;
- фильтрация периода по правилу `start <= time < end`;
- журнал событий загрузки `collection_events.jsonl`;
- манифест каждого запуска `run_manifest.json`;
- отдельный скрипт мягкой чистки обработанных таблиц `clean_collected_data.py`;
- отдельный скрипт нормализации по временной сетке `normalize_time_grid.py`;
- конфигурация правил интервального распространения `configs/time_normalization.json`;
- конфигурация правил отбора станций `configs/station_selection.json`;
- экспорт нормализованных обучающих таблиц по отдельным GIRO-станциям.

На этапе парсинга данные не отбрасываются. Даже если станция пустая, это фиксируется в журнале. Отбор станций по покрытию относится к следующему этапу обработки после полного сбора данных.

## Быстрый старт

Перейти в папку проекта:

```powershell
cd D:\IonoAutoML
```

Запустить сбор по настройкам из `config.toml`:

```powershell
python .\collect_hf_data.py --config .\config.toml
```

По умолчанию конфиг настроен на период:

```text
2024-01-01T00:00:00Z <= time < 2026-01-01T00:00:00Z
```

Это означает весь 2024 год и весь 2025 год.

## ML/AutoML запуск по станциям

Для основного эксперимента используется 15-минутная сетка, feature dataset
`features_2024_2025_exploration_v0_1_15min` и конфиг:

```text
configs/experiments/baseline_v0.1.json
```

Конфиг задаёт walk-forward схему по дням 2025 года, окна обучения `7`, `21` и
`28` дней, горизонты `15min`, `1h`, `3h`, `6h`, `24h`, baseline-модели,
`LinearRegression`, `ElasticNet`, `RandomForest`, `XGBoost`, `CatBoost` и
AutoML-подбор гиперпараметров.

Полный запуск лучше выполнять поэтапно, станция за станцией:

```powershell
.\.venv\Scripts\python.exe .\run_station_batch.py `
  --config .\configs\experiments\baseline_v0.1.json
```

Проверить первые станции без обучения:

```powershell
.\.venv\Scripts\python.exe .\run_station_batch.py `
  --config .\configs\experiments\baseline_v0.1.json `
  --max-stations 3 `
  --dry-run
```

Запустить одну станцию:

```powershell
.\.venv\Scripts\python.exe .\run_station_batch.py `
  --config .\configs\experiments\baseline_v0.1.json `
  --station TR169
```

Результаты каждой станции сохраняются отдельно:

```text
artifacts/baseline_v0.1/stations/<station>/
  <station>_<experiment>_<run_started_utc>_metrics.csv
  <station>_<experiment>_<run_started_utc>_metrics.parquet
  <station>_<experiment>_<run_started_utc>_predictions.csv
  <station>_<experiment>_<run_started_utc>_predictions.parquet
  <station>_<experiment>_<run_started_utc>_optuna_trials.csv
  <station>_<experiment>_<run_started_utc>_optuna_trials.parquet
  <station>_<experiment>_<run_started_utc>_best_params.csv
  <station>_<experiment>_<run_started_utc>_best_params.parquet
  <station>_<experiment>_<run_started_utc>_feature_importance.csv
  <station>_<experiment>_<run_started_utc>_feature_importance.parquet
  <station>_<experiment>_<run_started_utc>_report.md
  <station>_<experiment>_<run_started_utc>_run_summary.json
```

Такая схема нужна для большого прогона: каждая станция лежит в своей папке, а
имя файла сразу показывает станцию, эксперимент, UTC-время запуска и тип таблицы.
Повторные запуски не затирают старые результаты.

Файл `<station>_<experiment>_<run_started_utc>_report.md` автоматически собирает
сезонную таблицу метрик по моделям и короткий текстовый анализ в стиле статьи:
лучшие модели по `R2` и `RMSE`, слабые сезоны, `MAPE` и ведущие признаки по
важности, если модель их экспортирует.

Во время длинного запуска промежуточный прогресс можно смотреть через:

```powershell
.\.venv\Scripts\python.exe .\watch_run_progress.py `
  --run-dir .\artifacts\baseline_v0.1
```

Быстрый просмотр готовых результатов и промежуточных таблиц выполняется в:

```text
notebooks/ml_results_review.ipynb
```

### Быстрый ML-прогон foF2 на 24 часа

Для ускоренной проверки моделей подготовлен отдельный эксперимент:

```text
configs/experiments/fof2_fast_ml_24h.json
```

Он использует тот же 15-минутный feature dataset
`features_2024_2025_exploration_v0_1_15min`, целевой параметр `foF2`, горизонт
прогноза `24h`, walk-forward оценку на 2025 год и окна обучения `7`, `21`, `28`
дней. В быстрый запуск включены `LinearRegression`, `ElasticNet`,
`RandomForest`, `XGBoost` и `CatBoost`; AutoML/Optuna отключен, чтобы быстрее
получить сравнимые базовые результаты.

Список доступных станций отсортирован по покрытию целевого признака:

```text
configs/station_sets/fof2_fast_ml_24h_all_available.txt
```

Запуск по этому списку:

```powershell
.\.venv\Scripts\python.exe .\run_station_batch.py `
  --config .\configs\experiments\fof2_fast_ml_24h.json `
  --station-file .\configs\station_sets\fof2_fast_ml_24h_all_available.txt
```

Перед большим запуском удобно проверить план без обучения:

```powershell
.\.venv\Scripts\python.exe .\run_station_batch.py `
  --config .\configs\experiments\fof2_fast_ml_24h.json `
  --station-file .\configs\station_sets\fof2_fast_ml_24h_all_available.txt `
  --dry-run
```

В `run_ml_baselines.py` добавлен вывод текущего прогресса по станции, горизонту,
окну обучения, split-итерации и модели. Для просмотра уже выполненного анализа
сохранена копия ноутбука:

```text
notebooks/ml_results_review.executed.ipynb
```

## Запуск за свой период

Период можно переопределить через `--start` и `--end`.

Пример: январь 2024.

```powershell
python .\collect_hf_data.py --config .\config.toml `
  --start 2024-01-01T00:00:00Z `
  --end 2024-02-01T00:00:00Z `
  --output-dir data_test_2024_01
```

Граница `--end` не включается. В примере выше данные собираются с `2024-01-01T00:00:00Z` до последнего момента января, но `2024-02-01T00:00:00Z` уже не входит.

## Выбор источников

Параметр `--sources` позволяет запускать не все сразу, а только нужные источники.

Доступные значения:

- `all` - все источники;
- `giro` - только GIRO;
- `gfz` - только GFZ;
- `omni` - только NASA OMNIWeb;
- `noaa` - только NOAA/SWPC;
- комбинации через запятую, например `gfz,omni`.

### Только индексы GFZ и OMNI

Такой запуск быстрый и удобен для проверки временных шагов геофизических признаков:

```powershell
python .\collect_hf_data.py --config .\config.toml `
  --start 2024-01-01T00:00:00Z `
  --end 2024-02-01T00:00:00Z `
  --output-dir data_test_indices_2024_01 `
  --sources gfz,omni
```

Пробный результат за январь 2024:

- GFZ: 589 строк после фильтрации периода;
- OMNI: 744 строки после фильтрации периода.

### Только GIRO

Такой запуск может быть долгим, потому что GIRO опрашивается по станциям:

```powershell
python .\collect_hf_data.py --config .\config.toml `
  --start 2024-01-01T00:00:00Z `
  --end 2024-02-01T00:00:00Z `
  --output-dir data_test_giro_2024_01 `
  --sources giro
```

Для полного периода 2024-2025 GIRO лучше запускать отдельно от индексов, чтобы удобнее отслеживать прогресс и ошибки по станциям.

## Источники и признаки

### GIRO DIDBase / FastChar

GIRO используется для параметров ионосферы по станциям.

Примеры признаков:

- `foF2`;
- `foF1`;
- `foE`;
- `foEs`;
- `MUF(D)`;
- `hmF2`;
- `TEC`;
- `fmin`.

В конфиге стоит `characteristics = "all"`, поэтому запрашиваются все доступные FastChar-признаки.

Ограничение GIRO: один запрос не должен пересекать границу календарного года. Парсер учитывает это автоматически.

### GFZ Kp API

GFZ используется для исторических геомагнитных и солнечных индексов.

Сейчас собираются:

- `Kp` - шаг 3 часа;
- `ap` - шаг 3 часа;
- `Ap` - шаг 1 день;
- `Fobs` - наблюдаемый F10.7, шаг 1 день;
- `Fadj` - скорректированный F10.7, шаг 1 день.

Пробный сбор за январь 2024:

```text
Kp    248 строк, шаг 3 часа, пропусков 0
ap    248 строк, шаг 3 часа, пропусков 0
Ap     31 строка, шаг 1 день, пропусков 0
Fobs   31 строка, шаг 1 день, пропусков 0
Fadj   31 строка, шаг 1 день, пропусков 0
```

### NASA OMNIWeb

OMNIWeb используется для межпланетного магнитного поля, Dst и параметров солнечного ветра.

Сейчас собираются:

- `bz_gsm_nT`;
- `dst_nT`;
- `flow_speed_km_s`;
- `proton_density_n_cc`.

Подключен часовой OMNI2-режим.

Пробный сбор за январь 2024:

```text
744 строки, шаг 1 час
bz_gsm_nT              пропусков 0
dst_nT                 пропусков 0
flow_speed_km_s         пропусков 0
proton_density_n_cc     пропусков 0
```

## Куда сохраняются данные

Если не указан `--output-dir`, данные пишутся в `data/`.

Структура:

```text
data/
  raw/
    giro/
      station=<station_id>/
        year=<year>/
    gfz/
    omni/
    noaa/
  metadata/
    stations.json
  logs/
    collection_events.jsonl
  processed/
    giro_scaled.csv
    giro_scaled.json
    geophysical_indices.csv
    geophysical_indices.json
    omni_solar_wind.csv
    omni_solar_wind.json
    noaa_observations.csv
    noaa_observations.json
    analytical_hf_dataset.csv
    analytical_hf_dataset.json
  run_manifest.json
```

Дополнительные результаты постобработки могут сохраняться отдельно от сырого архива:

```text
cleaned_2024_2025/
  processed/
    geophysical_indices.csv
    omni_solar_wind.csv
  cleaning_report.md

normalized_2024_2025_top3_5min_by_station/
  analytical_time_grid.csv
  giro_time_grid.csv
  station_coverage.csv
  time_normalization_report.md
  reports/
    station_quality/
      <station>.json
    station_quality_summary.csv
  stations/
    <station>_time_grid.csv
```

Что смотреть после запуска:

- `run_manifest.json` - общий итог запуска: период, источники, количество строк, пути к файлам;
- `logs/collection_events.jsonl` - журнал загрузки, особенно полезен для GIRO;
- `metadata/stations.json` - список найденных GIRO-станций;
- `processed/geophysical_indices.csv` - нормализованные GFZ-индексы;
- `processed/omni_solar_wind.csv` - нормализованные OMNI-данные;
- `processed/giro_scaled.csv` - нормализованные GIRO-строки.

## Форматы

Сейчас используются:

- raw JSON/HTML/TXT - сырые ответы источников;
- CSV - нормализованные таблицы для просмотра и проверки;
- JSON - метаданные, манифесты и нормализованные записи;
- JSON Lines - журнал событий загрузки.

Для больших временных рядов позже может понадобиться Parquet. На текущем этапе CSV удобнее для проверки структуры и пропусков.

## Временные шаги

У источников разные временные шаги:

- GIRO зависит от станции и наличия измерений;
- GFZ `Kp/ap` - 3 часа;
- GFZ `Ap/F10.7` - 1 день;
- OMNI2 hourly - 1 час.

Парсер сохраняет исходные ряды в их родном разрешении. Приведение к единой временной сетке, заполнение пропусков, агрегации по окнам и отбор станций не выполняются на этапе парсинга.

Для подготовки обучающих таблиц используется отдельный этап нормализации:

```powershell
python .\clean_collected_data.py `
  --input-dir data_2024_2025_indices `
  --giro-raw-dir data_2024_2025_giro\raw\giro `
  --output-dir cleaned_2024_2025 `
  --row-threshold 0.30 `
  --column-threshold 0.45

python .\normalize_time_grid.py `
  --time-config configs\time_normalization.json `
  --station-selection-config configs\station_selection.json `
  --giro-raw-dir data_2024_2025_giro\raw\giro `
  --processed-dir cleaned_2024_2025\processed `
  --output-dir normalized_2024_2025_top3_5min_by_station `
  --freq 5min `
  --min-cs 50 `
  --top-stations 3 `
  --split-by-station
```

Для предварительного скана качества без записи больших CSV можно добавить `--quality-only`; тогда будут сформированы только отчёты качества и сводная таблица.

Практичный шаг для обучения сейчас `5min`: автоматическая оценка может находить более мелкий шаг, но он сильно увеличивает размер выборки. GIRO-измерения агрегируются внутри своего 5-минутного окна; если в окне не было измерения, значение не дорисовывается. GFZ/OMNI-значения распространяются вперед только до своего `valid_until`; для них сохраняются `source_time` и `valid_until`. Станции GIRO без строк измерений исключаются из обучающих датасетов. Значение `CS = -1` сохраняется как допустимое неизвестное качество, а не как плохая запись.

## Рабочие заметки

- Сначала собирается сырой архив.
- Пустые станции GIRO не удаляются во время парсинга.
- Для обучения пустые станции GIRO нужно исключать на этапе постобработки; в текущем сборе найдено 65 станций без строк измерений.
- Отчеты `reports/station_quality/<station>.json` фиксируют покрытие по годам, сезонам, тестовому периоду, распределение `CS`, предупреждения и мягкий класс качества станции.
- В текущем скане покрытия за 2024-2025 годы найдено 120 GIRO-станций: 65 станций без строк измерений и 55 станций с данными.
- Для 55 станций с данными при шаге `5min` получены мягкие классы качества: `good` - 18, `usable` - 23, `weak` - 13, `exclude` - 1.
- `good` означает хорошее покрытие целевого параметра и мало пропусков; такие станции подходят и для строгого датасета, и для первых моделей.
- `usable` означает, что данных достаточно для исследовательских экспериментов, но есть предупреждения по покрытию или пропускам.
- `weak` означает слабое покрытие: станцию лучше не брать в основной датасет, но можно оставить для отдельного анализа устойчивости.
- `exclude` означает почти отсутствие полезного целевого ряда; такую станцию не стоит использовать в обучении.
- Для первых исследовательских экспериментов подходят `good + usable`, всего 41 станция; для строгого датасета по текущим правилам проходят 15 станций.
- Набор 41 станции для первого эксперимента зафиксирован в `configs/station_sets/exploration_v0.1.json`.
- Координаты станций хранятся в `configs/stations_metadata.csv` и `configs/stations_metadata.json`; для всех 41 выбранных станций координаты найдены.
- Географическое распределение выбранных 41 станций: южные высокие широты - 1, южные средние - 4, низкие широты - 12, северные средние - 21, северные высокие - 3. Выборка пригодна для первого эксперимента, но имеет перекос в северные средние широты.
- Нормализованные обучающие выборки удобнее хранить отдельными CSV по станциям в `stations/<station>_time_grid.csv`.
- Быстрые визуальные проверки выполняются в `notebooks/quick_quality_plots_ascii.ipynb`: классы качества, сравнение временных шагов, временной ряд станции, пропуски и карта мира. Экспорт фигур по умолчанию выключен через `EXPORT_FIGURES = False`.
- Пустые логи можно удалять, а непустые переносить в `logs_archive/`, чтобы не засорять корень проекта.
- Generated-результаты, логи, сырые данные, нормализованные CSV, quality-scan директории, кэш Python/Jupyter и экспортированные фигуры не коммитятся в репозиторий.
- Отчет покрытия по станциям имеет смысл строить после полного сбора за 2024-2025 годы.
- GFZ и OMNI можно собирать сразу за месяц, год или другой заданный период.
- GIRO нужно собирать осторожнее: по окнам и без пересечения календарного года в одном запросе.
