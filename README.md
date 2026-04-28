# Итоговый проект по дисциплине «Внедрение моделей машинного обучения»

----
#MIFIML-2-2026

## Название проекта

Разработка и внедрение сервиса прогнозирования дефолта по кредитным картам с контейнеризацией и A/B-тестированием

## Описание проекта

**Цель проекта:** разработать и внедрить в production-like-среду сервис машинного обучения для прогнозирования дефолта по кредитным картам, который охватывает полный цикл от сохранения модели до организации A/B-тестирования.

**Домен:** финансы / кредитный скоринг.

**Датасет:** Default of Credit Card Clients Dataset с UCI Machine Learning Repository.

- **Ссылка:** [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).
- **Описание:** датасет содержит информацию о клиентах кредитной карты на Тайване с таргетом **default.payment.next.month** (дефолт в следующем месяце). Включает демографические данные, историю платежей, суммы счетов.
- **Актуальность:** идеально подходит для демонстрации полного цикла внедрения ML-модели в production без излишней сложности в части построения модели.

## Задачи и их решение

### **Подготовка модели к production и деплой**

#### Организация кода и контроль версий

- код организован в отдельные папки
- - ./configs - содержит скрипты, обеспечивающие доступ к конфигурационной информации (пути к файлам и прочее)
- - ./data - содержит сырые данные (датасет для обучения модели)
- - ./models - содержит сериализованные обченные модели
- - ./src - содержит файлы исходного кода проекта
- - ./tests - содержит скрипты автоматизированных тестов

- контроль версий осуществляется через ![git](https://github.com)

#### Создание, сохранение и загрузка модели

Текущий проект носит демонстративный характер и не реализует полноценный подход к созданию и обучению
модели. Также отсутствует полный EDA. Выполнена проверка на пропуски.
Файл ./src/train.py реализует алгоритм обучения и сохранения модели.

 - в качестве модели выбрана sklearn.ensemble.HistGradientBoostingClassifier
 - модель показывает точность ~0.822  
 - для сохранения использован joblib.dump()
 - для загрузки используется joblib.load()

Обучение модели

```shell
cd путь/credit_card
source .venv/bin/activate
$env:PYTHONPATH = "src"
python3 -m src.train
```

#### Веб-сервис на Flask

Файл ./src/app.py реализует логику веб-сервиса на Flask. Входящий HTTP-запрос проходит через pre-request hook (_start_timer), затем попадает в endpoint handler, после чего обрабатывается post-request hook (_access_log). Данная логика формирует сбор телеметрии (упрощенно - время выполнения запроса). В коде реализовано 2 endpoint (в соответствии с заданием):

- app.get("/health")
- app.post("/predict")

Для реализации автоматизированных тестов используется модуль pytest и набор тестов в папке ./tests.
В текущем проекте это реализовано минимально (для демонстрации)

- файл ./tests/conftest.py - определение необходимых fixture
- файл ./tests/test_api.py - набор тестов

Для корректной работы модуля тестирования в корне проекта необходим файл pytest.ini

Для определения путей к рабочим файлам используются переменные среды env, а также модуль ./src/config.py.

- env - имеет приоритет - позволяет в контейнере переопределить пути, не изменяя код
- config.py - обеспечивает значения по-умолчанию и работу в рамках локального проекта

#### Дополнительно

##### ONNX

Для ускорения инференса модели, а также инференса на таких языках, как Java, C++, можно использовать ONNX. Для преобразования Python модели необходимо использовать skl2onnx модуль.

Пример

```shell
pip install skl2onnx
```

```Python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# Convert into ONNX format.
from skl2onnx import to_onnx

onx = to_onnx(clr, X[:1])
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Compute the prediction with onnxruntime.
import onnxruntime as rt

sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
```
![Подробнее](https://github.com/onnx/sklearn-onnx)

##### uWSGI + NGINX

- **WSGI (Web-Server Gateway Interface)** - обеспечиает интерфейс, который запускает Flask приложение в нескольких процессах. uWSGI - веб-сервер, который реализует интерфейс WSGI.
- **NGINX** - это веб-сервер, который оптимизирует нагрузку за счёт асинхронной архитектуры, управляемой событиями.

В production среде данная связка используется для повышения производительности, стабильности и безопасности.

### **Воспроизводимость и контейнеризация**

#### Воспроизводимость окружения

Создание файла зависимостей. (перед созданием файла зависимостей нужно создать локальное окружение venv и установить в него необходимые библиотеки)

```shell
python -m venv .venv
# установка зависимостей
pip install scikit-learn
# ...
pip freeze > requirements.txt
```

#### Docker-контейнеризация

Docker файл - ./Dockerfile.

- использую образ python:3.12-slim
- задаёт рабочую директорию
- заполняет переменные env
- отключена запись в .pyc
- логи выводятся сразу в stdout/stderr
- обновляет индексы apt, устанавливает инструменты, чистит кэш apt
- копирует файл зависимостей
- устанавливает зависимости
- копирует исходный код и модель
- слушает порт 3031
- запускает uWSGI (с конфигом uwsgi.ini) на 0.0.0.0:3031

Настройки в uwsgi.ini

- импорт объекта app из модуля app
- включает мастер-процесс uWSGI
- запуск на 2-х worker процессах
- по 2 потока на каждый worker
- устанавливает socket=0.0.0.0:3031
- при остановке происходит очищение временных файлов
- завершает uWSGI при SIGTERM

Сборка, загрузка и запуск контейнера

```shell
docker buildx build -t credit-api:latest --output type=docker,dest=./credit-api.tar .
docker load -i ./credit-api.tar
docker run --rm -p 5000:3031 credit-api:latest
```

Запуск образа из docker-hub
```shell
docker pull qdwefac/credit-api:latest
docker run --rm -p 5000:3031 credit-api:latest
```

При этом запустится контейнер Flask + uWSGI, но curl http... не сработает, так как задумано, чтобы все работало через tcp от NGINX.

#### Создание образа веб-сервиса

После того, как реализован необходимый функционал и выполнена отладка переходим к созданию образа веб-сервиса. В текущем примере уже создан docker-compose.yaml (комментарии далее)

### **Сервисная архитектура и оркестрация**

#### Архитектура сервиса

Описана в ARCHITECTURE.md

#### Оркестрация

Собственно проект собирается в Docker Compose из корневой директории

```shell
docker compose up -d --build
```

В файле ./docker-compose.yml настроено следующее:

- запускается 2 контейнера api + nginx
- nginx запускается после api
- nginx публикует порт 80 контейнера на 8080 хоста
- обращение к сервису снаружи через localhost:8080

Пример запроса (Windows PowerShell):
```shell
> curl http://localhost:8080/health

StatusCode        : 200
StatusDescription : OK
Content           : {"service":"credit-api","status":"Ok","version":"1.0.0"}

RawContent        : HTTP/1.1 200 OK
                    Connection: keep-alive
                    Content-Length: 57
                    Content-Type: application/json
                    Date: Mon, 27 Apr 2026 12:46:55 GMT
                    Server: nginx/1.27.5

                    {"service":"credit-api","status":"Ok","version...
Forms             : {}
Headers           : {[Connection, keep-alive], [Content-Length, 57], [Content-Type, application/json], [Date, Mon, 27 A
                    pr 2026 12:46:55 GMT]...}
Images            : {}
InputFields       : {}
Links             : {}
ParsedHtml        : System.__ComObject
RawContentLength  : 57
```

#### Обзор инструментов MLops

- DVC (Data Version Control). Используется для контроля версий данных и артефактов моделей (напрмер датасет, на котором обучалась модель, конкретное разбиение train/test/val, model.joblib). Это позволяет воспроизводить обучение по конкретной версии данных, понимать на какой версии получена текущая модель, хранить "тяжелые" артефакты в удаленных хранилищах, а в git держать только метаданные.
- MLFlow. Используется для воспроизводимости экспериментов по обучению моделей на уровне гиперпараметров, используемых метрик и версий моделей.

#### Бизнес-метрики

1. Ожидаемые финансовые потери от ошибок модели.  
Можно учитывать стоимость ошибок классификации. Например по такой формуле:  
```latex
Loss = FP * C_FP + FN * C_FN
```
где
- FP - отклонили клиента, которого могли одобрить (ложный риск или упущенная прибыль)
- FN - не выявили дефолтного клиента (прямые потери)
- C_FP, C_FN - цена каждой ошибки

По сделаннм предсказаниям и реальным данным строим confusion_matrix, подставляем цены и считаем итоговые потери.

2. Доля одобренных заявок при фиксированном уровне риска.  
Максимизируем выдачу на заданном уровне риска.
- считаем вероятность дефолта для всех заявок (prob_def)
- выбираем порог t и одобряем, если prob_def < t
- считаем:
- - Доля одобренных = Всего одобрено / Всего заявок
- - Доля дефолтов среди одобренных = Дефолтов в одобренных / Всего одобрено  
Ищем такой порог t, при котором доля дефолтов среди одобренных не выше некоторого допустимого, а доля одобренных - максимальна.

### **Организация A/B-тестирования**

Описана в файле AB_TEST_PLAN.md

### **Документация и воспроизводимость**

#### README.md

Вы его читаете

#### Структура репозитория

Описана выше

## Описание API

```shell
$body = @{
  LIMIT_BAL=200000; SEX=2; EDUCATION=2; MARRIAGE=1; AGE=35
  PAY_0=0; PAY_2=0; PAY_3=0; PAY_4=0; PAY_5=0; PAY_6=0
  BILL_AMT1=0; BILL_AMT2=0; BILL_AMT3=0; BILL_AMT4=0; BILL_AMT5=0; BILL_AMT6=0
  PAY_AMT1=0; PAY_AMT2=0; PAY_AMT3=0; PAY_AMT4=0; PAY_AMT5=0; PAY_AMT6=0
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri "http://127.0.0.1:8080/predict" -Method Post -ContentType "application/json" -Body $body
```