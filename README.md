++++ 0. Установить pyenv, если его нет

    Следуя инструкции: https://pyenv-win.github.io/pyenv-win/docs/installation.html#powershell

    шаг 0. Открыть PowerShell c правами администратора

    шаг 1. Выполнить следующую команду:
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

    шаг 2. Выполнить следующие команды для определения переменных окружения:

        [System.Environment]::SetEnvironmentVariable('PYENV',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
        [System.Environment]::SetEnvironmentVariable('PYENV_ROOT',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")
        [System.Environment]::SetEnvironmentVariable('PYENV_HOME',$env:USERPROFILE + "\.pyenv\pyenv-win\","User")

        затем

        [System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('path', "User"),"User")

    шаг 3. Проверить установку pyenv:

        pyenv --version.

        Должно быть "pyenv 3.1.1" или что-то подобное

    шаг 4. Установить питон версии 3.10.5, используя pyenv и PowerShell, если его нет

        pyenv install 3.10.5

    шаг 5. Проверить установку питона

        python --version

        Должно быть что-то вроде "python 3.10.5"

    шаг 6. Установить требуемые зависимости из PowerShell в директории "feliksovma-ml\"

        pip install -r requirements.txt

++++ 1. Запустить, содержащую нужные контейнеры инфраструктуру

    шаг 0. Установить докер, если его нет

        https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe?_gl=1*18fc6sc*_ga*MTkwNTQzMDE4MC4xNzEwMTQ4NzM4*_ga_XJWPQMJYHQ*MTcxMzM4OTgzOC4xNS4xLjE3MTMzODk4NDAuNTguMC4w

    шаг 1. Запустить engine

    шаг 2. Запустить докер контейнеры проекта

        шаг 0. перейти в папку "feliksovma-ml\mlflow-docker-compose", содержащую проект, используя облочку выше

        шаг 1. выполнить команду "docker compose up"

++++ 2. Собрать docker image: kinopoisk

    шаг 0. Перейти в директорию "feliksovma-ml\mlflow-docker-compose\mlflow-kinopoisk-img", используя PowerShell в режиме админинстратора

    шаг 1. Собрать image "kinopoisk" выполнив команду "docker build -f .\Dockerfile -t kinopoisk ."? если его еще нет. Это можно
        проверить используя команду "docker image ls" и, если это необходимо, удалить существующий командой "docker image rmi {IMAGE_ID}"

++++ 3. Запуск эксперимент "kinopoisk"

    шаг 0. Добавить датасета kinopoisk_train.csv в Object Store MinIO, используя инструкцию "Webinar3.ipynb\Webinar3.ipynb"

    шаг 1. Используя PowerShell с административным доступом перейти в папку "feliksovma-ml\source", содержащую проект

    шаг 2. Выполнить команду "mlflow run . --experiment-name=kinopoisk --docker-args="network=mlflow-docker-compose_mlflow_net" --build-image"

        !!! НА ДАННОМ ШАГЕ ВОЗНИКНЕТ ИСКЛЮЧЕНИЕ, НО, ТЕМ НЕ МЕНЕЕ, IMAGE БУДЕТ СОЗДАН. ЭТО И ТРЕБУЕТСЯ !!!

    шаг 3. Выяснить TAG IMAGE "movie_rating_prediction" выполнив команду "docker image ls"

    шаг 4. Запустить эксперимент, используя найденный на шаге 3 TAG IMAGE "movie_rating_prediction" (ЗАМЕНИТЬ ++TAG_IMAGE+++ в следующей команде):

        шаг 0. Отредактировать команду. Нужно указать свои АБСОЛЮТНЫЕ ПУТИ и +++ID_IMAGE+++

        "docker run --rm --network mlflow-docker-compose_mlflow_net -v /d/2024/desktop/HomeWork/2055671/feliksovma-ml/source/mlruns:/mlflow/tmp/mlruns -v /d/2024/desktop/HomeWork/2055671/feliksovma-ml/source/mlruns/570021522064556026/0a741890416c49b4b40ffdff6b0578e7/artifacts:/d/2024/desktop/HomeWork/2055671/feliksovma-ml/source/mlruns/570021522064556026/0a741890416c49b4b40ffdff6b0578e7/artifacts -e MLFLOW_RUN_ID=0a741890416c49b4b40ffdff6b0578e7 -e MLFLOW_TRACKING_URI=file:///d/2024/desktop/HomeWork/2055671/feliksovma-ml/source/mlruns -e MLFLOW_EXPERIMENT_ID=570021522064556026 movie_rating_prediction:+++TAG_IMAGE+++ python entrypoint.py kinopoisk_train.csv"

        шаг 1 (КОМАНДА ДОЛЖНА БЫТЬ ЗАПУЩЕНА КАК ОДНА СТРОКА). Запустить эксперимент, используя отредактированную команду выше


