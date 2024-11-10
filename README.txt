#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ smart_city - умный город - detection
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~
событие alarm "Огонь" "fire-detection" "fire_detection"
0: "fire" - "огонь"
1: "non-fire" - "не огонь"
#~~~~~~~~~~~~~~~~~~~~~~~~
событие alarm "ДТП - столкновение легковых автомобилей" "car-accident-detection" "car_accident_detection"
0: "car-accident" - "ДТП - столкновение легковых автомобилей"
1: "non-car-accident" - `не ДТП - столкновение легковых автомобилей"
#~~~~~~~~~~~~~~~~~~~~~~~~
событие alarm "Падение человека" "person-fall-detection" "person_fall_detection"
0: "person-fall" - "падение человека"
1: "non-person-fall" - "падение человека"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Install the virtual environment
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ запускаем cmd от Администратора
#~ создаю новую среду -> virtualenv -> с именем "camenv8"
#~~~~~~~~~~~~~~~~~~~~~~~~
# c:
# d:
# cd c:\my_campy
# cd d:\my_campy
# python -m venv camenv8
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ Ubuntu Linux

1. Откройте терминал (Ctrl+Alt+T).
2. Обновите систему командой:
sudo apt update && sudo apt upgrade
3. Установите необходимые пакеты для работы с Python:
sudo apt install python3-venv python3-dev build-essential
4. Создайте новую виртуальную среду:
python3 -m venv my_env
Это создаст виртуальную среду под названием my_env.

akozlov@akozlov-desktop:~$ python3 -V
Python 3.6.9

создать папку для разработки
mkdir /home/akozlov/my_campy
cd /home/akozlov/my_campy

создаю виртуальную переменную
python3 -m venv camenv8
python3.10 -m venv camenv8

5. Активируйте виртуальную среду:
source camenv8/bin/activate

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
#~~~~~~~~~~~~~~~~~~~~~~~~
# c:
# d:
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python keras_trainer.py
#~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ pypi.org ---> все библиотеки для установки через pip
https://pypi.org/project/pyproj/
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install Depedencies
from requirements.txt file

pip install -r requirements.txt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python.exe -m pip install --upgrade pip
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
8.2.10   6 мая 2024 г.
pip install --upgrade ultralytics==8.2.10
pip install ultralytics==8.2.10
pip show ultralytics
Name: ultralytics Version: 8.2.10
(camenv8a) c:\my_campya>pip show ultralytics
Name: ultralytics
Version: 8.2.10
#~~~~~~~~~~~~~~~~~~~~~~~~
31 января 2023г.
pip install mediapipe
pip install mediapipe==0.9.1.0
pip install --upgrade mediapipe==0.9.1.0
pip show mediapipe
Name: mediapipe Version: 0.9.1.0
#~~~~~~~~~~~~~~~~~~~~~~~~
2.9.3   16 нояб. 2022 г.
pip install --upgrade tensorflow==2.9.3
pip show tensorflow
Name: tensorflow Version: 2.9.3
#~~~~~~~~~~~~~~~~~~~~~~~~
pip install pydot
pip show pydot
Name: pydot
Version: 3.0.1

pip install pydotplus
pip show pydotplus
Name: pydotplus
Version: 2.0.2

You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/)
for plot_model/model_to_dot to work.
Windows
    Stable Windows install packages, built with Microsoft Visual Studio 16 2019:
        graphviz-12.1.1
            graphviz-12.1.1 (32-bit) ZIP archive [sha256] (contains all tools and libraries)
            graphviz-12.1.1 (32-bit) EXE installer [sha256]
            graphviz-12.1.1 (64-bit) ZIP archive [sha256] (contains all tools and libraries)
            graphviz-12.1.1 (64-bit) EXE installer [sha256]

c:\Program Files\Graphviz\
c:\Program Files\Graphviz\bin\
c:\Program Files\Graphviz\lib\
#~~~~~~~~~~~~~~~~~~~~~~~~
1.1.0   28 янв. 2023 г.
pip install --upgrade autokeras==1.1.0
pip install autokeras==1.1.0
pip show autokeras
(camenv8a) c:\my_campya>pip show autokeras
Name: autokeras
Version: 1.1.0