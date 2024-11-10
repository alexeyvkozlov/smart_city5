#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import cv2
import numpy as np
import threading
from threading import Thread
import time #~ работа со временем
import datetime #~ работа с датой
import requests

from tensorflow.keras.models import load_model #загрузка сохраненной модели

from settings_reader import SettingsReader
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FireThread(Thread):
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, prog_path: str, report_dir: str):
    super().__init__()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ не настраиваемые параметры
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ cоздание словаря camera-dictionary
    self.cam_dict = {}
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.cam_id = 0
    self.cam_name = 'Perimeter0'
    #~ список классов для детектирования
    self.classes_lst1 = ['fire', 'non-fire']
    #~ id детектируемого класса -> 'fire'
    self.class_id = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ параметры, которые считываю из ini-файла
    ini_reader = SettingsReader(prog_path)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ секция [FIRE_DETECTOR] 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.weights = ini_reader.get_fire_model_weights()
    self.img_size = ini_reader.get_fire_model_image_size()
    self.threshold = ini_reader.get_fire_predict_threshold()
    self.rdir = report_dir
    self.alarm_count = ini_reader.get_fire_alarm_count()
    self.alarm_time = ini_reader.get_fire_alarm_time()
    self.siren_time = ini_reader.get_fire_siren_time()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ секция [TELEGRAM_BOT] 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.is_telegram = ini_reader.get_telegram_is_active()
    self.bot_url = ini_reader.get_telegram_url()
    self.bot_token = ini_reader.get_telegram_token()
    self.chat_id = ini_reader.get_telegram_chat_id()
    print(f'[INFO.FireThread] camera: dictionary: {self.cam_dict}')
    print(f'[INFO.FireThread] classes: len: {len(self.classes_lst1)}, {self.classes_lst1}')
    print(f'[INFO.FireThread] class_id: {self.class_id}')
    print(f'[INFO.FireThread] weights file name: `{self.weights}`')
    print(f'[INFO.FireThread] keras image size: {self.img_size}')
    print(f'[INFO.FireThread] fire predict threshold: {self.threshold}')
    print(f'[INFO.FireThread] report directory: `{self.rdir}`')
    print(f'[INFO.FireThread] alarm-count: {self.alarm_count}, alarm-time: {self.alarm_time}, siren-time: {self.siren_time}')
    print(f'[INFO.FireThread] is-telegram: {self.is_telegram}')
    print(f'[INFO.FireThread] bot-url: {self.bot_url}')
    # print(f'[INFO.FireThread] bot-token: {self.bot_token}')
    print(f'[INFO.FireThread] chat-id: {self.chat_id}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    if os.path.isdir(self.rdir):
      if not os.path.exists(self.rdir):
        os.makedirs(self.rdir)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ текщий кадр
    self.frame = None
    #~ флаг готовности нового кадра для обработки
    self.frame_ready = False
    #~ флаг остановки выхода из цикла
    self.stop_event = threading.Event()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.daemon = True
    self.start()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_stop(self):
    return self.stop_event.is_set()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def set_stop(self, value):
    print('[INFO.FireThread] set_stop...')
    self.stop_event.clear()
    self.stop_event.wait(value)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_frame_ready(self):
    return self.frame_ready

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ метод для добавления изображения
  def add_frame(self, cam_id: int, cam_name: str, frame: np.ndarray):
    # print(f'[INFO.FireThread] type(frame): `{type(frame)}`')
    # print('[INFO.FireThread] add_frame...')
    if not self.frame_ready:
      self.cam_id = cam_id
      self.cam_name = cam_name
      self.frame = frame.copy()
      #~ устанавливаем флаг
      self.frame_ready = True
      # print('[INFO.FireThread] add_frame start processing...')

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def run(self):
    print('[INFO.FireThread] run: start...')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ tensorflow загрузка модели из файла
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model = load_model(self.weights)
    print(f'[INFO.FireThread] keras model-weights: `{self.weights}`')
    #~ проверяем архитектуру модели
    # print(model.summary())
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ указываем размер изображения, на котором была обучена нейронка  
    #~ первое значение — это ширина, второе значение — это высота.
    target_size=(self.img_size, self.img_size)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    while not self.get_stop():
      #~ получаем текущую дату и время
      current_time = datetime.datetime.now()
      # print(f'[INFO.FireThread] =>while iteration[{current_time}]: frame_ready: {self.frame_ready}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ ввожу таймаут для уменьшения потребления ресурсов
      #~ и для реагирования на события от пользователя
      if not self.frame_ready:
        #~ время указывается в секундах
        #~ 25fps->1/25=0.04sec, 30fps->1/30=0.03sec, 100fps->1/100=0.01sec, 
        time.sleep(0.01) #0.01 0.03 0.04 0.1 1 10
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ обрабатываем изображение - классифицируем - предсказываем: "fire","non-fire" 
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ масштабирование/сжатие изображения до размеров 224x224
      #~ в любом случае сжимаем, так как видео-камер с разрешение 224x224 не бывает
      #~ а изображение с очень высокой вероятностью с видеокамеры
      frame224 = cv2.resize(self.frame, target_size, interpolation=cv2.INTER_AREA)
      # print(f'[INFO.FireThread] original-frame: width: {self.frame.shape[1]}, height: {self.frame.shape[0]}')
      # print(f'[INFO.FireThread]   frame224: width: {frame224.shape[1]}, height: {frame224.shape[0]}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ для предсказания нормализуем значения пикселей - делим на 255, приводим к диапазону [0..1]
      frame224 = frame224/255.0
      #~ добавление одной оси в начале, чтобы нейронка могла распознать пример
      Xexd = np.expand_dims(frame224, axis=0)
      #~ распознавание примера изображения - определение его класса 
      prediction = model.predict(Xexd, verbose=0) 
      is_alarmN = False
      if prediction[0][0] < self.threshold:
        is_alarmN = True
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # if prediction[0][0] < 0.5:
      #   print(f'[INFO.FireThread] prediction: {prediction}: threshold: {self.threshold} -> {is_alarmN}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if not is_alarmN:
        self.frame_ready = False
        continue
      # print('[INFO.FireThread] --->Alarm1')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ произошел alarm - детектирование требуемой сущности
      alarm_datetimeN = datetime.datetime.now()
      #~ проверяем эту камеру в словаре, если нет, то добавляем ее
      if not self.cam_id in self.cam_dict:
        alarm_datetime1 = alarm_datetimeN.replace(year=alarm_datetimeN.year - 1)
        self.cam_dict[self.cam_id] = (1,alarm_datetimeN,alarm_datetime1)
        #~ вероятность детектирования события по одному кадру значительно ниже, чем по нескольким последовательным
        #~ поэтому необходимо минимум два кадра, чтобы считать, что событие аларма состоялось 
        self.frame_ready = False
        continue
      # print('[INFO.FireThread] --->Alarm2')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ это как минимум второй кадр по этому событию
      #~ делаем необходимые проверки, чтобы понять это действительно аларм на заданных настройках или нет
      alarmN = self.cam_dict[self.cam_id]
      #~ вычисляем разницу между алармами
      alarm_datetime1 = alarmN[2]
      delta_alarm_sec1 = (alarm_datetimeN - alarm_datetime1).total_seconds()
      delta_alarm_secN = (alarm_datetimeN - alarmN[1]).total_seconds()
      # print(f'[INFO.FireThread] alarmN: {alarmN}')
      # print(f'[INFO.FireThread]   alarmN[0]: {alarmN[0]}')
      # print(f'[INFO.FireThread]   alarmN[1]: {alarmN[1]}')
      # print(f'[INFO.FireThread]   alarm_datetime1: {alarm_datetime1}')
      # print(f'[INFO.FireThread]   delta_alarm_secN: {delta_alarm_secN}, delta_alarm_sec1: {delta_alarm_sec1}')
      if delta_alarm_sec1 < self.siren_time:
        #~ порог от предыдущего аларма не превышен
        self.frame_ready = False
        continue
      # print('[INFO.FireThread] --->Alarm3')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем прервана псевдопоследовательности кадров алармов или нет
      if delta_alarm_secN > self.alarm_time:
        #~ псевдопоследовательности прервана - инициализируем ожидание нового аларма
        self.cam_dict[self.cam_id] = (1,alarm_datetimeN,alarm_datetime1)
        self.frame_ready = False
        continue
      # print('[INFO.FireThread] --->Alarm4')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сравниваем число последовательных алармов, с заданным порогом
      alarm_countN = alarmN[0]+1
      # print(f'[INFO.FireThread]    alarm_count2: {alarm_countN}')
      if alarm_countN < self.alarm_count:
        #~ требуемого значения по порогу еще не достигли - продолжаем
        #~ накапливать последовательные алармы
        self.cam_dict[self.cam_id] = (alarm_countN,alarm_datetimeN,alarm_datetime1)
        self.frame_ready = False
        continue
      # print('[INFO.FireThread] --->Alarm5')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ аларм состоялся
      self.cam_dict[self.cam_id] = (1,alarm_datetimeN,alarm_datetimeN)
      print(f'[INFO.FireThread] ===>Alarm: {self.classes_lst1[self.class_id]}!!! {alarm_datetimeN.strftime("%Y.%m.%d %H:%M:%S")}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняю изображение кадра с алармом на диск
      alarm_dir = os.path.join(self.rdir, alarm_datetimeN.strftime("%Y%m%d"))
      if not os.path.exists(alarm_dir):
        os.makedirs(alarm_dir)
      #~ и подписываем продетектированное событие -> predict
      #~ добавляем в изображение этикетку события
      x1 = self.frame.shape[1]-266
      cv2.rectangle(self.frame, (x1,3), (self.frame.shape[1]-4,64), (255,255,255), -1)
      alarm_datetime_str1 = f'time: {alarm_datetimeN.strftime("%Y.%m.%d %H:%M:%S")}'
      cv2.putText(self.frame, alarm_datetime_str1, (x1+2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
      alarm_cam = f'camera: {self.cam_name}'
      cv2.putText(self.frame, alarm_cam, (x1+2,38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
      alarm_type = f'alarm: {self.classes_lst1[self.class_id]}'
      cv2.putText(self.frame, alarm_type, (x1+2,56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
      # fname2 = f'{alarm_datetimeN.strftime("%Y%m%d_%H%M%S")}.jpg'
      fname2 = f'{alarm_datetimeN.strftime("%Y%m%d_%H%M%S")}_{self.cam_name}_{self.classes_lst1[self.class_id]}.jpg'
      img_fname2 = os.path.join(alarm_dir, fname2) #png jpg
      # print(f'[INFO] img_fname2: `{img_fname2}`')
      cv2.imwrite(img_fname2, self.frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отправляем сообщение в телеграм-бота
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if self.is_telegram:
        caption = f'time: {alarm_datetimeN.strftime("%Y.%m.%d %H:%M:%S")}, camera: {self.cam_name}, alarm: {self.classes_lst1[self.class_id]}'
        files = {'photo': open(img_fname2, 'rb')}
        resp = requests.post(self.bot_url + self.bot_token + '/sendPhoto?chat_id=' + self.chat_id + '&caption=' + caption, files=files)
        print(f'[INFO.FireThread]   telebot response code: {resp.status_code}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ выставляем флаг, что отработали это изображение
      self.frame_ready = False
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ поток ожидания и обработки изображений завершен
    print('[INFO.FireThread] run: finish!')