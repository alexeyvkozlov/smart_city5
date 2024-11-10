#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from threading import Thread
import time #~ работа со временем
import datetime #~ работа с датой
import requests

import mediapipe as mp
import math

from settings_reader import SettingsReader
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class PersonFallThread(Thread):
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
    self.classes_lst1 = ['person-fall', 'non-person-fall']
    #~ id детектируемого класса -> 'person-fall'
    self.class_id = 0
    #~ детектируем только класс 'person' -> COCO:
    #~ 0: person
    #~ 1: bicycle
    #~ 2: car
    #~ 3: motorcycle
    self.yolo_class_id = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ параметры, которые считываю из ini-файла
    ini_reader = SettingsReader(prog_path)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ секция [FALL_DETECTOR] 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model_mode1 = ini_reader.get_fall_yolo_model()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ YOLO - You Only Look Once
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ model mode
    #~ n: YOLOv8n -> nano
    #~ s: YOLOv8s -> small
    #~ m: YOLOv8m -> medium
    #~ l: YOLOv8l -> large
    #~ x: YOLOv8x -> extra large
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ load a model 'n','s','m','l','x'
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model_mode2 = 'm'
    if 'nano' == model_mode1:
      model_mode2 = 'n'
    elif 'small' == model_mode1:
      model_mode2 = 's'
    elif 'medium' == model_mode1:
      model_mode2 = 'm'
    elif 'large' == model_mode1:
      model_mode2 = 'l'
    elif 'extra large' == model_mode1:
      model_mode2 = 'x'
    self.yolo_model = f'yolov8{model_mode2}.pt'
    self.yolo_img_size = ini_reader.get_fall_yolo_image_size()
    self.yolo_conf = ini_reader.get_fall_yolo_confidence()
    self.person_ratio = ini_reader.get_fall_person_ratio()
    self.pipe_conf = ini_reader.get_fall_pipe_confidence()
    self.footgrav = ini_reader.get_fall_footgrav_deg()
    self.rdir = report_dir
    self.alarm_count = ini_reader.get_accident_alarm_count()
    self.alarm_time = ini_reader.get_accident_alarm_time()
    self.siren_time = ini_reader.get_accident_siren_time()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ секция [TELEGRAM_BOT] 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.is_telegram = ini_reader.get_telegram_is_active()
    self.bot_url = ini_reader.get_telegram_url()
    self.bot_token = ini_reader.get_telegram_token()
    self.chat_id = ini_reader.get_telegram_chat_id()
    print(f'[INFO.PersonFallThread] camera: dictionary: {self.cam_dict}')
    print(f'[INFO.PersonFallThread] classes: len: {len(self.classes_lst1)}, {self.classes_lst1}')
    print(f'[INFO.PersonFallThread] class_id: {self.class_id}, yolo_class_id: {self.yolo_class_id}')
    print(f'[INFO.PersonFallThread] yolo model: `{self.yolo_model}`')
    print(f'[INFO.PersonFallThread] yolo image size: {self.yolo_img_size}')
    print(f'[INFO.PersonFallThread] yolo confidence: {self.yolo_conf}')
    print(f'[INFO.PersonFallThread] person ratio: {self.person_ratio}')
    print(f'[INFO.PersonFallThread] pipe confidence: {self.pipe_conf}')
    print(f'[INFO.PersonFallThread] foot gravity: {self.footgrav}')
    print(f'[INFO.PersonFallThread] report directory: `{self.rdir}`')
    print(f'[INFO.PersonFallThread] alarm-count: {self.alarm_count}, alarm-time: {self.alarm_time}, siren-time: {self.siren_time}')
    print(f'[INFO.PersonFallThread] is-telegram: {self.is_telegram}')
    print(f'[INFO.PersonFallThread] bot-url: {self.bot_url}')
    # print(f'[INFO.PersonFallThread] bot-token: {self.bot_token}')
    print(f'[INFO.PersonFallThread] chat-id: {self.chat_id}')
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
    print('[INFO.PersonFallThread] set_stop...')
    self.stop_event.clear()
    self.stop_event.wait(value)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_frame_ready(self):
    return self.frame_ready

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ метод для добавления изображения
  def add_frame(self, cam_id: int, cam_name: str, frame: np.ndarray):
    # print(f'[INFO.PersonFallThread] type(frame): `{type(frame)}`')
    # print('[INFO.PersonFallThread] add_frame...')
    if not self.frame_ready:
      self.cam_id = cam_id
      self.cam_name = cam_name
      self.frame = frame.copy()
      #~ устанавливаем флаг
      self.frame_ready = True
      # print('[INFO.PersonFallThread] add_frame start processing...')

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def calc_azim(self, x2: int, y2: int):
    tan_alpha = 0.0
    alpha = 0.0
    if 0 == x2 and 0 == y2:
      alpha = 0.0
    elif 0 == x2 and y2 > 0:
      alpha = 0.0
    elif x2 > 0 and y2 > 0:
      tan_alpha = x2/y2
      alpha =  math.atan(tan_alpha)
    elif x2 > 0 and 0 == y2:
      alpha = math.pi/2.0
    elif x2 > 0 and y2 < 0:
      tan_alpha = -1.0*y2/x2
      alpha = math.atan(tan_alpha) + math.pi/2.0
    elif 0 == x2 and y2 < 0:
      alpha = math.pi
    elif x2 < 0 and y2 < 0:
      tan_alpha = x2/y2
      alpha = math.atan(tan_alpha) + math.pi
    elif x2 < 0 and 0 == y2:
      alpha = 1.5*math.pi
    elif x2 < 0 and y2 > 0:
      tan_alpha = -1.0*y2/x2
      alpha = math.atan(tan_alpha) + 1.5*math.pi
    #~~~~~~~~~~~~~~~~~~~~~~~~
    ialpha = int(alpha*180.0/math.pi)
    return ialpha

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def run(self):
    print('[INFO.PersonFallThread] run: start...')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ load YOLOv8 model on standard dataset 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model = YOLO(self.yolo_model)
    print(f'[INFO.PersonFallThread] yolo standard weights: `{self.yolo_model}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    while not self.get_stop():
      #~ получаем текущую дату и время
      current_time = datetime.datetime.now()
      # print(f'[INFO.PersonFallThread] =>while iteration[{current_time}]: frame_ready: {self.frame_ready}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ ввожу таймаут для уменьшения потребления ресурсов
      #~ и для реагирования на события от пользователя
      if not self.frame_ready:
        #~ время указывается в секундах
        #~ 25fps->1/25=0.04sec, 30fps->1/30=0.03sec, 100fps->1/100=0.01sec, 
        time.sleep(0.01) #0.01 0.03 0.04 0.1 1 10
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ обрабатываем изображение - ищем сущности на изображении 
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # print('~'*70)
      #~ предсказание модели -> на первую детекцию уходит продолжительное время
      #~ все последующие происходят значительно быстрее
      # print('-------[INFO.PersonFallThread] start YOLO analysis-------')
      # yodets = model(self.frame, imgsz=self.img_size, verbose=True)[0]
      yodets = model(self.frame, imgsz=self.yolo_img_size, verbose=False)[0]
      # print('=======[INFO.PersonFallThread] finish YOLO analysis=======')
      is_alarmN = False
      x1,y1,x2,y2 = 0,0,0,0
      for yodet in yodets.boxes.data.tolist():
        yox1, yoy1, yox2, yoy2, yoconf, yoclass_id = yodet
        # print(f'[INFO]  yox1: {yox1}, yoy1: {yoy1}, yox2: {yox2}, yoy2: {yoy2}')
        feature_id = int(yoclass_id)
        # print(f'[INFO]  yoclass_id: {yoclass_id}, self.class_id: {self.class_id}, feature_id: {feature_id}')
        if not feature_id == self.yolo_class_id:
          continue
        # print(f'[INFO]  yoconf: {yoconf}, self.conf: {self.conf}')
        if yoconf < self.yolo_conf:
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        person_width = yox2 - yox1
        person_height = yoy2 - yoy1
        person_ratio = person_width/person_height
        # person_ratio_lbl = f'{round(person_ratio, 2)}'
        # print(f'[INFO] person: width: {round(person_width, 2)}, height: {round(person_height, 2)}, ratio: {round(person_ratio, 2)}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        # #~ координаты продетектированного feature -> person
        x1 = int(yox1)
        y1 = int(yoy1)
        x2 = int(yox2)
        y2 = int(yoy2)
        feature_conf = yoconf
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ сравниваем отношение ширины к высоте person, если порог превышен, фиксируем как падение 
        if person_ratio < self.person_ratio:
          # print('[INFO.PersonFallThread] YOLO Fall False')
          continue
        # print('[INFO.PersonFallThread] YOLO Fall True')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ с высокой вероятностью продетектированная сущность - падающий человек,
        #~ дополнительно проверим по узлам, с использованием media pipe
        #~ опытным путем установил, что media pipe лучше отрабатывает, рамку yolo увеличить
        border_width1 = 0.6*(yox2 - yox1)
        border_height1 = 0.6*(yoy2 - yoy1)
        #~ m - media pipe
        x1m = int(yox1 - border_width1)
        x2m = int(yox2 + border_width1)
        y1m = int(yoy1 - border_height1)
        y2m = int(yoy2 + border_height1)
        if x1m < 0:
          x1m = 0
        if x2m >= self.frame.shape[1]:
          x2m = self.frame.shape[1]-1
        if y1m < 0:
          y1m = 0
        if y2m > self.frame.shape[0]:
          y2m = self.frame.shape[0]-1
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ выделение фрагмента
        img2m = self.frame[y1m:y2m, x1m:x2m]
        img_width2 = img2m.shape[1]
        img_height2 = img2m.shape[0]
        # print(f'[INFO.PersonFallThread] ===>img2m: width: {img_width2}, height: {img_height2}')
        # print(f'[INFO.PersonFallThread]   x1m: {x1m}, x2m: {x2m}, y1m: {y1m}, y2m: {y2m}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        # img_fname2 = os.path.join(self.dst_dir2, base_fname1+'-01.png')
        # cv2.imwrite(img_fname2, img1)
        # img_fname2 = os.path.join(self.dst_dir2, base_fname1+'-02.png')
        # cv2.imwrite(img_fname2, img2m)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        try:
          #~ рассчитываем скелет человеческого тела с помощью Media Pipe - Pose Estimation
          mp_pose = mp.solutions.pose
          npose = mp_pose.Pose(min_detection_confidence=self.pipe_conf)
          podets = npose.process(img2m)
          #~ human pose landmarks
          # print('[INFO] human pose landmarks') 
          human_dots = podets.pose_landmarks.landmark
          # print(f'[INFO] human_dots: len: {len(human_dots)}')
          # [INFO] human_dots: len: 33
          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          #~ 23 - left hip - левое бедро
          #~ 24 - right hip - правое бедро
          #~~~~~~~~~~~~~~~~~~~~~~~~
          left_hip_x = int(human_dots[mp_pose.PoseLandmark.LEFT_HIP].x * img_width2)
          left_hip_y = int(human_dots[mp_pose.PoseLandmark.LEFT_HIP].y * img_height2)
          # print(f'[INFO.PersonFallThread]   23-1 - left hip - левое бедро: x: {left_hip_x}, y: {left_hip_y}')
          if left_hip_x < 0 or left_hip_x >= img_width2 or left_hip_y < 0 or left_hip_y >= img_height2:
            continue
          left_hip_x += x1m
          left_hip_y += y1m
          # print(f'[INFO.PersonFallThread]     23-2 - left hip - левое бедро: x: {left_hip_x}, y: {left_hip_y}')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_hip_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_HIP].x * img_width2)
          right_hip_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_HIP].y * img_height2)
          # print(f'[INFO.PersonFallThread]   24-1 - right hip - правое бедро: x: {right_hip_x}, y: {right_hip_y}')
          if right_hip_x < 0 or right_hip_x >= img_width2 or right_hip_y < 0 or right_hip_y >= img_height2:
            continue
          right_hip_x += x1m
          right_hip_y += y1m
          # print(f'[INFO.PersonFallThread]     24-2 - right hip - правое бедро: x: {right_hip_x}, y: {right_hip_y}')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ center of gravity of whole body - центр тяжести всего тела
          bodygrav_x = int((left_hip_x+right_hip_x)/2)
          bodygrav_y = int((left_hip_y+right_hip_y)/2)
          # print(f'[INFO.PersonFallThread]     center of gravity of whole body - центр тяжести всего тела: x: {bodygrav_x}, y: {bodygrav_y}')
          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          #~ 27 - left ankle - левая лодыжка
          #~ 28 - right ankle - правая лодыжка
          #~ 29 - left heel - левая пятка
          #~ 30 - right heel - правая пятка
          #~ 31 - left foot index - указательный палец левой стопы
          #~ 32 - right foot index - указательный палец правой стопы
          #~~~~~~~~~~~~~~~~~~~~~~~~
          left_ankle_x = int(human_dots[mp_pose.PoseLandmark.LEFT_ANKLE].x * img_width2)
          left_ankle_y = int(human_dots[mp_pose.PoseLandmark.LEFT_ANKLE].y * img_height2)
          # print('[INFO.PersonFallThread] 27 - left ankle - левая лодыжка')
          if left_ankle_x < 0 or left_ankle_x >= img_width2 or left_ankle_y < 0 or left_ankle_y >= img_height2:
            continue
          left_ankle_x += x1m
          left_ankle_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_ankle_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img_width2)
          right_ankle_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img_height2)
          # print('[INFO.PersonFallThread] 28 - right ankle - правая лодыжка')
          if right_ankle_x < 0 or right_ankle_x >= img_width2 or right_ankle_y < 0 or right_ankle_y >= img_height2:
            continue
          right_ankle_x += x1m
          right_ankle_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          left_heel_x = int(human_dots[mp_pose.PoseLandmark.LEFT_HEEL].x * img_width2)
          left_heel_y = int(human_dots[mp_pose.PoseLandmark.LEFT_HEEL].y * img_height2)
          # print('[INFO.PersonFallThread] 29 - left heel - левая пятка')
          if left_heel_x < 0 or left_heel_x >= img_width2 or left_heel_y < 0 or left_heel_y >= img_height2:
            continue
          left_heel_x += x1m
          left_heel_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_heel_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_HEEL].x * img_width2)
          right_heel_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_HEEL].y * img_height2)
          # print('[INFO.PersonFallThread] 30 - right heel - правая пятка')
          if right_heel_x < 0 or right_heel_x >= img_width2 or right_heel_y < 0 or right_heel_y >= img_height2:
            continue
          right_heel_x += x1m
          right_heel_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          left_foot_index_x = int(human_dots[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * img_width2)
          left_foot_index_y = int(human_dots[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * img_height2)
          # print('[INFO.PersonFallThread] 31 - left foot index - указательный палец левой стопы')
          if left_foot_index_x < 0 or left_foot_index_x >= img_width2 or left_foot_index_y < 0 or left_foot_index_y >= img_height2:
            continue
          left_foot_index_x += x1m
          left_foot_index_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_foot_index_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * img_width2)
          right_foot_index_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * img_height2)
          # print('[INFO.PersonFallThread] 31 - left foot index - указательный палец правой стопы')
          if right_foot_index_x < 0 or right_foot_index_x >= img_width2 or right_foot_index_y < 0 or right_foot_index_y >= img_height2:
            continue
          right_foot_index_x += x1m
          right_foot_index_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ center of gravity is between the foots - центр тяжести между ступнями
          footgrav_x = int((left_ankle_x+right_ankle_x+left_heel_x+right_heel_x+left_foot_index_x+right_foot_index_x)/6)
          footgrav_y = int((left_ankle_y+right_ankle_y+left_heel_y+right_heel_y+left_foot_index_y+right_foot_index_y)/6)
          # print('[INFO.PersonFallThread] center of gravity is between the foots - центр тяжести между ступнями')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ устанавливаю флаг, что продетектировано падение
          is_alarmN = True
        except:
          # print('[WARNING.PersonFallThread] ошибка извлечения ключевых точек')
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ одного продетектированного падения достачно, для формирования alarm
        #~ остальные сущности person не рассматриваем
        if is_alarmN:
          break
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if not is_alarmN:
        self.frame_ready = False
        continue
      # print('[INFO.PersonFallThread] --->Alarm1')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем отклонение по пороговому углу
      if is_alarmN:
        #~ media pipe подтвердило падение
        #~ переходим к локальной смещеной системе координат - центр центра тяжести тела
        #~ green -> ground -> earth 
        green_x = footgrav_x - bodygrav_x
        green_y = bodygrav_y - footgrav_y
        green_azim = self.calc_azim(green_x, green_y)
        # print(f'[INFO.PersonFallThread] green_x: {green_x}, green_y: {green_y}, green_azim: {green_azim}')
        azim1 = 180 - self.footgrav
        azim2 = 180 + self.footgrav 
        # print(f'[INFO.PersonFallThread] green_azim: {green_azim}, azim: {azim1}, azim2: {azim2}')
        is_alarmN = False
        if green_azim < azim1 or green_azim > azim2:
          is_alarmN = True
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if not is_alarmN:
        self.frame_ready = False
        continue
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
      # print('[INFO.PersonFallThread] --->Alarm2')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ это как минимум второй кадр по этому событию
      #~ делаем необходимые проверки, чтобы понять это действительно аларм на заданных настройках или нет
      alarmN = self.cam_dict[self.cam_id]
      #~ вычисляем разницу между алармами
      alarm_datetime1 = alarmN[2]
      delta_alarm_sec1 = (alarm_datetimeN - alarm_datetime1).total_seconds()
      delta_alarm_secN = (alarm_datetimeN - alarmN[1]).total_seconds()
      # print(f'[INFO.PersonFallThread] alarmN: {alarmN}')
      # print(f'[INFO.PersonFallThread]   alarmN[0]: {alarmN[0]}')
      # print(f'[INFO.PersonFallThread]   alarmN[1]: {alarmN[1]}')
      # print(f'[INFO.PersonFallThread]   alarm_datetime1: {alarm_datetime1}')
      # print(f'[INFO.PersonFallThread]   delta_alarm_secN: {delta_alarm_secN}, delta_alarm_sec1: {delta_alarm_sec1}')
      if delta_alarm_sec1 < self.siren_time:
        #~ порог от предыдущего аларма не превышен
        self.frame_ready = False
        continue
      # print('[INFO.PersonFallThread] --->Alarm3')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем прервана псевдопоследовательности кадров алармов или нет
      if delta_alarm_secN > self.alarm_time:
        #~ псевдопоследовательности прервана - инициализируем ожидание нового аларма
        self.cam_dict[self.cam_id] = (1,alarm_datetimeN,alarm_datetime1)
        self.frame_ready = False
        continue
      # print('[INFO.PersonFallThread] --->Alarm4')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сравниваем число последовательных алармов, с заданным порогом
      alarm_countN = alarmN[0]+1
      # print(f'[INFO.PersonFallThread]    alarm_count2: {alarm_countN}')
      if alarm_countN < self.alarm_count:
        #~ требуемого значения по порогу еще не достигли - продолжаем
        #~ накапливать последовательные алармы
        self.cam_dict[self.cam_id] = (alarm_countN,alarm_datetimeN,alarm_datetime1)
        self.frame_ready = False
        continue
      # print('[INFO.PersonFallThread] --->Alarm5')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ аларм состоялся
      self.cam_dict[self.cam_id] = (1,alarm_datetimeN,alarm_datetimeN)
      print(f'[INFO.PersonFallThread] ===>Alarm: {self.classes_lst1[self.class_id]}!!! {alarm_datetimeN.strftime("%Y.%m.%d %H:%M:%S")}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняю изображение кадра с алармом на диск
      alarm_dir = os.path.join(self.rdir, alarm_datetimeN.strftime("%Y%m%d"))
      if not os.path.exists(alarm_dir):
        os.makedirs(alarm_dir)
      feature_color = (0,0,255)
      dot_color = (255, 0, 255)
      #~ yolo bbox
      cv2.rectangle(self.frame, (x1,y1), (x2,y2), (0,0,255), 2)
      feature_conf_lbl = f'{round(feature_conf, 2)}'
      cv2.putText(self.frame, feature_conf_lbl, (x1+2,y1+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
      #~ media pipe узлы
      cv2.circle(self.frame, (left_hip_x, left_hip_y), 5, dot_color, -1)
      cv2.circle(self.frame, (right_hip_x, right_hip_y), 5, dot_color, -1)
      cv2.circle(self.frame, (bodygrav_x, bodygrav_y), 5, feature_color, -1)
      cv2.circle(self.frame, (left_ankle_x, left_ankle_y), 5, dot_color, -1)
      cv2.circle(self.frame, (right_ankle_x, right_ankle_y), 5, dot_color, -1)
      cv2.circle(self.frame, (left_heel_x, left_heel_y), 5, dot_color, -1)
      cv2.circle(self.frame, (right_heel_x, right_heel_y), 5, dot_color, -1)
      cv2.circle(self.frame, (left_foot_index_x, left_foot_index_y), 5, dot_color, -1)
      cv2.circle(self.frame, (right_foot_index_x, right_foot_index_y), 5, dot_color, -1)
      cv2.circle(self.frame, (footgrav_x, footgrav_y), 5, feature_color, -1)
      cv2.circle(self.frame, (bodygrav_x, y2), 5, feature_color, -1)
      cv2.line(self.frame, (bodygrav_x, bodygrav_y), (footgrav_x, footgrav_y), feature_color, 2)
      cv2.line(self.frame, (bodygrav_x, bodygrav_y), (bodygrav_x, y2), feature_color, 2)
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
        print(f'[INFO.PersonFallThread]   telebot response code: {resp.status_code}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ выставляем флаг, что отработали это изображение
      self.frame_ready = False
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ поток ожидания и обработки изображений завершен
    print('[INFO.PersonFallThread] run: finish!')