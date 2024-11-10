#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python media_pipe_checker.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
from ultralytics import YOLO
import cv2
import mediapipe as mp
import math

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MediaPipeChecker:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, model_mode1: str, yolo_confidence1: float, person_ratio1: float, pipe_confidence1: float, footgrav_deg1: int, class_labels_lst1: list[str], dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
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
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    self.model_name1 = f'yolov8{model_mode2}.pt'
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.yolo_conf1 = yolo_confidence1
    self.pers_ratio1 = person_ratio1
    self.pipe_conf1 = pipe_confidence1
    self.footgrav_deg1 = footgrav_deg1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.classes_lst1 = []
    for i in range(len(class_labels_lst1)):
      self.classes_lst1.append(class_labels_lst1[i])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.model_name1: `{self.model_name1}`')
    print(f'[INFO] self.yolo_conf1: {self.yolo_conf1}')
    print(f'[INFO] self.pers_ratio1: {self.pers_ratio1}')
    print(f'[INFO] self.pipe_conf1: {self.pipe_conf1}')
    print(f'[INFO] self.footgrav_deg1: {self.footgrav_deg1}')
    print(f'[INFO] self.classes_lst1: len: {len(self.classes_lst1)}, {self.classes_lst1}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

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
  def image_check(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ устанавливаю минимальный размер изображения, с которым буду работать
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_min_size = 16
    #~ детектируем только класс 'person' -> COCO:
    #~ 0: person
    #~ 1: bicycle
    #~ 2: car
    #~ 3: motorcycle
    class_id = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ формируем список изображений
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ создаем объект YOLO 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model = YOLO(self.model_name1)
    print(f'[INFO] YOLO model: {self.model_name1}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ с параметрами, объектами определились,
    #~ побежали по изображениям в списке
    for i in range(img_lst_len):
      print('~'*70)
      print(f'[INFO] {i}->{img_lst_len-1}: `{img_lst[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем изображение на корректность
      img1 = cv2.imread(img_fname1)
      img_width1 = 0
      img_height1 = 0
      try:
        img_width1 = img1.shape[1]
        img_height1 = img1.shape[0]
      except:
        print(f'[WARNING] corrupted image: `{img_fname1}`')
        continue
      if img_width1 < img_min_size or img_height1 < img_min_size:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # #~ делаю копию изображения для отрисовки вспомогателльной графики
      # img2 = img1.copy()
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ предсказание модели
      yodets = model(img1, imgsz=640, verbose=True)[0]
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ флаг, что yolo продетектировала падающего человека
      is_fall = False
      #~ предсказание модели
      for yodet in yodets.boxes.data.tolist():
        yox1, yoy1, yox2, yoy2, yoconf, yoclass_id = yodet
        # print(f'[INFO]  yox1: {yox1}, yoy1: {yoy1}, yox2: {yox2}, yoy2: {yoy2}')
        feature_id = int(yoclass_id)
        # print(f'[INFO]  yoclass_id: {yoclass_id}, class_id: {class_id}, feature_id: {feature_id}')
        if not feature_id == class_id:
          continue
        # print(f'[INFO]  yoconf: {yoconf}, self.feature_conf1: {self.feature_conf1}')
        if yoconf < self.yolo_conf1:
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        person_width = yox2 - yox1
        person_height = yoy2 - yoy1
        person_ratio = person_width/person_height
        # person_ratio_lbl = f'{round(person_ratio, 2)}'
        print(f'[INFO] person: width: {round(person_width, 2)}, height: {round(person_height, 2)}, ratio: {round(person_ratio, 2)}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        # #~ координаты продетектированного feature -> person
        x1 = int(yox1)
        y1 = int(yoy1)
        x2 = int(yox2)
        y2 = int(yoy2)
        # cv2.rectangle(img2,(x1,y1),(x2,y2),(0,0,255),2)
        # cv2.putText(img2, person_ratio_lbl, (x1+2,y1+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ сравниваем отношение ширины к высоте person, если порог превышен, фиксируем как падение 
        if person_ratio < self.pers_ratio1:
          print('[INFO] YOLO Fall False')
          continue
        print('[INFO] YOLO Fall True')
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
        if x2m >= img_width1:
          x2m = img_width1-1
        if y1m < 0:
          y1m = 0
        if y2m > img_height1:
          y2m = img_height1-1
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ выделение фрагмента
        img2m = img1[y1m:y2m, x1m:x2m]
        img_width2 = img2m.shape[1]
        img_height2 = img2m.shape[0]
        if img_width2 < img_min_size or img_height2 < img_min_size:
          continue
        print(f'[INFO] ===>img2m: width: {img_width2}, height: {img_height2}')
        print(f'[INFO]   x1m: {x1m}, x2m: {x2m}, y1m: {y1m}, y2m: {y2m}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        # img_fname2 = os.path.join(self.dst_dir2, base_fname1+'-01.png')
        # cv2.imwrite(img_fname2, img1)
        # img_fname2 = os.path.join(self.dst_dir2, base_fname1+'-02.png')
        # cv2.imwrite(img_fname2, img2m)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        try:
          #~ рассчитываем скелет человеческого тела с помощью Media Pipe - Pose Estimation
          mp_pose = mp.solutions.pose
          npose = mp_pose.Pose(min_detection_confidence=self.pipe_conf1)
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
          print(f'[INFO]   23-1 - left hip - левое бедро: x: {left_hip_x}, y: {left_hip_y}')
          if left_hip_x < 0 or left_hip_x >= img_width2 or left_hip_y < 0 or left_hip_y >= img_height2:
            continue
          left_hip_x += x1m
          left_hip_y += y1m
          print(f'[INFO]     23-2 - left hip - левое бедро: x: {left_hip_x}, y: {left_hip_y}')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_hip_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_HIP].x * img_width2)
          right_hip_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_HIP].y * img_height2)
          print(f'[INFO]   24-1 - right hip - правое бедро: x: {right_hip_x}, y: {right_hip_y}')
          if right_hip_x < 0 or right_hip_x >= img_width2 or right_hip_y < 0 or right_hip_y >= img_height2:
            continue
          right_hip_x += x1m
          right_hip_y += y1m
          print(f'[INFO]     24-2 - right hip - правое бедро: x: {right_hip_x}, y: {right_hip_y}')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ center of gravity of whole body - центр тяжести всего тела
          bodygrav_x = int((left_hip_x+right_hip_x)/2)
          bodygrav_y = int((left_hip_y+right_hip_y)/2)
          print(f'[INFO]     center of gravity of whole body - центр тяжести всего тела: x: {bodygrav_x}, y: {bodygrav_y}')
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
          print('[INFO] 27 - left ankle - левая лодыжка')
          if left_ankle_x < 0 or left_ankle_x >= img_width2 or left_ankle_y < 0 or left_ankle_y >= img_height2:
            continue
          left_ankle_x += x1m
          left_ankle_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_ankle_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img_width2)
          right_ankle_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img_height2)
          print('[INFO] 28 - right ankle - правая лодыжка')
          if right_ankle_x < 0 or right_ankle_x >= img_width2 or right_ankle_y < 0 or right_ankle_y >= img_height2:
            continue
          right_ankle_x += x1m
          right_ankle_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          left_heel_x = int(human_dots[mp_pose.PoseLandmark.LEFT_HEEL].x * img_width2)
          left_heel_y = int(human_dots[mp_pose.PoseLandmark.LEFT_HEEL].y * img_height2)
          print('[INFO] 29 - left heel - левая пятка')
          if left_heel_x < 0 or left_heel_x >= img_width2 or left_heel_y < 0 or left_heel_y >= img_height2:
            continue
          left_heel_x += x1m
          left_heel_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_heel_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_HEEL].x * img_width2)
          right_heel_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_HEEL].y * img_height2)
          print('[INFO] 30 - right heel - правая пятка')
          if right_heel_x < 0 or right_heel_x >= img_width2 or right_heel_y < 0 or right_heel_y >= img_height2:
            continue
          right_heel_x += x1m
          right_heel_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          left_foot_index_x = int(human_dots[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * img_width2)
          left_foot_index_y = int(human_dots[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * img_height2)
          print('[INFO] 31 - left foot index - указательный палец левой стопы')
          if left_foot_index_x < 0 or left_foot_index_x >= img_width2 or left_foot_index_y < 0 or left_foot_index_y >= img_height2:
            continue
          left_foot_index_x += x1m
          left_foot_index_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          right_foot_index_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * img_width2)
          right_foot_index_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * img_height2)
          print('[INFO] 31 - left foot index - указательный палец правой стопы')
          if right_foot_index_x < 0 or right_foot_index_x >= img_width2 or right_foot_index_y < 0 or right_foot_index_y >= img_height2:
            continue
          right_foot_index_x += x1m
          right_foot_index_y += y1m
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ center of gravity is between the foots - центр тяжести между ступнями
          footgrav_x = int((left_ankle_x+right_ankle_x+left_heel_x+right_heel_x+left_foot_index_x+right_foot_index_x)/6)
          footgrav_y = int((left_ankle_y+right_ankle_y+left_heel_y+right_heel_y+left_foot_index_y+right_foot_index_y)/6)
          print('[INFO] center of gravity is between the foots - центр тяжести между ступнями')
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ устанавливаю флаг, что продетектировано падение
          is_fall = True
        except:
          print('[ERROR] ошибка извлечения ключевых точек') 
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ одного продетектированного падения достачно, для формирования alarm
        #~ остальные сущности person не рассматриваем
        if is_fall:
          break
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if is_fall:
        #~ media pipe подтвердило падение
        #~ переходим к локальной смещеной системе координат - центр центра тяжести тела
        #~ green -> ground -> earth 
        green_x = footgrav_x - bodygrav_x
        green_y = bodygrav_y - footgrav_y
        green_azim = self.calc_azim(green_x, green_y)
        print(f'[INFO] green_x: {green_x}, green_y: {green_y}, green_azim: {green_azim}')
        azim1 = 180 - self.footgrav_deg1
        azim2 = 180 + self.footgrav_deg1 
        print(f'[INFO] green_azim: {green_azim}, azim: {azim1}, azim2: {azim2}')
        is_fall = False
        if green_azim < azim1 or green_azim > azim2:
          is_fall = True
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ и настраиваем цвета для отрисовки
      print(f'[INFO] Media Pipe Fall: {is_fall}')
      pred_inx = 1
      feature_color = (255, 0, 0)
      if is_fall:
        pred_inx = 0
        feature_color = (0, 0, 255)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~ отрисовываю поясняющую графику
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ yolo bbox
        cv2.rectangle(img1,(x1,y1),(x2,y2),feature_color,2)
        #~ media pipe узлы
        dot_color = (255, 0, 255)
        cv2.circle(img1, (left_hip_x, left_hip_y), 5, dot_color, -1)
        cv2.circle(img1, (right_hip_x, right_hip_y), 5, dot_color, -1)
        cv2.circle(img1, (bodygrav_x, bodygrav_y), 5, feature_color, -1)
        cv2.circle(img1, (left_ankle_x, left_ankle_y), 5, dot_color, -1)
        cv2.circle(img1, (right_ankle_x, right_ankle_y), 5, dot_color, -1)
        cv2.circle(img1, (left_heel_x, left_heel_y), 5, dot_color, -1)
        cv2.circle(img1, (right_heel_x, right_heel_y), 5, dot_color, -1)
        cv2.circle(img1, (left_foot_index_x, left_foot_index_y), 5, dot_color, -1)
        cv2.circle(img1, (right_foot_index_x, right_foot_index_y), 5, dot_color, -1)
        cv2.circle(img1, (footgrav_x, footgrav_y), 5, feature_color, -1)
        cv2.circle(img1, (bodygrav_x, y2), 5, feature_color, -1)
        cv2.line(img1, (bodygrav_x, bodygrav_y), (footgrav_x, footgrav_y), feature_color, 2)
        cv2.line(img1, (bodygrav_x, bodygrav_y), (bodygrav_x, y2), feature_color, 2)


      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняем продетектированное изображение
      img_fname2 = os.path.join(self.dst_dir2, base_fname1+'.png')
      pred_lbl = f'predict: {self.classes_lst1[pred_inx]}'
      cv2.rectangle(img1, (0,0), (240,25), (255,255,255), -1)
      cv2.putText(img1, pred_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
      cv2.imwrite(img_fname2, img1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ завершили чтение изображений по списку
    print('='*70)
    print(f'[INFO] обработано {img_lst_len} изображений')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] MediaPipeChecker ver.2024.09.27')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями для детектирования объектов
  src_dir1 = 'c:/my_campy/smart_city/smart_city_check_photo_video/final_video/i80-fall20241004-1877'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ model mode
  #~ YOLOv8n -> nano
  #~ YOLOv8s -> small
  #~ YOLOv8m -> medium
  #~ YOLOv8l -> large
  #~ YOLOv8x -> extra large
  # model_mode1 = 'nano'
  # model_mode1 = 'small'
  # model_mode1 = 'medium'
  model_mode1 = 'medium'
  # model_mode1 = 'extra large'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ порог детектирования person в YOLO
  #~ 0.2, 0.3, 0.4, 0.5, 0.7
  yolo_confidence1 = 0.2
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ опытным путем установил, что сжатие w/h для детекции падающего человека должно превышать 0.65
  person_ratio1 = 0.65
  #~~~~~~~~~~~~~~~~~~~~~~~~
  pipe_confidence1 = 0.5
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отклонение в градусах центральной точки между ступнями от вертикальной оси
  footgrav_deg1 = 36
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список классов для детектирования
  class_labels_lst1 = ['person-fall', 'non-person-fall']
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями результатами детектирования-отрисованными сущностями
  dst_dir2 = 'c:/my_campy/smart_city/smart_city_check_photo_video/final_video/i80-fall20241004-1877___fall'


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python media_pipe_checker.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  #~~~~~~~~~~~~~~~~~~~~~~~~
  mpchr_obj = MediaPipeChecker(src_dir1, model_mode1, yolo_confidence1, person_ratio1, pipe_confidence1, footgrav_deg1, class_labels_lst1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  mpchr_obj.image_check()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  mpchr_obj.timer_obj.elapsed_time()