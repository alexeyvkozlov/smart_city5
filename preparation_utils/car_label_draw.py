#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python car_label_draw.py
#~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cd /home/akozlov/my_campy
# source camenv8/bin/activate
# cd /home/akozlov/my_campy/smart_city/preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python car_label_draw.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import uuid
from ultralytics import YOLO
import cv2
import random

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CarLabelDraw:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, model_mode1: str, feature_confidence1: float, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
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
    self.model_name = f'yolov8{model_mode2}.pt'
    print(f'[INFO] self.model_name: `{self.model_name}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.feature_conf1 = feature_confidence1
    print(f'[INFO] self.feature_conf1: {round(self.feature_conf1,2)}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    self.dst_dir2 = dst_dir2
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # def get_feature_label_lst(self, lbl_fname1: str) -> list[str]:
  def get_feature_label_arr(self, lbl_fname1: str, img_width: int, img_height: int):
    bbox_arr0 = []
    if not self.dir_filer.file_exists(lbl_fname1):
      return bbox_arr0
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ читаем файл по строкам
    lines1 = []
    input_file = open(lbl_fname1, 'r', encoding='utf-8')
    lines1 = input_file.readlines()
    input_file.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ оставляем только не пустые строки
    for line1 in lines1:
      #~ удаляем пробелы в начале и конце строки
      line2 = line1.strip()
      if len(line2) < 1:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fields5 = line2.split()
      if not 5 == len(fields5):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      try:
        #~ преобразование строки в числа
        class_id = int(fields5[0])
        x_center = float(fields5[1])
        y_center = float(fields5[2])
        width = float(fields5[3])
        height = float(fields5[4])
      except ValueError as e:
        print(f'[ERROR] произошла ошибка при преобразовании строки в число: `{line2}`, : {e}')
        continue

      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ оставляем только определенный класс объектов
      #~ 0, 1, 2, 6
      if not 0 == class_id:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ если необходимо вернуть оринигальную разметку
      # lines2.append(line2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ возвращаем скорректированную разметку
      width05 = width/2.0
      height05 = height/2.0
      x1 = (x_center - width05)*img_width
      x2 = (x_center + width05)*img_width
      y1 = (y_center - height05)*img_height
      y2 = (y_center + height05)*img_height
      #~ mm-min-max
      #~                x1 x2 y1 y2 x1mm x2mm y1mm y2mm
      #~                0  1  2  3  4    5    6    7
      bbox_arr0.append([x1,x2,y1,y2,-1.0,-1.0,-1.0,-1.0])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    return bbox_arr0

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def write_feature_label_file(self, lbl_fname2: str, img_width: int, img_height: int, bbox_arr0, bbox_lst1):
    if len(bbox_arr0) < 1 and len(bbox_lst1) < 1:
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~
    label_file = open(lbl_fname2, 'w', encoding='utf-8')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 0: "car-accident" - "ДТП-легковые автомобили"
    bbox_lst_len0 = len(bbox_arr0)
    if bbox_lst_len0 > 0:
      #~  x1 x2 y1 y2 x1mm x2mm y1mm y2mm
      #~  0  1  2  3  4    5    6    7
      #~((x1,x2,y1,y2,-1.0,-1.0,-1.0,-1.0))
      for i in range(bbox_lst_len0):
        if bbox_arr0[i][4] > -1.0:
          ix1mm = bbox_arr0[i][4]
          ix2mm = bbox_arr0[i][5]
          iy1mm = bbox_arr0[i][6]
          iy2mm = bbox_arr0[i][7]
        else:
          ix1mm = bbox_arr0[i][0]
          ix2mm = bbox_arr0[i][1]
          iy1mm = bbox_arr0[i][2]
          iy2mm = bbox_arr0[i][3]
        iwmm = ix2mm - ix1mm
        ihmm = iy2mm - iy1mm
        ixcen = ix1mm + iwmm/2.0
        iycen = iy1mm + ihmm/2.0
        # print(f'[INFO]  ix1mm: {ix1mm}, ix2mm: {ix2mm}, iy1mm: {iy1mm}, iy2mm: {iy2mm}')
        fline0 = f'0 {ixcen/img_width} {iycen/img_height} {iwmm/img_width} {ihmm/img_height}'
        label_file.write(fline0 + '\n')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 1: "non-car-accident" - `не ДТП-легковые автомобили"
    bbox_lst_len1 = len(bbox_lst1)
    if bbox_lst_len1 > 0:
      for i in range(bbox_lst_len1):
        fline1 = f'1 {bbox_lst1[i][0]/img_width} {bbox_lst1[i][1]/img_height} {bbox_lst1[i][2]/img_width} {bbox_lst1[i][3]/img_height}'
        label_file.write(fline1 + '\n')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    label_file.close()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def draw_feature_label_file(self, img_fname2: str, lbl_fname2: str, img_fname3: str):
    img = cv2.imread(img_fname2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    lbl_file = open(lbl_fname2, 'r', encoding='utf-8')
    lbl_lines = lbl_file.readlines()
    lbl_file.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for lbl_line in lbl_lines:
      lbl_line2 = lbl_line.strip()
      fields5 = lbl_line2.split()
      if not 5 == len(fields5):
        continue
      try:
        #~ преобразование строки в числа
        class_id = int(fields5[0])
        x_center = float(fields5[1])
        y_center = float(fields5[2])
        width = float(fields5[3])
        height = float(fields5[4])
      except ValueError as e:
        print(f'[ERROR] произошла ошибка при преобразовании строки в число: `{lbl_line2}`, : {e}')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~
      img_width = 1
      img_height = 1
      try:
        img_width = img.shape[1]
        img_height = img.shape[0]
      except:
        print("[WARNING] Oops!  The image is damaged...")
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      x_min = int((x_center - width/2) * img_width)
      y_min = int((y_center - height/2) * img_height)
      x_max = int((x_center + width/2) * img_width)
      y_max = int((y_center + height/2) * img_height)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      feature_color = (0, 0, 255)
      if 1 == class_id:
        feature_color = (255, 0, 0)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      cv2.rectangle(img, (x_min, y_min), (x_max, y_max), feature_color, 2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ и сохраняем изображение с нарисованными bbox - обведенными сущностями
    cv2.imwrite(img_fname3, img)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def image_draw(self):
    img_lst = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    # random.shuffle(img_lst)
    # print(f'[INFO]   shuffle: img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ директория для изображений с отрисованной разметкой 
    dst_dir3 = os.path.join(self.dst_dir2, 'draw')
    print(f'[INFO] dst_dir3: `{dst_dir3}`')
    self.dir_filer.remove_create_directory(dst_dir3)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ открываем указанную модель YOLO для детектирования 
    model = YOLO(self.model_name)
    print(f'[INFO] start YOLO model: {self.model_name}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    #~ побежали по изображениям в списке
    #~ счетчик кадров -> frame counter
    #~ префикс - для счетчика кадров, для того чтобы потом понятнее было для ручного разбора
    fcounter = 0
    digits = 5
    for i in range(img_lst_len):
      # print('~'*70)
      print(f'[INFO] {i}->{img_lst_len}: `{img_lst[i]}`')
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      lbl_fname1 = os.path.join(self.src_dir1, base_fname1 + '.txt')
      # if not self.dir_filer.file_exists(lbl_fname1):
      #   continue
      img_fname1 = os.path.join(self.src_dir1, img_lst[i])
      #~~~~~~~~~~~~~~~~~~~~~~~~
      prefix_inx = f'f{self.dir_filer.format_counter(fcounter, digits)}-'
      unic_fname = f'{uuid.uuid1()}'
      # unic_fname = base_fname1
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # img_fname2 = os.path.join(self.dst_dir2, unic_fname + suffix_fname1)
      # lbl_fname2 = os.path.join(self.dst_dir2, unic_fname + '.txt')
      # img_fname3 = os.path.join(dst_dir3, unic_fname + suffix_fname1)
      #~~~
      img_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + suffix_fname1)
      lbl_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + '.txt')
      img_fname3 = os.path.join(dst_dir3, prefix_inx+unic_fname + suffix_fname1)
      #~~~
      # print(f'[INFO]   img_fname1: `{img_fname1}`')
      # print(f'[INFO]   lbl_fname1: `{lbl_fname1}`')
      # print(f'[INFO]   img_fname2: `{img_fname2}`')
      # print(f'[INFO]   lbl_fname2: `{lbl_fname2}`')
      # print(f'[INFO]   img_fname3: `{img_fname3}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ детектирую объекты на текущем изображении
      img = cv2.imread(img_fname1)
      img_width = 0
      img_height = 0
      try:
        img_width = img.shape[1]
        img_height = img.shape[0]
      except:
        print(f'[WARNING] corrupted image: {img_fname1}')
        continue
      # print(f'[INFO] img_width: {img_width}, img_height: {img_height}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      bbox_arr0 = self.get_feature_label_arr(lbl_fname1, img_width, img_height)
      bbox_lst_arr0 = len(bbox_arr0)
      # print(f'[INFO] bbox_arr0: len: {bbox_lst_arr0}, `{bbox_arr0}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if bbox_lst_arr0 < 1:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ mm-min-max
      #~  x1 x2 y1 y2 x1mm x2mm y1mm y2mm
      #~  0  1  2  3  4    5    6    7
      #~([x1,x2,y1,y2,-1.0,-1.0,-1.0,-1.0])
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ yolo detections
      bbox_lst1 = []
      #~ coco.txt
      #~  0: person
      #~  1: bicycle
      #~  2: car
      feature_class_id = 2
      # yodets = model(frame, imgsz=640, verbose=True)[0]
      yodets = model(img, imgsz=640, verbose=True)[0]
      for yodet in yodets.boxes.data.tolist():
        yox1, yoy1, yox2, yoy2, yoconf, yoclass_id = yodet
        class_id = int(yoclass_id)
        # print(f'[INFO]  yox1: {yox1}, yoy1: {yoy1}, yox2: {yox2}, yoy2: {yoy2}')
        # print(f'[INFO]  yoclass_id: {yoclass_id}, class_id: {class_id}, feature_class_id: {feature_class_id}')
        #~  2: car
        if not feature_class_id == class_id:
          continue
        # print(f'[INFO]  yoconf: {yoconf}, self.feature_conf1: {self.feature_conf1}')
        if yoconf < self.feature_conf1:
          continue
        #~ f - feature 
        fxcen = (yox1+yox2)/2.0
        fycen = (yoy1+yoy2)/2.0
        fwidth = yox2 - yox1
        fheight = yoy2 - yoy1
        # print(f'[INFO]  fxcen: {fxcen}, fycen: {fycen}, fwidth: {fwidth}, fheight: {fheight}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ только отдельные машины добавляем - без столкновения
        #~ продетектированная машина - это отдельная машина
        # 1: "non-car-accident" - `не ДТП-легковые автомобили"
        # bbox_lst1.append((fxcen,fycen,fwidth,fheight))
        #~~~~~~~~~~~~~~~~~~~~~~~~
        for j in range(bbox_lst_arr0):
          # print(f'[INFO]  j: {j}, bbox_lst_arr0: {bbox_lst_arr0}')
          #~ mm-min-max
          #~  x1 x2 y1 y2 x1mm x2mm y1mm y2mm
          #~  0  1  2  3  4    5    6    7
          #~([x1,x2,y1,y2,-1.0,-1.0,-1.0,-1.0])
          jx1 = bbox_arr0[j][0]
          jx2 = bbox_arr0[j][1]
          jy1 = bbox_arr0[j][2]
          jy2 = bbox_arr0[j][3]
          # print(f'[INFO]  jx1: {jx1}, jx2: {jx2}, jy1: {jy1}, jy2: {jy2}')
          if jx1 < fxcen and fxcen < jx2 and jy1 < fycen and fycen < jy2:
            #~ продетектированная машина находится внутри bbox car-accident
            # 0: "car-accident" - "ДТП-легковые автомобили"
            if bbox_arr0[j][4] < 0:
              #~ это первая рамка, сохраняем ее как стартовую объединенную рамку
              bbox_arr0[j][4] = yox1
              bbox_arr0[j][5] = yox2
              bbox_arr0[j][6] = yoy1
              bbox_arr0[j][7] = yoy2
            else:
              #~ это вторая и далее рамки, поэтому выбираю внешние границы
              if yox1 < bbox_arr0[j][4]:
                bbox_arr0[j][4] = yox1 
              if yox2 > bbox_arr0[j][5]:
                bbox_arr0[j][5] = yox2 
              if yoy1 < bbox_arr0[j][6]:
                bbox_arr0[j][6] = yoy1 
              if yoy2 > bbox_arr0[j][7]:
                bbox_arr0[j][7] = yoy2 
          else:
            #~ продетектированная машина - это отдельная машина
            # 1: "non-car-accident" - `не ДТП-легковые автомобили"
            bbox_lst1.append((fxcen,fycen,fwidth,fheight))
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ итак, имеем два списка: один с подправленными рамками "столкновения", второй - отдельные машины
      #~~~~~~~~~~~~~~~~~~~~~~~~
      car_count = bbox_lst_arr0 + len(bbox_lst1)
      if car_count < 1:
        continue
      #~ записываем в новый файл прочитанную и сформированную разметку 
      self.write_feature_label_file(lbl_fname2, img_width, img_height, bbox_arr0, bbox_lst1)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ копируем файл изображения
      self.dir_filer.copy_file(img_fname1, img_fname2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # #~ 1280x720 -> 640
      # #~ чтение изображения
      # img1280 = cv2.imread(img_fname1)
      # #~ масштабирование изображения до размеров 640x640
      # img640 = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
      # #~ cохранение результата
      # cv2.imwrite(img_fname2, img640)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ и отрисовываем файл-изображение с размеченными и продетектированными сущностями
      self.draw_feature_label_file(img_fname2, lbl_fname2, img_fname3)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      fcounter += 1


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] CarLabelDraw ver.2024.09.12')
  print('~'*70)
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
  model_mode1 = 'large'
  # model_mode1 = 'extra large'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 0.2, 0.3, 0.4, 0.5, 0.7
  feature_confidence1 = 0.5
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ test train valid
  #~ train_valid
  src_dir1 = 'd:/my_campy/smart_city_dataset_car_accident/d24_stage3'
  dst_dir2 = 'd:/my_campy/smart_city_dataset_car_accident/d25_stage3'

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ python car_label_draw.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj = CarLabelDraw(src_dir1, model_mode1, feature_confidence1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj.image_draw()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  car_obj.timer_obj.elapsed_time()
