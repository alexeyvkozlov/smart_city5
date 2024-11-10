#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_checker.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
from ultralytics import YOLO
import cv2
import numpy as np
import time

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class YoloChecker:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self,
               src_dir1: str,
               yoloimg_wh1: int,
               weights_fname1: str,
               class_labels_lst1: list[str],
               feature_confidence1: float,
               rep_img_width2: int, rep_img_height2: int,
               is_test_img2: bool,
               dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.yoloimg_wh1 = yoloimg_wh1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.weights_fname1 = weights_fname1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.classes_lst1 = []
    for i in range(len(class_labels_lst1)):
      self.classes_lst1.append(class_labels_lst1[i])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.feature_conf1 = feature_confidence1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.rep_img_width2 = rep_img_width2
    self.rep_img_height2 = rep_img_height2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.is_test_img2 = is_test_img2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.yoloimg_wh1: {self.yoloimg_wh1}')
    print(f'[INFO] self.weights_fname1: `{self.weights_fname1}`')
    print(f'[INFO] self.classes_lst1: len: {len(self.classes_lst1)}, {self.classes_lst1}')
    print(f'[INFO] self.feature_conf1: {round(self.feature_conf1,2)}')
    print(f'[INFO] self.rep_img_width2: {self.rep_img_width2}')
    print(f'[INFO] self.rep_img_height2: {self.rep_img_height2}')
    print(f'[INFO] self.is_test_img2: {self.is_test_img2}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def format_execution_time(self, execution_time):
    if execution_time < 1:
      return f"{execution_time:.3f} sec"
    #~~~~~~~~~~~~~~~~~~~~~~~~
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    if execution_time < 60:
      return f"{seconds}.{int((execution_time % 1) * 1000):03d} sec"
    elif execution_time < 3600:
      return f"{minutes} min {seconds:02d} sec"
    else:
      return f"{hours} h {minutes:02d} min {seconds:02d} sec"

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def format_execution_time2(self, execution_time):
    return f"{execution_time:.3f}"

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def image_check(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 0: "car-accident" - "ДТП - столкновение легковых автомобилей"
    #~ 1: "non-car-accident" - `не ДТП - столкновение легковых автомобилей"
    class_id0 = 0
    class_id1 = 1
    #~ цвет продетектированной сущности
    feature_color0 = (0, 0, 255)
    feature_color1 = (255, 0, 0)
    #~ указываем размер изображения, на котором была обучена нейронка  
    #~ первое значение — это ширина,
    #~ второе значение — это высота.
    yotarget_size=(self.yoloimg_wh1, self.yoloimg_wh1)
    #~ размер кадра для отчетного документа
    reptarget_size=(self.rep_img_width2, self.rep_img_height2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ если обрабатываем тестовые изображения, то необходимо добавить поддиректорию images
    #~~~~~~~~~~~~~~~~~~~~~~~~
    src_img_dir1 = os.path.join(self.src_dir1,'images')
    src_lbl_dir1 = os.path.join(self.src_dir1,'labels')
    # print(f'[INFO] src_img_dir1: `{src_img_dir1}`')
    # print(f'[INFO] src_lbl_dir1: `{src_lbl_dir1}`')
    if not self.is_test_img2:
      src_img_dir1 = self.src_dir1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst = self.dir_filer.get_image_list(src_img_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ инициализируем файл-отчет для записи времени детектирования по кадру-изображению
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ et -> elapsed-time report-file
    et_fname = os.path.join(self.dst_dir2, 'elapsed_time.txt')
    print(f'[INFO] elapsed-time report-file: `{et_fname}`')
    #~ elapsed-time report-file
    et_file = open(et_fname, 'w', encoding='utf-8')
    et_line = '# file name, elapsed time (sec)\n'
    et_file.write(et_line)
    et_line = '#'+'-'*64+'\n'
    et_file.write(et_line)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ YOLOv8 model on custom dataset
    #~ load a model
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # model = YOLO('yolov8m.pt')
    model = YOLO(self.weights_fname1)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ число кадров с продетектированными сущностями
    fcounter = 0
    #~ общее затраченное время на выполнение
    total_elapsed_time = 0.0
    #~ побежали по изображениям в списке
    for i in range(img_lst_len):
      print('~'*70)
      print(f'[INFO] {i}->{img_lst_len-1}: `{img_lst[i]}`')
      img_fname1 = os.path.join(src_img_dir1, img_lst[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      lbl_fname1 = os.path.join(src_lbl_dir1, base_fname1 + '.txt')
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      # print(f'[INFO]  lbl_fname1: `{lbl_fname1}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем изображение на корректность
      frame = cv2.imread(img_fname1)
      img_width1 = 0
      img_height1 = 0
      try:
        img_width1 = frame.shape[1]
        img_height1 = frame.shape[0]
      except:
        print(f'[WARNING] corrupted image: `{img_fname1}`')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ делаю копию оригинального изображения
      frame_origin = frame.copy()
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ масштабирование/сжатие изображения до размеров 640x640
      #~ и предсказание модели
      #~ в любом случае сжимаем, так как видео-камер с разрешение 640x640 не бывает
      #~ а изображение скорее всего с видеокамеры
      #~~~~~~~~~~~~~~~~~~~~~~~~
      frame640 = cv2.resize(frame, yotarget_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ фиксируем время старта обработки изображение-кадра
      #~~~~~~~~~~~~~~~~~~~~~~~~
      start_time = time.time()
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ детектируем сущности в кадре-изображении
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ предсказание модели
      # results = model(frame640)[0]
      # yodets = model(frame640, imgsz=640, verbose=True)[0]
      yodets = model(frame640, imgsz=self.yoloimg_wh1, verbose=True)[0]
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ фиксируем время финиша обработки изображение-кадра
      #~~~~~~~~~~~~~~~~~~~~~~~~
      end_time = time.time()
      elapsed_time = end_time - start_time
      total_elapsed_time += elapsed_time
      elapsed_time_str = self.format_execution_time2(elapsed_time)
      print(f'[INFO] elapsed-time: `{elapsed_time}`, total-elapsed-time: `{total_elapsed_time}`, elapsed-time: `{elapsed_time_str}`')
      et_line = f'{base_fname1}, {elapsed_time_str}\n'
      et_file.write(et_line)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ f - feature frame
      #~ для сохранения отчетного видео и кадров все кадры поджимаем
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if img_width1 != self.rep_img_width2 or img_height1 != self.rep_img_height2:
        frame = cv2.resize(frame, reptarget_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      is_fdet = False
      for yodet in yodets.boxes.data.tolist():
        yox1, yoy1, yox2, yoy2, yoconf, yoclass_id = yodet
        # print(f'[INFO]  yox1: {yox1}, yoy1: {yoy1}, yox2: {yox2}, yoy2: {yoy2}')
        feature_id = int(yoclass_id)
        # print(f'[INFO]  yoclass_id: {yoclass_id}, class_id: {class_id}, feature_id: {feature_id}')
        if not feature_id == class_id0:
          continue
        # print(f'[INFO]  yoconf: {yoconf}, self.feature_conf1: {self.feature_conf1}')
        if yoconf < self.feature_conf1:
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        if not is_fdet:
          is_fdet = True
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ отрисовываем продетектированную сущность, если уверенность определения больше указанной  
        x_min = int(self.rep_img_width2*yox1/self.yoloimg_wh1)
        y_min = int(self.rep_img_height2*yoy1/self.yoloimg_wh1)
        x_max = int(self.rep_img_width2*yox2/self.yoloimg_wh1)
        y_max = int(self.rep_img_height2*yoy2/self.yoloimg_wh1)
        # print(f'[INFO]  x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}')
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), feature_color0, 2)
        # feature_lbl = f'{self.class_labels_lst1[class_id]}: {round(yoconf, 2)}'
        # cv2.putText(frame, feature_lbl, (x_min+3, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 2)
        # cv2.putText(frame, feature_lbl, (x_min+3, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        #~ подписываю вероятность детектирования
        feature_conf_lbl = f'{round(yoconf, 2)}'
        # cv2.putText(frame, feature_conf_lbl, (x_min+2,y_min+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, feature_conf_lbl, (x_min+2,y_min+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if is_fdet:
        fcounter += 1
        #~ и подписываю вероятность детектирования
        pred_lbl = f'predict: {self.classes_lst1[class_id0]}'
        feature_color = feature_color0
        rect_width = 208
      else:
        pred_lbl = f'predict: {self.classes_lst1[class_id1]}'
        feature_color = feature_color1
        rect_width = 258
      cv2.rectangle(frame, (0,0), (rect_width,25), (255,255,255), -1)
      cv2.putText(frame, pred_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ в любом случае сохраняем изображение с продетектированными сущностями,
      #~ даже если не было детекции
      # img_fname2 = os.path.join(self.dst_dir2, img_lst[i])
      img_fname2 = os.path.join(self.dst_dir2, base_fname1+'.png')
      # print(f'[INFO]  img_fname2: `{img_fname2}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отрисовываем оригинальную разметку
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if self.is_test_img2:
        #~ читаем файл по строкам
        lines1 = []
        input_file = open(lbl_fname1, 'r', encoding='utf-8')
        lines1 = input_file.readlines()
        input_file.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ оставляем только не пустые строки
        is_fdet0 = False
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
          #~ f - feature
          try:
            #~ преобразование строки в числа
            fclass_id = int(fields5[0])
            fx_center = float(fields5[1])
            fy_center = float(fields5[2])
            fwidth = float(fields5[3])
            fheight = float(fields5[4])
          except ValueError as e:
            print(f'[ERROR] произошла ошибка при преобразовании строки в число: `{line2}`, : {e}')
            continue
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ оставляем только определенный класс объектов
          #~ 0: "car-accident" - "ДТП - столкновение легковых автомобилей"
          #~ 1: "non-car-accident" - `не ДТП - столкновение легковых автомобилей"
          if not class_id0 == fclass_id:
            continue
          #~~~~~~~~~~~~~~~~~~~~~~~~
          #~ в файле разметки присутствует интересующая сущность  
          is_fdet0 = True
          fwidth05 = fwidth/2.0
          fheight05 = fheight/2.0
          # print(f'[INFO] fx_center: {fx_center}, fy_center: {fy_center}, fwidth: {fwidth}, fheight: {fheight}, fwidth05: {fwidth05}, fheight05: {fheight05}')
          # print(f'[INFO] img_width1: {img_width1}, img_height1: {img_height1}')
          fx1 = int((fx_center - fwidth05)*img_width1)
          fx2 = int((fx_center + fwidth05)*img_width1)
          fy1 = int((fy_center - fheight05)*img_height1)
          fy2 = int((fy_center + fheight05)*img_height1)
          # print(f'[INFO] fclass_id: {fclass_id}, fx1: {fx1}, fy1: {fy1}, fx2: {fx2}, fy2: {fy2}')
          cv2.rectangle(frame_origin, (fx1, fy1), (fx2, fy2), feature_color0, 2)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ подписываем изображение
        if is_fdet0:
          class_lbl = f'original: {self.classes_lst1[class_id0]}'
          feature_color = feature_color0
          rect_width = 212
        else:
          class_lbl = f'original: {self.classes_lst1[class_id1]}'
          feature_color = feature_color1
          rect_width = 262
        cv2.rectangle(frame_origin, (0,0), (rect_width,25), (255,255,255), -1)
        cv2.putText(frame_origin, class_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ объединяем два изображения в одно
        #~ создание белого холста
        merge_img = np.ones((img_height1, img_width1 * 2 + 10, 3), dtype='uint8') * 255
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ добавляем первое изображения на холст
        #~ m - merge
        mx1 = 0
        my1 = 0
        merge_img[my1:my1+img_height1, mx1:mx1+img_width1, :] = frame_origin
        #~ добавляем второе изображения на холст
        mx2 = mx1+img_width1+10
        my2 = 0
        merge_img[my2:my2+img_height1, mx2:mx2+img_width1, :] = frame
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ сохраняем результаты
        cv2.imwrite(img_fname2, merge_img)
      else:
        cv2.imwrite(img_fname2, frame)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ завершили чтение изображений по списку
    print('='*70)
    print(f'[INFO] число обработанных изображений: {img_lst_len}')
    print(f'[INFO] число изображений с продетектированными сущностями: {fcounter}')
    elapsed_time_str = self.format_execution_time(total_elapsed_time)
    print(f'[INFO] общее время затраченное на детектирование сущностей: {elapsed_time_str}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ закрываем файл с отметками затраченного времени на детектирование сущностей
    #~~~~~~~~~~~~~~~~~~~~~~~~
    et_line = '#'+'='*64+'\n'
    et_file.write(et_line)
    et_line = f'# число обработанных изображений: {img_lst_len}\n'
    et_file.write(et_line)
    et_line = f'# число изображений с продетектированными сущностями: {fcounter}\n'
    et_file.write(et_line)
    elapsed_time_str = self.format_execution_time(total_elapsed_time)
    et_line = f'# общее время затраченное на детектирование сущностей: {elapsed_time_str}'
    et_file.write(et_line)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    et_file.close()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloChecker ver.2024.10.15')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями для детектирования объектов по рассчитаным весам в YOLO
  # src_dir1 = 'c:/my_campy/smart_city/weights_car_accident/dataset_car_accident4_test/test'
  src_dir1 = 'c:/my_campy/smart_city/weights_car_accident/accident7-frames'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями результатами детектирования-отрисованными сущностями
  # dst_dir2 = 'c:/my_campy/smart_city/weights_car_accident/20241031_small_101epochs_optimizer_RMSProp/dataset_car_accident4_test'
  dst_dir2 = 'c:/my_campy/smart_city/weights_car_accident/20241031_small_101epochs_optimizer_RMSProp/accident7-frames'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ для изображений из папки test - добавляем оригинальную разметку
  #~ True False
  is_test_img2 = False
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ размер изображения при обучении
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ yolo image width,height ширина-высота изображения, на котором обучалась yolo
  yoloimg_wh1 = 640
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ файл с рассчитанными весами
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 262 epochs completed in 1.626 hours.
  #~ Optimizer stripped from runs\detect\train4\weights\last.pt, 22.6MB
  #~ Optimizer stripped from runs\detect\train4\weights\best.pt, 22.5MB
  # weights_fname1 = 'c:/my_campy/smart_city/preparation_utils/runs/detect/train4/weights/last.pt'
  # weights_fname1 = 'c:/my_campy/smart_city/preparation_utils/runs/detect/train4/weights/best.pt'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ model mode
  #~ YOLOv8n -> nano
  #~ YOLOv8s -> small
  #~ YOLOv8m -> medium
  #~ YOLOv8l -> large
  #~ YOLOv8x -> extra large
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 20240916_small_210epoch
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20240916_small_210epochs/train3/weights/best.pt'
  #~ 20240917_small_262epochs
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20240917_small_262epochs/train4/weights/best.pt'
  #~ 20241014_small_300epoch
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241014_small_300epochs/train6/weights/best.pt'
  #~ 20241015_small_401epochs
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241015_small_401epochs/train7/weights/best.pt'
  #~ 20241016_medium_412epochs
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241016_medium_412epochs/train8/weights/best.pt'
  #~ 20241017_large_394epochs
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241017_large_394epochs/train10/weights/best.pt'
  #~ 20241017_nano_500epochs
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241017_nano_500epochs/train9/weights/best.pt'
  #~ 20241023_small_401epochs_optimizer_auto
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241023_small_401epochs_optimizer_auto/train/weights/best.pt'
  #~ 20241024_small_462epochs_optimizer_SGD
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241024_small_462epochs_optimizer_SGD/train2/weights/best.pt'
  #~ 20241026_extra_large_435epochs
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241026_extra_large_435epochs/train3/weights/best.pt'
  #~ 20241028_small_500epochs_optimizer_Adam
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241028_small_500epochs_optimizer_Adam/train4/weights/best.pt'
  #~ 20241029_small_827epochs_optimizer_Adam
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241029_small_827epochs_optimizer_Adam/train5/weights/best.pt'
  #~ 20241030_small_551epochs_optimizer_AdamW
  # weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241030_small_551epochs_optimizer_AdamW/train6/weights/best.pt'
  #~ 20241031_small_101epochs_optimizer_RMSProp
  weights_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241031_small_101epochs_optimizer_RMSProp/train7/weights/best.pt'


  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список классов для детектирования
  #~~~~~~~~~~~~~~~~~~~~~~~~
  class_labels_lst1 = ['car-accident', 'non-car-accident']
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ порог уверенности, не ниже которой будут продетектированы сущности
  #~ 0.2, 0.3, 0.4, 0.5, 0.7, 0.9
  #~ 0.25 0.5 0.8 0.85 0.92
  #~~~~~~~~~~~~~~~~~~~~~~~~
  feature_confidence1 = 0.8
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ rep - report
  #~ размер изображения для отчетного документа 
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 960x540 640x640
  rep_img_width2 = 960
  rep_img_height2 = 540
  if is_test_img2:
    rep_img_width2 = 640
    rep_img_height2 = 640
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ запускаем на выполнение
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychr_obj = YoloChecker(src_dir1,
                         yoloimg_wh1,
                         weights_fname1,
                         class_labels_lst1,
                         feature_confidence1,
                         rep_img_width2, rep_img_height2,
                         is_test_img2,
                         dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ychr_obj.image_check()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  ychr_obj.timer_obj.elapsed_time()
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # cd c:\my_campy
  # .\camenv8\Scripts\activate
  # cd c:\my_campy\smart_city\preparation_utils
  # python yolo_checker.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~