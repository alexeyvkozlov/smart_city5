#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python keras_checker.py
#~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import cv2
import numpy as np #работа с массивами
from tensorflow.keras.models import load_model #загрузка сохраненной модели

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class KerasChecker:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self,
               src_dir1: str,
               kerasimg_wh1: int,
               model_fname1: str,
               class_labels_lst1: list[str],
               rep_img_width2: int,
               rep_img_height2: int,
               is_test_img2: bool,
               test_class_inx2: int,
               dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.kerasimg_wh1 = kerasimg_wh1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.model_fname1 = model_fname1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.classes_lst1 = []
    for i in range(len(class_labels_lst1)):
      self.classes_lst1.append(class_labels_lst1[i])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.rep_img_width2 = rep_img_width2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.rep_img_height2 = rep_img_height2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.is_test_img2 = is_test_img2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.test_class_inx2 = test_class_inx2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.kerasimg_wh1: {self.kerasimg_wh1}')
    print(f'[INFO] self.model_fname1: `{self.model_fname1}`')
    print(f'[INFO] self.classes_lst1: len: {len(self.classes_lst1)}, {self.classes_lst1}')
    print(f'[INFO] self.rep_img_width2: {self.rep_img_width2}')
    print(f'[INFO] self.rep_img_height2: {self.rep_img_height2}')
    print(f'[INFO] self.is_test_img2: {self.is_test_img2}')
    print(f'[INFO] self.test_class_inx2: {self.test_class_inx2}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def image_check(self):
    #~ указываем размер изображения, на котором была обучена нейронка  
    #~ первое значение — это ширина,
    #~ второе значение — это высота.
    target_size=(self.kerasimg_wh1, self.kerasimg_wh1)
    #~ размер кадра для отчетного документа
    reptarget_size=(self.rep_img_width2, self.rep_img_height2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ tensorflow загрузка модели из файла
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.model_fname1: `{self.model_fname1}`')
    model = load_model(self.model_fname1)
    #~ проверяем архитектуру модели
    print(model.summary())
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    for i in range(img_lst_len):
      # print('~'*70)
      print(f'[INFO] {i}->{img_lst_len-1}: `{img_lst[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
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
      #~ масштабирование/сжатие изображения до размеров 224x224
      #~ и предсказание модели
      #~ в любом случае сжимаем, так как видео-камер с разрешение 224x224 не бывает
      #~ а изображение скорее всего с видеокамеры
      frame224 = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ для сохранения отчетного видео и кадров все кадры поджимаем
      if not self.is_test_img2:
        if img_width1 != self.rep_img_width2 or img_height1 != self.rep_img_height2:
          frame = cv2.resize(frame, reptarget_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ для предсказания нормализуем значения пикселей - делим на 255, приводим к диапазону [0..1]
      frame224 = frame224/255.0
      #~ добавление одной оси в начале, чтобы нейронка могла распознать пример
      Xexd = np.expand_dims(frame224, axis=0)
      #~ распознавание примера изображения - определение его класса 
      prediction = model.predict(Xexd, verbose=0) 
      pred_inx = 0
      feature_color = (0, 0, 255)
      rect_width = 118
      rect_height = 25
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if self.is_test_img2:
        rect_height = 48
      #~~~~~~~~~~~~~~~~~~~~~~~~
      class_lbl = f'original: {self.classes_lst1[self.test_class_inx2]}'
      class_color = (0, 0, 255)
      if self.test_class_inx2 == 1:
        rect_width = 168
        class_color = (255, 0, 0)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if prediction[0][0] > 0.5:
        pred_inx = 1
        feature_color = (255, 0, 0)
        rect_width = 168
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if not is_test_img2:
        rect_width = 115
        if prediction[0][0] > 0.5:
          rect_width = 166
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняем продетектированное изображение
      img_fname2 = os.path.join(self.dst_dir2, base_fname1+'.png')
      pred_lbl = f'predict: {self.classes_lst1[pred_inx]}'
      cv2.rectangle(frame, (0,0), (rect_width,rect_height), (255,255,255), -1)
      if self.is_test_img2:
        cv2.putText(frame, class_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 1, cv2.LINE_AA)
        cv2.putText(frame, pred_lbl, (2,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
      else:  
        cv2.putText(frame, pred_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
      cv2.imwrite(img_fname2, frame)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] KerasChecker ver.2024.09.26')
  print('~'*70)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями для детектирования объектов по рассчитаным весам
  # src_dir1 = 'c:/my_campy/smart_city/weights_fire/dataset_fire5_video_frame'
  # src_dir1 = 'c:/my_campy/smart_city/weights_fire/dataset_fire5_test/fire'
  src_dir1 = 'c:/my_campy/smart_city/weights_fire/dataset_fire5_test/non-fire'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ keras image width,height ширина-высота изображения, на котором произведено обучение keras
  kerasimg_wh1 = 224
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ файл с рассчитанными весами
  model_fname1 = 'c:/my_campy/smart_city/weights_fire/20240930/model20240926.h5'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список классов для детектирования
  class_labels_lst1 = ['fire', 'non-fire']
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ rep - report
  rep_img_width2 = 960
  rep_img_height2 = 540
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ для изображений из папки test - добавляем оригинальную разметку
  #~ True False
  is_test_img2 = True
  #~ 0: fire, 1: non-fire 
  test_class_inx2 = 1
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с файлами-изображениями результатами детектирования-отрисованными сущностями
  # dst_dir2 = 'c:/my_campy/smart_city/weights_fire/20240930_model20240926_fire5'
  # dst_dir2 = 'c:/my_campy/smart_city/weights_fire/20240930_model20240926_fire5_test_fire'
  dst_dir2 = 'c:/my_campy/smart_city/weights_fire/20240930_model20240926_fire5_test_non_fire'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  kchr_obj = KerasChecker(src_dir1, kerasimg_wh1, model_fname1, class_labels_lst1, rep_img_width2, rep_img_height2, is_test_img2, test_class_inx2, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  kchr_obj.image_check()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  kchr_obj.timer_obj.elapsed_time()
  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python keras_checker.py
#~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~