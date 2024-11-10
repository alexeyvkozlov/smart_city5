# -*- coding: utf-8 -*-
"""Диплом. 5этап. Пожары. Проверка AutoKeras. Алексей Козлов.

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13eFti1ibH0tlR5LHFDMpCv7ExS73hd46
"""

!pip install autokeras

import os
import shutil
import numpy as np
import cv2
import gdown
import time
import datetime
from tensorflow.keras.models import load_model #загрузка сохраненной модели

#~ фиксирую время начала выполнения процесса
time1 = time.time()

#~ скачиваем данные
#~ 20241108_1102_trial4_epocs10_50_greedy_acc96_2
# gdown.download('https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1fSaHbJhd_hZUxL_ul6m93VFbgq5O6Xr7', None, quiet=True)
#~ 20241108_1659_trial4_epocs10_40_bayesian_acc_96_2
gdown.download('https://drive.google.com/uc?export=download&confirm=no_antivirus&id=14Hd7IaCJeJgWf5SMHlKEY4vSdnGDpQpl', None, quiet=True)

# !unzip -qo dataset_fire5_check.zip -d /content/
!unzip -qo dataset_fire5_check_bayesian.zip -d /content/

# model_fname1 = '/content/dataset_fire5_check/model_greedy/model20241108k.keras'
model_fname1 = '/content/dataset_fire5_check_bayesian/model_bayesian/model20241108k.keras'
print(f'[INFO] model_fname1: `{model_fname1}`')
model = load_model(model_fname1)

#~ проверяем архитектуру модели
print(model.summary())

#~ keras image width,height ширина-высота изображения, на котором произведено обучение keras
kerasimg_wh1 = 224
print(f'[INFO] kerasimg_wh1: {kerasimg_wh1}')
#~ список категорий классов
classes_lst1 = ['fire', 'non-fire']
print(f'[INFO] classes_lst1: len: {len(classes_lst1)}, `{classes_lst1}`')
#~ rep - report
rep_img_width2 = 960
rep_img_height2 = 540
print(f'[INFO] rep_img_width2: {rep_img_width2}, rep_img_height2: {rep_img_height2}')
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ указываем размер изображения, на котором была обучена нейронка
#~ первое значение — это ширина,
#~ второе значение — это высота.
target_size=(kerasimg_wh1, kerasimg_wh1)
#~ размер кадра для отчетного документа rep - report
reptarget_size=(rep_img_width2, rep_img_height2)

#~ функция для получения списка имен image-файлов
def get_image_list(directory_path: str) -> list[str]:
  img_lst = []
  #~~~~~~~~~~~~~~~~~~~~~~~~
  if not os.path.exists(directory_path):
    return img_lst
  #~~~~~~~~~~~~~~~~~~~~~~~~
  for fname in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, fname)):
      if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        img_lst.append(fname)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  return img_lst

#~ `IMG_0035.JPG` -> base_fname: `IMG_0035`, suffix_fname: `.JPG`
def get_fname_base_suffix(file_name: str) -> tuple:
  #~ разделяем имя файла и расширение
  #~ находим индекс последней точки в строке
  last_dot_index = file_name.rfind('.')
  #~ возвращаем подстроку начиная с начала строки до последней точки включительно
  base_fname = file_name[:last_dot_index]
  #~ расширение
  suffix_fname = file_name[last_dot_index:]
  #~ возвращаем имя и расширение
  return base_fname,suffix_fname

def image_check(src_dir1: str, is_test_img2: bool, test_class_inx2: int, dst_dir2: str):
  print(f'[INFO] src_dir1: `{src_dir1}`')
  print(f'[INFO] is_test_img2: {is_test_img2}')
  print(f'[INFO] test_class_inx2: {test_class_inx2}')
  print(f'[INFO] dst_dir2: `{dst_dir2}`')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ создаю директорию для результатов
  if not os.path.exists(dst_dir2):
    os.makedirs(dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  img_lst1 = get_image_list(src_dir1)
  img_lst_len1 = len(img_lst1)
  if img_lst_len1 < 1:
    print('[WARNING]  img_lst1 is empty')
    return
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ побежали по изображениям в списке
  for i in range(img_lst_len1):
    # print('~'*70)
    print(f'[INFO]  {i}->{img_lst_len1-1}: `{img_lst1[i]}`')
    img_fname1 = os.path.join(src_dir1, img_lst1[i])
    base_fname1,suffix_fname1 = get_fname_base_suffix(img_lst1[i])
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
    #~ для сохранения отчетных кадров -> поджимаем по размерам
    if not is_test_img2:
      if img_width1 != rep_img_width2 or img_height1 != rep_img_height2:
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
    if is_test_img2:
      rect_height = 48
    #~~~~~~~~~~~~~~~~~~~~~~~~
    class_lbl = f'original: {classes_lst1[test_class_inx2]}'
    class_color = (0, 0, 255)
    if test_class_inx2 == 1:
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
        rect_width = 165
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ сохраняем продетектированное изображение
    img_fname2 = os.path.join(dst_dir2, base_fname1+'.png')
    pred_lbl = f'predict: {classes_lst1[pred_inx]}'
    cv2.rectangle(frame, (0,0), (rect_width,rect_height), (255,255,255), -1)
    if is_test_img2:
      cv2.putText(frame, class_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 1, cv2.LINE_AA)
      cv2.putText(frame, pred_lbl, (2,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
    else:
      cv2.putText(frame, pred_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feature_color, 1, cv2.LINE_AA)
    cv2.imwrite(img_fname2, frame)

#~ провека на тестовых изображениях fire
# src_dir1 = '/content/dataset_fire5_check/fire'
src_dir1 = '/content/dataset_fire5_check_bayesian/fire'
is_test_img2 = True
test_class_inx2 = 0
dst_dir2 = '/content/dataset_fire5_check_result/fire'
#~~~~~~~~~~~~~~~~~~~~~~~~
image_check(src_dir1, is_test_img2, test_class_inx2, dst_dir2)

#~ провека на тестовых изображениях non-fire
# src_dir1 = '/content/dataset_fire5_check/non-fire'
src_dir1 = '/content/dataset_fire5_check_bayesian/non-fire'
is_test_img2 = True
test_class_inx2 = 1
dst_dir2 = '/content/dataset_fire5_check_result/non-fire'
#~~~~~~~~~~~~~~~~~~~~~~~~
image_check(src_dir1, is_test_img2, test_class_inx2, dst_dir2)

#~ провека на тестовых изображениях video_frame, которые были скачаны из интернета
# src_dir1 = '/content/dataset_fire5_check/video_frame'
src_dir1 = '/content/dataset_fire5_check_bayesian/video_frame'
is_test_img2 = False
dst_dir2 = '/content/dataset_fire5_check_result/video_frame'
#~~~~~~~~~~~~~~~~~~~~~~~~
image_check(src_dir1, is_test_img2, test_class_inx2, dst_dir2)

#~ архивируем результаты, чтобы их можно было скачать
archive_time = datetime.datetime.now()
archive2 = f'/content/result_fire5_check_{archive_time.strftime("%Y%m%d_%H%M%S")}.zip'
dst_dir2 = '/content/dataset_fire5_check_result'
print(f'[INFO] archive2: `{archive2}`')
print(f'[INFO] dst_dir2: `{dst_dir2}`')
!zip -r {archive2} {dst_dir2}

#~ отображаю время, затраченное на выполнение всей программы
result_time2 = time.time() - time1
result_hour2 = int(result_time2//3600)
result_min2 = int(result_time2//60) - result_hour2*60
result_sec2 = int(round(result_time2%60))
result_msec2 = round(1000*result_time2%60)
execution_time2 = ''
if result_hour2 > 0:
  execution_time2 = f'Время обработки: {result_hour2} час. {result_min2} мин.'
elif result_min2 > 0:
  execution_time2 = f'Время обработки: {result_min2} мин. {result_sec2} сек.'
elif result_sec2 > 0:
  execution_time2 = f'Время обработки: {result_sec2} сек.'
else:
  execution_time2 = f'Время обработки: {result_msec2} мсек.'
print(f'[INFO] {execution_time2}')

finish_datetime = datetime.datetime.now()
print(f'[INFO] finish time: {finish_datetime.strftime("%Y.%m.%d %H:%M:%S")}')