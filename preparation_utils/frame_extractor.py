#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python frame_extractor.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import cv2
import uuid

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FrameExtractor:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_fname1: str, step_ms1: int, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_fname1 = src_fname1
    self.step_ms1 = step_ms1
    if self.step_ms1 < 1:
      self.step_ms1 = -1
    self.dst_dir2 = dst_dir2
    print(f'[INFO] self.src_fname1: `{self.src_fname1}`')
    print(f'[INFO] self.step_ms1: `{self.step_ms1}`')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def separate(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcap = cv2.VideoCapture(self.src_fname1)
    if not vcap.isOpened():
      print('[ERROR] can`t open video-file: `{self.src_fname1}`')
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~
    start_time1 = 0
    end_time2 = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT) / vcap.get(cv2.CAP_PROP_FPS) * 1000)
    print(f'[INFO] start_time1: {start_time1} msec, end_time2: {end_time2} msec')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ счетчик кадров -> frame counter
    fcounter = 0
    digits = 5
    #~ последовательно читаю кадры из видео-файла
    step_num = -1
    while vcap.isOpened():
      #~ читаем очерендной кадр  
      ret, frame = vcap.read()
      if not ret:
        break
      #~~~~~~~~~~~~~~~~~~~~~~
      #~ увеличиваем счетчик шагов
      step_num += 1
      #~ сдвигаемся вперед на заданный шаг
      if self.step_ms1 > 0:
        time_position2 = start_time1 + step_num*self.step_ms1
        if time_position2 > end_time2:
          break
        vcap.set(cv2.CAP_PROP_POS_MSEC, time_position2)
      else:
        if vcap.get(cv2.CAP_PROP_POS_MSEC) > end_time2:
          break
      print(f'[INFO] frame: {fcounter}')
      #~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняем текущий кадр
      prefix_inx = f'f{self.dir_filer.format_counter(fcounter, digits)}-'
      unic_fname = f'{uuid.uuid1()}'
      img_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + '.jpg')
      print(f'[INFO] img_fname2: `{img_fname2}`')
      #~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняю кадр
      cv2.imwrite(img_fname2, frame)
      #~ отображаю кадр
      cv2.imshow('frame_extractor', frame)
      #~ если нажата клавиша 'q', выходим из цикла
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #   break
      #~ если нажата клавиша 'esc', выходим из цикла
      key_press = cv2.waitKey(1) & 0xFF
      if 27 == key_press:
        print('[INFO] press key `esc` -> exit')
        break
      #~~~~~~~~~~~~~~~~~~~~~~
      fcounter += 1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcap.release()
    cv2.destroyAllWindows()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print(f'[INFO] Число сохраненных кадров, полученных при расщеплении видео: {fcounter}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] CropResizer ver.2024.09.11')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ исходный видео-файл
  src_fname1 = 'c:/my_campy/smart_city_dataset_car_accident/video1/testing2.mp4'
  #~ интервал между сохраняемыми кадрами, если -1, то все кадры
  step_ms1 = -1
  # step_ms1 = 300
  #~ директория для сохранения скриншотов
  dst_dir2 = 'c:/my_campy/smart_city_dataset_car_accident/video2-frames'


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ python frame_extractor.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~
  frm_obj = FrameExtractor(src_fname1, step_ms1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  frm_obj.separate()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  frm_obj.timer_obj.elapsed_time()