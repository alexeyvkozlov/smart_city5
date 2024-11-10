#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python video_splitter.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import time
import cv2
import uuid

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class VideoSplitter:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_fname1: str, step_ms2: int, frame_width2: int, frame_height2: int, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_fname1 = src_fname1
    #~~~
    self.step_ms2 = step_ms2
    if self.step_ms2 < 1:
      self.step_ms2 = -1
    #~~~
    self.frame_width2 = frame_width2
    self.frame_height2 = frame_height2
    if self.frame_width2 < 1 or self.frame_height2 < 1:
      self.frame_width2 = -1
      self.frame_height2 = -1
    #~~~
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_fname1: `{self.src_fname1}`')
    print(f'[INFO] self.step_ms2: {self.step_ms2}')
    print(f'[INFO] self.frame_width2: {self.frame_width2}, self.frame_height2: {self.frame_height2}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def frame_extract(self):
    print('~'*70)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    target_size=(self.frame_width2, self.frame_height2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ открываю видео-файл
    vcam = cv2.VideoCapture(self.src_fname1)
    if not vcam.isOpened():
      print(f'[ERROR] can`t open video-file: `{self.src_fname1}`')
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ оригинальные размеры кадра
    frame_width = int(vcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] original frame size: width: {frame_width}, height: {frame_height}, ratio: {round(frame_width/frame_height,5)}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    fps = vcam.get(cv2.CAP_PROP_FPS)
    print(f'[INFO] fps: {fps}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ [INFO] original frame size: width: 1920, height: 1080, ratio: 1.77778
    #~ [INFO] fps: 25.009117432531
    #~~~~~~~~~~~~~~~~~~~~~~~~
    start_time2 = 0
    end_time2 = int(vcam.get(cv2.CAP_PROP_FRAME_COUNT) / fps * 1000)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по кадрам -> детектируем на каждом события
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ число извлеченных из видео кадров
    fcounter = 0
    digits = 5
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # #~ если видео было предварительно записано с частотой 30fps,
    # #~ а на выходе хотим получить 25fps, значит надо отбросить каждый 6-ой кадр
    # iframe = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~
    while vcam.isOpened():
      #~ читаем очередной кадр  
      ret, frame = vcam.read()
      fcounter += 1
      if not ret:
        break
      # #~~~~~~~~~~~~~~~~~~~~~~~~
      # #~ отпрасываем каждый 6-ой кадр
      # iframe += 1
      # if 6 == iframe:
      #   iframe = 0
      #   continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ если указа шаг считывание кадров в millisecond - миллисекундах
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if self.step_ms2 > 0:
        time_position2 = start_time2 + fcounter*step_ms2
        # print(f'[INFO]  fcounter: {fcounter}, start_time2: {start_time2} msec, time_position2: {time_position2} msec')
        if time_position2 > end_time2:
          break
        vcam.set(cv2.CAP_PROP_POS_MSEC, time_position2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ масштабирование/сжатие изображения до размеров target_size = target_width x target_height
      if self.frame_width2 > 0:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # #~ закрашиваю серым цветом, если есть в этом необходимость
      # x1 = 0
      # y1 = 0
      # x2 = 120
      # y2 = 540
      # cv2.rectangle(frame, (x1, y1), (x2, y2), (145,147,144), -1)
      # x1 = 800
      # y1 = 0
      # x2 = 960
      # y2 = 540
      # cv2.rectangle(frame, (x1, y1), (x2, y2), (145,147,144), -1)
      # x1 = 0
      # y1 = 0
      # x2 = 960
      # y2 = 35
      # cv2.rectangle(frame, (x1, y1), (x2, y2), (145,147,144), -1)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняю кадр на диск
      fname2 = f'i80f{self.dir_filer.format_counter(fcounter, digits)}-{uuid.uuid1()}.png'
      img_fname2 = os.path.join(self.dst_dir2, fname2)
      print(f'[INFO] {fname2}')
      cv2.imwrite(img_fname2, frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отображаем кадр
      cv2.imshow('video-frame', frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ если нажата клавиша 'q', выходим из цикла
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #   break
      #~ если нажата клавиша 'esc', выходим из цикла
      key_press = cv2.waitKey(1) & 0xFF
      if 27 == key_press:
        print('[INFO] press key `esc` -> exit')
        break

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ освобождаем ресурсы
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcam.release()
    #~ закрываем все окна
    cv2.destroyAllWindows()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print(f'[INFO] число извлеченных видеокадров: {fcounter-1}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] VideoSplitter ver.2024.09.26')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  src_fname1 = 'c:/my_campy/smart_city/smart_city_check_photo_video/final_video/fall20241004-02.mp4'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ интервал чтения 1000мсек -> то есть считываем в одну секунду один кадр
  #~ если значение -1, то значит все кадры
  step_ms2 = -1
  #~ размеры целевого кадра, если его необходимо сжать
  #~ если значение -1, то значит сжатие не производим
  frame_width2 = 960
  frame_height2 = 540 #540
  #~ директория с извлеченными кадрами
  dst_dir2 = 'c:/my_campy/smart_city/smart_city_check_photo_video/final_video/final_frame-8'
  #~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python video_splitter.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  vsplitter_obj = VideoSplitter(src_fname1, step_ms2, frame_width2, frame_height2, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  vsplitter_obj.frame_extract()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  vsplitter_obj.timer_obj.elapsed_time()