#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city
# cd d:\my_campy\smart_city
#~~~~~~~~~~~~~~~~~~~~~~~~
# python smart_camera.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import time
import cv2

from settings_reader import SettingsReader
from fire_thread import FireThread
from car_accident_thread import CarAccidentThread
from person_fall_thread import PersonFallThread


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SmartCamera:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('~'*70)
    print('[INFO.SmartCamera] Smart Camera ver.2024.10.07')
    print('~'*70)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ путь к папке из которой запустили программу
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.prog_path = os.getcwd()
    print(f'[INFO.SmartCamera] program path: `{self.prog_path}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def start(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ читаем настройки из ini-фала
    #~~~~~~~~~~~~~~~~~~~~~~~~
    ini_reader = SettingsReader(self.prog_path)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ секция [CAMERA] 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    cam_id = ini_reader.get_camera_id()
    cam_name = ini_reader.get_camera_name()
    cam_url = ini_reader.get_camera_url()
    interval_ms = ini_reader.get_interval_ms()
    report_dir = ini_reader.get_report_dir()
    print(f'[INFO.SmartCamera]  camera id: {cam_id}')
    print(f'[INFO.SmartCamera]  camera name: `{cam_name}`')
    print(f'[INFO.SmartCamera]  camera url: `{cam_url}`')
    print(f'[INFO.SmartCamera]  interval, ms: {interval_ms}')
    print(f'[INFO.SmartCamera]  report: directory: `{report_dir}`')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ создаем потоки 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ детектирование "fire"
    fire_objthread = FireThread(self.prog_path, report_dir)
    #~ детектирование "car-accident"
    accident_objthread = CarAccidentThread(self.prog_path, report_dir)
    #~ для детектирования "person-fall"
    fall_objthread = PersonFallThread(self.prog_path, report_dir)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ открываю уличную видеокамеру или видео-файл
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcam = cv2.VideoCapture(cam_url)
    if not vcam.isOpened():
      print(f'[ERROR.SmartCamera] can`t open video-camera: `{cam_url}`')
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ непрерывным потоком читаю видео кадры
    #~~~~~~~~~~~~~~~~~~~~~~~~
    while True:
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ 2024.03.24 эти настройки для rtsp-камер из Воронежа не работают
      # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
      # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отрабатываем ожидание нажатия кнопки выхода - `esc`
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # key_press = cv2.waitKey(1) & 0xFF
      key_press = cv2.waitKey(interval_ms) & 0xFF
      if 27 == key_press:
        print('[INFO.SmartCamera] press key `esc` -> exit')
        break
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #~ читаем очередной кадр
      #~~~~~~~~~~~~~~~~~~~~~~~~
      ret, frame = vcam.read()    
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if not ret:
        #~ на rtsp-камерах иногда происходит срыв потока, поэтому закрываю и отрываю камеру повторно
        vcam.release()
        vcam = cv2.VideoCapture(cam_url)
        if not vcam.isOpened():
          print('[ERROR.SmartCamera] can`t open video-camera')
          break
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отправляем текущий кадр три потока-нейронки  
      #~ если флаг готовности кадра стоит в состоянии True, то значит он находится в обработке
      #~ если False, то значит его можно добавить в обработку
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ fire
      if not fire_objthread.get_frame_ready():
        fire_objthread.add_frame(cam_id,cam_name,frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ car-accident
      if not accident_objthread.get_frame_ready():
        accident_objthread.add_frame(cam_id,cam_name,frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ person-fall
      if not fall_objthread.get_frame_ready():
        fall_objthread.add_frame(cam_id,cam_name,frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отображаем кадр
      cv2.imshow('smart-camera', frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ после завершение работы с камерой останавливаем потоки
    #~ и освобождаем ресурсы
    #~~~~~~~~~~~~~~~~~~~~~~~~
    fire_objthread.set_stop(True)
    accident_objthread.set_stop(True)
    fall_objthread.set_stop(True)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcam.release()
    cv2.destroyAllWindows()
    #~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  scam_obj = SmartCamera()
  scam_obj.start()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city
# cd d:\my_campy\smart_city
#~~~~~~~~~~~~~~~~~~~~~~~~
# python smart_camera.py
#~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~