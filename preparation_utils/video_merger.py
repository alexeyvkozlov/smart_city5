#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python video_merger.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import time
import cv2

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class VideoMerger:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, frame_width2: int, frame_height2: int, fps2: float, dst_fname2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    self.frame_width2 = frame_width2
    self.frame_height2 = frame_height2
    self.fps2 = fps2
    self.dst_fname2 = dst_fname2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.frame_width2: {self.frame_width2}')
    print(f'[INFO] self.frame_height2: {self.frame_height2}')
    print(f'[INFO] self.fps2: {self.fps2}')
    print(f'[INFO] self.dst_fname2: `{self.dst_fname2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_file(self.dst_fname2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def frame_merge(self):
    img_lst = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ определяем кодек и FPS для сохранения видео
    #~~~~~~~~~~~~~~~~~~~~~~~~
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(self.dst_fname2, fourcc, self.fps2, (self.frame_width2, self.frame_height2))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # #~ помечаем область интереса, которую надо закрасить серым цветом
    # xroi1 = 82
    # xroi2 = 137
    # yroi1 = 982
    # yroi2 = 1037
    # #~ цвет, которым закрасим область интереса
    # back_color_roi = (149, 149, 149)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # #~ если видео было предварительно записано с частотой 30fps,
    # #~ а на выходе хотим получить 25fps, значит надо отбросить каждый 6-ой кадр
    # iframe = 0
    # #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    for i in range(img_lst_len):
      # print('~'*70)
      print(f'[INFO] {i}->{img_lst_len-1}: {img_lst[i]}')
      img_fname1 = os.path.join(self.src_dir1, img_lst[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      # #~~~~~~~~~~~~~~~~~~~~~~~~
      # #~ отпрасываем каждый 6-ой кадр
      # iframe += 1
      # if 6 == iframe:
      #   iframe = 0
      #   continue
      # #~~~~~~~~~~~~~~~~~~~~~~~~
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
      if not img_width1 == self.frame_width2:
        print(f'[WARNING] unsupported image width: `{img_lst[i]}`')
        continue
      if not img_height1 == self.frame_height2:
        print(f'[WARNING] unsupported image height: {img_lst_len}: `{img_lst[i]}`')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # #~ рисуем прямоугольник, который заливаем серым цветом
      # cv2.rectangle(img1, (xroi1, yroi1), (xroi2, yroi2), back_color_roi, -1)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ и сохраняем кадр в видеофайл
      vout.write(img1)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отображаем прочитанный кадр
      cv2.imshow('frame', img1)
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
    vout.release()
    #~ закрываем все окна
    cv2.destroyAllWindows()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print('[INFO] Формирование видеофайла завершено:')
    print(f'[INFO]  количество добавленных видеокадров: {img_lst_len}')
    print(f'[INFO]  видеофайл: `{self.dst_fname2}`')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] VideoSplitter ver.2024.09.18')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ входные папаметры
  #~~~~~~~~~~~~~~~~~~~~~~~~
  src_dir1 = 'c:/my_campy/smart_city/smart_city_check_photo_video/final_video/final_frame-18'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ ширина и высота кадра (должны совпадать с исходными кадрами)
  #~ [INFO] original frame size: width: 1920, height: 1080, ratio: 1.77778
  #~ [INFO] fps: 25.009117432531
  frame_width2 = 960
  frame_height2 = 540
  #~ frame per second - количество кадров, сменяющих друг друга за одну секунду
  fps2 = 25.0
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с извлеченными кадрами
  dst_fname2 = 'c:/my_campy/smart_city/smart_city_check_photo_video/final_video/video18.mp4'
  #~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python video_merger.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  vmerger_obj = VideoMerger(src_dir1, frame_width2, frame_height2, fps2, dst_fname2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  vmerger_obj.frame_merge()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  vmerger_obj.timer_obj.elapsed_time()