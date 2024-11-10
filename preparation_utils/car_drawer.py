#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python car_drawer.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import cv2

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CarDrawer:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    self.dst_dir2 = os.path.join(src_dir1, 'draw')
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def draw_feature_label_file(self, img_fname1: str, lbl_fname1: str, img_fname2: str):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # print(f'[INFO]   img_fname1: `{img_fname1}`')
    # print(f'[INFO]   lbl_fname1: `{lbl_fname1}`')
    # print(f'[INFO]   img_fname2: `{img_fname2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img = cv2.imread(img_fname1)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    lbl_file = open(lbl_fname1, 'r', encoding='utf-8')
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
      feature_color = (0, 255, 255)
      if 0 == class_id:
        feature_color = (0, 0, 255)
      elif 1 == class_id:
        feature_color = (255, 0, 0)

      #~~~~~~~~~~~~~~~~~~~~~~~~
      cv2.rectangle(img, (x_min, y_min), (x_max, y_max), feature_color, 3)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ и сохраняем изображение с нарисованными bbox - обведенными сущностями
    cv2.imwrite(img_fname2, img)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def image_draw(self):
    img_lst = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    for i in range(img_lst_len):
      # print('~'*70)
      print(f'[INFO] {i}->{img_lst_len}: `{img_lst[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      lbl_fname1 = os.path.join(self.src_dir1, base_fname1 + '.txt')
      if not self.dir_filer.file_exists(lbl_fname1):
        continue
      img_fname2 = os.path.join(self.dst_dir2, base_fname1 + suffix_fname1)
      # print(f'[INFO]   img_fname1: `{img_fname1}`')
      # print(f'[INFO]   lbl_fname1: `{lbl_fname1}`')
      # print(f'[INFO]   img_fname2: `{img_fname2}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ и отрисовываем файл-изображение по указанной разметке
      self.draw_feature_label_file(img_fname1, lbl_fname1, img_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] CarLabelDraw ver.2024.09.15')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  src_dir1 = 'd:/my_campy/smart_city_dataset_car_accident/d32-ttest'

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python car_drawer.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj = CarDrawer(src_dir1)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj.image_draw()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  car_obj.timer_obj.elapsed_time()