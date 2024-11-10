#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python copy_pairs_by_bbox.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import cv2
import random
import uuid

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CopyPairsByBbox:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    self.dst_dir2 = dst_dir2
    # print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    # print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def copy_pairs(self):
    # print('[INFO] copy_pairs')
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    bbox_dir = os.path.join(self.src_dir1, 'draw')
    print(f'[INFO] bbox_dir: `{bbox_dir}`')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst = self.dir_filer.get_image_list(bbox_dir)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst in `draw` directory is empty')
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst)
    # print(f'[INFO]   shuffle: img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ счетчик кадров -> frame counter
    # fcounter = 0
    digits = 5
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    pair_counter = 0
    for i in range(img_lst_len):
      # print('~'*70)
      # print(f'[INFO] {i}->{img_lst_len}: `{img_lst[i]}`')
      print(f'[INFO] {i}->{img_lst_len-1}')
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      # print(f'[INFO] base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      img_fname1,img_sname1 = self.dir_filer.get_image_fname(self.src_dir1, base_fname1)
      if len(img_fname1) < 1:
        continue
      lbl_fname1 = os.path.join(self.src_dir1, base_fname1 + '.txt')
      if not self.dir_filer.file_exists(lbl_fname1):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяю, что изображение корректно и читаемо
      img = cv2.imread(img_fname1)
      img_width = 0
      img_height = 0
      try:
        img_width = img.shape[1]
        img_height = img.shape[0]
      except:
        print(f'[WARNING] corrupted image: {img_fname1}')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      prefix_inx = f'f{self.dir_filer.format_counter(pair_counter, digits)}-'
      unic_fname = f'{uuid.uuid1()}'
      #~~~~~~~~~~~~~~~~~~~~~~~~
      img_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + img_sname1)
      lbl_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + '.txt')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # print(f'[INFO] img_fname1: `{img_fname1}`, lbl_fname1: `{lbl_fname1}`')
      # print(f'[INFO] img_fname2: `{img_fname2}`, lbl_fname2: `{lbl_fname2}`')
      self.dir_filer.copy_file(img_fname1, img_fname2)
      self.dir_filer.copy_file(lbl_fname1, lbl_fname2)
      pair_counter += 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print(f'[INFO] скопировано пар `image+label`: {pair_counter}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] CopyPairsByBbox ver.2024.09.14')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ test train valid
  #~~~~~~~~~~~~~~~~~~~~~~~~
  src_dir1 = 'd:/my_campy/smart_city_dataset_car_accident/d30'
  dst_dir2 = 'd:/my_campy/smart_city_dataset_car_accident/d31'

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python copy_pairs_by_bbox.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj = CopyPairsByBbox(src_dir1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj.copy_pairs()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  car_obj.timer_obj.elapsed_time()