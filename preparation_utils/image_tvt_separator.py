#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python image_tvt_separator.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import uuid
import cv2
import random

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ train, valid, test
class ImageTvtSeparator:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ img_count если минус 1, значит все изображения
  def __init__(self, src_dir1: str, fprefix2: str, img_count2: int, train_perc2: int, valid_perc2: int, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.fprefix2 = fprefix2
    self.img_count2 = img_count2
    self.train_perc2 = train_perc2
    self.valid_perc2 = valid_perc2
    test_percent2 = 100 - self.train_perc2 - self.valid_perc2
    if not 100 == (self.train_perc2 + self.valid_perc2 + test_percent2):
      self.train_perc2 = 70
      self.valid_perc2 = 29
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_dir2 = dst_dir2
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.fprefix2: `{self.fprefix2}`')
    print(f'[INFO] self.img_count2: {self.img_count2}')
    print(f'[INFO] self.train_perc2: {self.train_perc2}')
    print(f'[INFO] self.valid_perc2: {self.valid_perc2}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def image_separate(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst1 = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len1 = len(img_lst1)
    if img_lst_len1 < 1:
      print('[WARNING] img_lst1 is empty')
      return
    print(f'[INFO] img_lst1: len: {img_lst_len1}')
    # print(f'[INFO] img_lst1: len: {img_lst_len1}, `{img_lst1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    train_dir2 = os.path.join(self.dst_dir2, 'train')
    self.dir_filer.create_directory(train_dir2)
    #~~~
    valid_dir2 = os.path.join(self.dst_dir2, 'valid')
    self.dir_filer.create_directory(valid_dir2)
    #~~~
    test_dir2 = os.path.join(self.dst_dir2, 'test')
    self.dir_filer.create_directory(test_dir2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ рассчитываем количество файлов для каждой части (train, valid, test)
    final_img_count2 = self.img_count2
    if final_img_count2 < 1:
      final_img_count2 = img_lst_len1
    train_count2 = int(final_img_count2 * self.train_perc2 / 100)
    valid_count2 = int(final_img_count2 * self.valid_perc2 / 100)
    print(f'[INFO]  final_img_count2: {final_img_count2}')
    print(f'[INFO]  train_count2: {train_count2}')
    print(f'[INFO]  valid_count2: {valid_count2}')
    print(f'[INFO]  test_count2: {final_img_count2-train_count2-valid_count2}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst1)
    # print(f'[INFO]   shuffle: img_lst1: len: {len(img_lst1)}, `{img_lst1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ счетчик кадров -> frame counter
    fcounter = 1
    digits = 5
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    for i in range(final_img_count2):
      # print('~'*70)
      print(f'[INFO] {i}->{img_lst_len1-1}: `{img_lst1[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst1[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst1[i])
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ проверяем изображение на корректность
      img1 = cv2.imread(img_fname1)
      img_width1 = 0
      img_height1 = 0
      try:
        img_width1 = img1.shape[1]
        img_height1 = img1.shape[0]
      except:
        print(f'[WARNING] corrupted image: {img_fname1}')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      dst_dir2 = test_dir2
      if fcounter <= train_count2:
        dst_dir2 = train_dir2
      elif fcounter <= train_count2 + valid_count2:
        dst_dir2 = valid_dir2
      # print(f'[INFO]  fcounter: {fcounter}, dst_dir2: `{dst_dir2}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fname2 = f'{self.fprefix2}{self.dir_filer.format_counter(fcounter, digits)}-{uuid.uuid1()}'
      img_fname2 = os.path.join(dst_dir2, fname2 + suffix_fname1)
      # print(f'[INFO]  fname2: `{fname2}`, img_fname2: `{img_fname2}`, ')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      self.dir_filer.copy_file(img_fname1, img_fname2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fcounter += 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ копирование изображений завершено
    print('='*70)
    print(f'[INFO] копирование изображений завершено: {img_lst_len1} -> {fcounter-1}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] ImageTvtSeparator ver.2024.09.12')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  src_dir1 = 'c:/my_campy/smart_city_dataset_fire/d16-my-must/non_fire224_unique640_3merge_1182'
  #~ 'f_' - fire, 'nf_' - non fire
  fprefix2 = 'nf_' 
  #~ img_count если минус 1, значит все изображения
  img_count2 = -1
  # img_count2 = 1114
  #~ perc - percent -> указаываем знасение в процентах
  train_perc2 = 70
  valid_perc2 = 29
  dst_dir2 = 'c:/my_campy/smart_city_final_fire_dataset/fire_dataset/non_fire'

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ python image_tvt_separator.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  tvt_obj = ImageTvtSeparator(src_dir1, fprefix2, img_count2, train_perc2, valid_perc2, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  tvt_obj.image_separate()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  tvt_obj.timer_obj.elapsed_time()