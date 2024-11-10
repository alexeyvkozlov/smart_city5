#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_tvt_separator.py
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
class YoloTvtSeparator:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, train_perc2: int, valid_perc2: int, class_lst2: list[str], dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.train_perc2 = train_perc2
    self.valid_perc2 = valid_perc2
    test_percent2 = 100 - self.train_perc2 - self.valid_perc2
    if not 100 == (self.train_perc2 + self.valid_perc2 + test_percent2):
      self.train_perc2 = 70
      self.valid_perc2 = 29
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.class_lst2 = []
    for i in range(len(class_lst2)):
      self.class_lst2.append(class_lst2[i])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.train_perc2: {self.train_perc2}')
    print(f'[INFO] self.valid_perc2: {self.valid_perc2}')
    print(f'[INFO] self.class_lst2: len: {len(self.class_lst2)}, `{self.class_lst2}`')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def pair_separate(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst1 = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len1 = len(img_lst1)
    if img_lst_len1 < 1:
      print('[WARNING] img_lst1 is empty')
      return
    print(f'[INFO] img_lst1: len: {img_lst_len1}')
    # print(f'[INFO] img_lst1: len: {img_lst_len1}, `{img_lst1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    train_img_dir2 = os.path.join(self.dst_dir2, 'train', 'images')
    train_lbl_dir2 = os.path.join(self.dst_dir2, 'train', 'labels')
    self.dir_filer.create_directory(train_img_dir2)
    self.dir_filer.create_directory(train_lbl_dir2)
    #~~~
    valid_img_dir2 = os.path.join(self.dst_dir2, 'valid', 'images')
    valid_lbl_dir2 = os.path.join(self.dst_dir2, 'valid', 'labels')
    self.dir_filer.create_directory(valid_img_dir2)
    self.dir_filer.create_directory(valid_lbl_dir2)
    #~~~
    test_img_dir2 = os.path.join(self.dst_dir2, 'test', 'images')
    test_lbl_dir2 = os.path.join(self.dst_dir2, 'test', 'labels')
    self.dir_filer.create_directory(test_img_dir2)
    self.dir_filer.create_directory(test_lbl_dir2)
    #~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ рассчитываем количество файлов для каждой части (train, valid, test)
    train_count2 = int(img_lst_len1 * self.train_perc2 / 100)
    valid_count2 = int(img_lst_len1 * self.valid_perc2 / 100)
    print(f'[INFO]  train_count2: {train_count2}')
    print(f'[INFO]  valid_count2: {valid_count2}')
    print(f'[INFO]  test_count2: {img_lst_len1-train_count2-valid_count2}')
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
    for i in range(img_lst_len1):
      # print('~'*70)
      print(f'[INFO] {i}->{img_lst_len1-1}: `{img_lst1[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst1[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst1[i])
      lbl_fname1 = os.path.join(self.src_dir1, base_fname1 + '.txt')
      if not self.dir_filer.file_exists(lbl_fname1):
        continue
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      # print(f'[INFO]  lbl_fname1: `{lbl_fname1}`')
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
      dst_img_dir2 = test_img_dir2
      dst_lbl_dir2 = test_lbl_dir2
      if fcounter <= train_count2:
        dst_img_dir2 = train_img_dir2
        dst_lbl_dir2 = train_lbl_dir2
      elif fcounter <= train_count2 + valid_count2:
        dst_img_dir2 = valid_img_dir2
        dst_lbl_dir2 = valid_lbl_dir2
      # print(f'[INFO]  fcounter: {fcounter}, dst_dir2: `{dst_dir2}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fname2 = f'i{self.dir_filer.format_counter(fcounter, digits)}-{uuid.uuid1()}'
      img_fname2 = os.path.join(dst_img_dir2, fname2 + suffix_fname1)
      lbl_fname2 = os.path.join(dst_lbl_dir2, fname2 + '.txt')
      # print(f'[INFO]  fname2: `{fname2}`')
      # print(f'[INFO]    img_fname2: `{img_fname2}`')
      # print(f'[INFO]    lbl_fname2: `{lbl_fname2}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      self.dir_filer.copy_file(img_fname1, img_fname2)
      self.dir_filer.copy_file(lbl_fname1, lbl_fname2)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fcounter += 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ копирование пар "image+label" "изображение+разметка" завершено
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ записываем файл `data.yaml`, необходимый для обучения YOLO
    data_yaml_fname = os.path.join(self.dst_dir2, 'data.yaml')
    # print(f'[INFO] data-yaml-fname: `{data_yaml_fname}`')
    with open(data_yaml_fname, 'w', encoding='utf-8') as file_yaml:
      file_yaml.write('train: ../train/images\n')
      file_yaml.write('val: ../valid/images\n')
      file_yaml.write('test: ../test/images\n\n')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ number of classes
      file_yaml.write(f'nc: {len(self.class_lst2)}\n\n')
      #~ class names
      file_yaml.write(f'names: {self.class_lst2}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print(f'[INFO] копирование пар "image+label" "изображение+разметка" завершено: {img_lst_len1} -> {fcounter-1}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloTvtSeparator ver.2024.09.15')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
 
  src_dir1 = 'd:/my_campy/smart_city_dataset_car_accident/d31'
  #~ perc - percent -> указаываем знасение в процентах
  train_perc2 = 70
  valid_perc2 = 29
  #~ список классов
  class_lst2 = ['car-accident', 'non-car-accident']
  #~ папка с результатами
  dst_dir2 = 'd:/my_campy/smart_city_dataset_car_accident/d32'


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python yolo_tvt_separator.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


  tvt_obj = YoloTvtSeparator(src_dir1, train_perc2, valid_perc2, class_lst2, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  tvt_obj.pair_separate()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  tvt_obj.timer_obj.elapsed_time()