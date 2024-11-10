#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# cd /home/akozlov/my_campy
# source camenv8/bin/activate
# cd /home/akozlov/my_campy/smart_city/preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python image_comparator.py
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
class ImageComparator:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, similar_threshold1: float, unique_img_size1: int, is_label1: bool, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    self.similar_threshold1 = similar_threshold1
    self.unique_img_size1 = unique_img_size1
    self.is_label1 = is_label1
    self.dst_dir2 = dst_dir2
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.similar_threshold1: {self.similar_threshold1}')
    print(f'[INFO] self.unique_img_size1: {self.unique_img_size1}')
    print(f'[INFO] self.is_label1: {self.is_label1}')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # self.dir_filer.create_directory(self.dst_dir2)
    self.dir_filer.remove_create_directory(self.dst_dir2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ список - хэшей, которые отобраны как уникальные
    self.hash_lst = []

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ функция для вычисления хэша изображения
  def dhash(self, image, hash_size=8):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 1. cv2.resize: Функция изменяет размер изображения до заданного размера. 
    #~ В данном случае, мы задаем размеры (hashsize + 1, hashsize). 
    #~ Это значит, что ширина будет на единицу больше высоты.
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 2. resized:, 1: > resized:, :-1: Этот оператор сравнивает каждую пару соседних пикселей друг с другом. 
    #~ Если правый пиксель ярче левого, то результат равен True, иначе False. Эта логическая маска используется
    #~ для определения направления градиента каждого пикселя.
    diff = resized[:, 1:] > resized[:, :-1]
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 3. sum(2 ** i for (i, v) in enumerate(diff.flatten()) if v): Мы проходим по всем элементам маски diff
    #~ и умножаем их индексы на два в степени этих индексов. Если элемент равен True (пиксели различаются),
    # то включаем этот множитель в сумму. В результате получается уникальный числовой код, который служит
    # идентификатором изображения.
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

  # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # def is_unique_image(self, img_fname1: str) -> bool:
  #   is_unique1 = True
  #   #~~~~~~~~~~~~~~~~~~~~~~~~
  #   #~ загрузка изображения
  #   print(f'[INFO]  =>img_fname1: `{img_fname1}`')
  #   img1 = cv2.imread(img_fname1)
  #   hash1 = self.dhash(img1)
  #   print(f'[INFO]  =>hash1: {hash1}')
  #   #~~~~~~~~~~~~~~~~~~~~~~~~
  #   img_lst2 = self.dir_filer.get_image_list(self.dst_dir2)
  #   img_lst_len2 = len(img_lst2)
  #   print(f'[INFO]  =>self.dst_dir2: `{self.dst_dir2}`')
  #   print(f'[INFO]  =>img_lst_len2: {img_lst_len2}')
  #   if img_lst_len2 < 1:
  #     return is_unique1,hash1
  #   #~ преобразование изображения в массив
  #   for j in range(img_lst_len2):
  #     print(f'[INFO]   j: {j}->{img_lst_len2-1}: `{img_lst2[j]}`')
  #     img_fname2 = os.path.join(self.dst_dir2, img_lst2[j])
  #     img2 = cv2.imread(img_fname2)
  #     # print(f'[INFO]  img_fname2: `{img_fname2}`')
  #     #~~~~~~~~~~~~~~~~~~~~~~~~
  #     hash2 = self.dhash(img2)
  #     print(f'[INFO]  =>hash2: {hash2}')
  #     #~~~~~~~~~~~~~~~~~~~~~~~~
  #     difference = bin(hash1 ^ hash2).count('1')
  #     #~ similar - схожий
  #     similarity = (1 - difference / (8 * 8)) * 100
  #     print(f'[INFO]   similarity: {similarity}')
  #     if similarity >= self.similar_threshold:
  #       is_unique1 = False
  #       break
  #   #~~~~~~~~~~~~~~~~~~~~~~~~
  #   return is_unique1,hash1

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def is_unique_image(self, img_fname1: str) -> bool:
    is_unique1 = True
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ загрузка изображения
    # print(f'[INFO]  =>img_fname1: `{img_fname1}`')
    img1 = cv2.imread(img_fname1)
    hash1 = self.dhash(img1)
    # print(f'[INFO]  =>hash1: {hash1}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    hash_lst_len2 = len(self.hash_lst)
    # print(f'[INFO]  =>hash_lst_len2: {hash_lst_len2}')
    if hash_lst_len2 < 1:
      return is_unique1,hash1
    #~ сравниваем хэш-изображения с хэшами из списка
    for j in range(hash_lst_len2):
      hash2 = self.hash_lst[j]
      difference = bin(hash1 ^ hash2).count('1')
      #~ similar - схожий
      similarity = (1 - difference / (8 * 8)) * 100
      # print(f'[INFO]   =>j: {j}->{hash_lst_len2-1}, similarity: {similarity}')
      # print(f'[INFO]   =>hash2: {hash2}')
      if similarity >= self.similar_threshold1:
        is_unique1 = False
        break
    #~~~~~~~~~~~~~~~~~~~~~~~~
    return is_unique1,hash1

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def compare_images(self):
    self.hash_lst.clear()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst1 = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len1 = len(img_lst1)
    if img_lst_len1 < 1:
      print('[WARNING] img_lst1 is empty')
      return
    # print(f'[INFO] img_lst1: len: {img_lst_len1}')
    # print(f'[INFO] img_lst1: len: {img_lst_len1}, `{img_lst1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst1)
    # print(f'[INFO]   shuffle: img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    #~ счетчик кадров -> frame counter
    #~ префикс - для счетчика кадров, для того чтобы потом понятнее было для ручного разбора
    fcounter = 0
    digits = 5
    for i in range(img_lst_len1):
      # print('='*70)
      print(f'[INFO] {i}->{img_lst_len1-1}: `{img_lst1[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst1[i])
      #~~~~~~~~~~~~~~~~~~~~~~~~
      img1 = cv2.imread(img_fname1)
      img_width1 = 0
      img_height1 = 0
      try:
        img_width1 = img1.shape[1]
        img_height1 = img1.shape[0]
      except:
        print(f'[WARNING] corrupted image: {img_fname1}')
        continue
      # print(f'[INFO]  img_width: {img_width}, img_height: {img_height}')
      #~ проверяем, что размеры изображения соответсвуют заданным размерам
      if not self.unique_img_size1 == img_width1:
        print(f'[WARNING] unsupported width image: {img_fname1}')
        continue
      if not self.unique_img_size1 == img_height1:
        print(f'[WARNING] unsupported height image: {img_fname1}')
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst1[i])
      lbl_fname1 = os.path.join(self.src_dir1, base_fname1 + '.txt')
      # print(f'[INFO] img_fname1: `{img_fname1}`')
      # print(f'[INFO] lbl_fname1: `{lbl_fname1}`')
      if self.is_label1:
        if not self.dir_filer.file_exists(lbl_fname1):
          continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      is_unique1,hash1 = self.is_unique_image(img_fname1)
      # print(f'[INFO] is_unique1: {is_unique1}, hash1: `{hash1}`')
      if is_unique1:
        #~ img_fname1 - уникальный файл, в целевой папке такого нет,
        #~ поэтому копируем это изображение и разметку 
        #~ и сохраняем в список хэш изображения img_fname1
        #~~~~~~~~~~~~~~~~~~~~~~~~
        prefix_inx = f'f{self.dir_filer.format_counter(fcounter, digits)}-'
        unic_fname = f'{uuid.uuid1()}'
        #~~~~~~~~~~~~~~~~~~~~~~~~
        # img_fname2 = os.path.join(self.dst_dir2, img_lst1[i])
        # lbl_fname2 = os.path.join(self.dst_dir2, base_fname1 + '.txt')
        #~~~
        img_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + suffix_fname1)
        # img_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + '.jpg')
        lbl_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + '.txt')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        # print(f'[INFO]  img_fname2: `{img_fname2}`')
        # print(f'[INFO]  lbl_fname2: `{lbl_fname2}`')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~
        self.dir_filer.copy_file(img_fname1, img_fname2)
        #~~~
        # #~ масштабирование/сжатие изображения до размеров 224x224
        # #~ первое значение (224) — это ширина,
        # #~ второе значение (224) — это высота.
        # target_size=(self.unique_img_size1, self.unique_img_size1)
        # img640 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
        # cv2.imwrite(img_fname2, img640)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.is_label1:
          self.dir_filer.copy_file(lbl_fname1, lbl_fname2)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        self.hash_lst.append(hash1)
        # print(f'[INFO] self.hash_lst: len: {len(self.hash_lst)}, {self.hash_lst}')
        # print(f'[INFO] self.hash_lst: len: {len(self.hash_lst)}')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        fcounter += 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print('[INFO] Число изображений:')
    print(f'[INFO]   до обработки: {img_lst_len1}')
    print(f'[INFO]   после обработки: {fcounter}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] ImageComparator ver.2024.09.14')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ параметр порога сходства-похожести
  #~ для YOLO с размером изображений 640x640 -> 70.0, 60.0
  #~ для изображений 224x224 -> 30.0, 10.0
  similar_threshold1 = 60.0
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ размер изображения - ширина и высота в пикселях,
  #~ все изображения должны быть одинакового размера
  unique_img_size1 = 640
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ для YOLO - True, для остальных False 
  is_label1 = True
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ fire224 non_fire224 fire224_unique non_fire224_unique

  src_dir1 = 'd:/my_campy/smart_city_dataset_car_accident/d26_stage3'
  dst_dir2 = 'd:/my_campy/smart_city_dataset_car_accident/d27_stage3'

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python image_comparator.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~
  comp_obj = ImageComparator(src_dir1, similar_threshold1, unique_img_size1, is_label1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  comp_obj.compare_images()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  comp_obj.timer_obj.elapsed_time()