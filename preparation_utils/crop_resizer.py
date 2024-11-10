#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python crop_resizer.py
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
class CropResizer:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_dir1 = src_dir1
    self.dst_dir2 = dst_dir2
    print(f'[INFO] self.src_dir1: `{self.src_dir1}`')
    print(f'[INFO] self.dst_dir2: `{self.dst_dir2}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    self.dir_filer.remove_create_directory(self.dst_dir2)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def is_ratio169(self, img_width: int, img_height: int):
    return img_width * 9 == img_height * 16

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def image_prepare(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    target_width = 224
    target_height = 224
    #~ первое значение (224) — это ширина,
    #~ второе значение (224) — это высота.
    target_size=(target_width, target_height)
    cam_ratio169 = 16.0/9.0
    print(f'[INFO] target_size: {target_size}, cam_ratio169: {cam_ratio169}')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    img_lst = self.dir_filer.get_image_list(self.src_dir1)
    img_lst_len = len(img_lst)
    if img_lst_len < 1:
      print('[WARNING] img_lst is empty')
      return
    print(f'[INFO] img_lst: len: {img_lst_len}')
    # print(f'[INFO] img_lst: len: {img_lst_len}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ перемешиваем список файлов
    random.shuffle(img_lst)
    # print(f'[INFO]   shuffle: img_lst: len: {len(img_lst)}, `{img_lst}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ счетчик кадров -> frame counter
    fcounter = 0
    digits = 5
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ побежали по изображениям в списке
    for i in range(img_lst_len):
      # print('~'*70)
      print(f'[INFO] {i}->{img_lst_len-1}: `{img_lst[i]}`')
      img_fname1 = os.path.join(self.src_dir1, img_lst[i])
      base_fname1,suffix_fname1 = self.dir_filer.get_fname_base_suffix(img_lst[i])
      # print(f'[INFO]  img_fname1: `{img_fname1}`')
      # print(f'[INFO]  base_fname1: `{base_fname1}`, suffix_fname1: `{suffix_fname1}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      prefix_inx = f'f{self.dir_filer.format_counter(fcounter, digits)}-'
      unic_fname = f'{uuid.uuid1()}'
      #~~~~~~~~~~~~~~~~~~~~~~~~
      # img_fname2 = os.path.join(self.dst_dir2, img_lst[i])
      img_fname2 = os.path.join(self.dst_dir2, prefix_inx+unic_fname + '.jpg')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      img = cv2.imread(img_fname1)
      img_width = 0
      img_height = 0
      try:
        img_width = img.shape[1]
        img_height = img.shape[0]
      except:
        print(f'[WARNING] corrupted image: {img_fname1}')
        continue
      # print(f'[INFO]  img_width: {img_width}, img_height: {img_height}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if img_width < target_width:
        continue
      if img_height < target_height:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      if self.is_ratio169(img_width, img_height):
        # print('[INFO]  ===>соотношение сторон 16:9, просто сжимаем изображение')
        #~ масштабирование/сжатие изображения до размеров 224x224
        img224 = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        #~ и сохраняем трансформированное/сжатое изображение
        cv2.imwrite(img_fname2, img224)
      else:
        # print('[INFO]  ->изображение предварительно необходимо обрезать')
        crop_w = int(cam_ratio169*img_height)
        # print(f'[INFO]  img_width: {img_width}, crop_w: {crop_w}')
        if crop_w <= img_width:
          x1 = int((img_width - crop_w)/2)
          x2 = x1 + crop_w
          if x2 > img_width:
            x2 = img_width
          # print(f'[INFO]  x1: {x1}, x2: {x2}, x2-x1: {x2-x1}')
          img_crop1 = img[:, :x2]
          # print(f'[INFO]  img_crop1: w: {img_crop1.shape[1]}, h: {img_crop1.shape[0]}')
          img_crop2 = img_crop1[:, x1:]
          # print(f'[INFO]  img_crop2: w: {img_crop2.shape[1]}, h: {img_crop2.shape[0]}')
          img224w = cv2.resize(img_crop2, target_size, interpolation=cv2.INTER_AREA)
          #~ и сохраняем трансформированное изображение
          cv2.imwrite(img_fname2, img224w)
          # print(f'[INFO]  w->сохранил сжатое изображение: `{img_fname2}`')
        else:
          crop_h = int(img_width/cam_ratio169)
          if crop_h > img_height:
            continue
          y1 = int((img_height - crop_h)/2)
          y2 = y1 + crop_h
          if y2 > img_height:
            y2 = img_height
          img_crop1 = img[:y2, :]
          img_crop2 = img_crop1[y1:, :]
          img224h = cv2.resize(img_crop2, target_size, interpolation=cv2.INTER_AREA)
          cv2.imwrite(img_fname2, img224h)
          # print(f'[INFO]  h->сохранил сжатое изображение: `{img_fname2}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fcounter += 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('='*70)
    print('[INFO] Число изображений:')
    print(f'[INFO]   до обработки: {img_lst_len}')
    print(f'[INFO]   после обработки: {fcounter}')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] CropResizer ver.2024.09.11')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ fire non_fire
  src_dir1 = 'c:/my_campy/smart_city_dataset_fire/d12/fire'
  dst_dir2 = 'c:/my_campy/smart_city_dataset_fire/d12/fire224'


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ python crop_resizer.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj = CropResizer(src_dir1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  car_obj.image_prepare()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  car_obj.timer_obj.elapsed_time()