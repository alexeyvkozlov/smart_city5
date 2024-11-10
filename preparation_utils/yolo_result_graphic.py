#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_result_graphic.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import numpy as np
from matplotlib import pyplot as plt

from task_timer import TaskTimer


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class YoloResultGraphic:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def read_data(self, graphic_mode: int, src_fname: str):
    epoch_lst = []
    val_lst = []
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO.read_data] graphic_mode: {graphic_mode}, src_fname: `{src_fname}`:')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ читаем файл по строкам
    lines1 = []
    input_file1 = open(src_fname, 'r', encoding='utf-8')
    lines1 = input_file1.readlines()
    input_file1.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    lines1_len = len(lines1)
    # print(f'[INFO.read_data] lines1: len: {lines1_len}')
    if lines1_len < 2:
      print(f'[WARNING.read_data] lines1 len < 2: {lines1_len}')
      return epoch_lst,val_lst
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(1,lines1_len):
      # print(f'[INFO.read_data] {i}-{lines1_len-1}: {lines1[i]}')
      line2 = lines1[i].strip()
      if len(line2) < 1:
        continue
      #~            0      1      2         3          4  
      # fline2_5 = 'epoch, mAP50, mAP50-95, precision, recall\n'
      fields5 = line2.split(',')
      epoch_str = fields5[0].strip()
      mAP50_str = fields5[1].strip()
      mAP50_95_str = fields5[2].strip()
      precision_str = fields5[3].strip()
      recall_str = fields5[4].strip()
      # print(f'[INFO.read_data]   epoch_str: `{epoch_str}`')
      # print(f'[INFO.read_data]   mAP50_str: `{mAP50_str}`')
      # print(f'[INFO.read_data]   mAP50_95_str: `{mAP50_95_str}`')
      # print(f'[INFO.read_data]   precision_str: `{precision_str}`')
      # print(f'[INFO.read_data]   recall_str: `{recall_str}`')
      try:
        epoch_int = int(epoch_str)
        mAP50_float = float(mAP50_str)
        mAP50_95_float = float(mAP50_95_str)
        precision_float = float(precision_str)
        recall_float = float(recall_str)
      except ValueError as e:
        print(f'[WARNING] произошла ошибка при преобразовании строки в число: `{line2}`, : {e}')
        continue
      # print(f'[INFO.read_data]     epoch_int: {epoch_int}')
      # print(f'[INFO.read_data]     mAP50_float: {mAP50_float}')
      # print(f'[INFO.read_data]     mAP50_95_float: {mAP50_95_float}')
      # print(f'[INFO.read_data]     precision_float: {precision_float}')
      # print(f'[INFO.read_data]     recall_float: {recall_float}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      epoch_lst.append(epoch_int)
      #~ 0: mAP50
      #~ 1: mAP50-95
      #~ 2: precision
      #~ 3: recall
      if 0 == graphic_mode:
        val_lst.append(mAP50_float)
      elif 1 == graphic_mode:
        val_lst.append(mAP50_95_float)
      elif 2 == graphic_mode:
        val_lst.append(precision_float)
      elif 3 == graphic_mode:
        val_lst.append(recall_float)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    return epoch_lst,val_lst

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def draw_graphic1(self, graphic_mode: int, src_fname_lst: list[str], graphic_name_lst: list[str], dst_fname2: str):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # color_lst = ['red','green','blue']
    color_lst = ['#ED1C24', #~0 красный 
                 '#22B14C', #~1 зеленый
                 '#3F48CC', #~2 синий
                 '#880015', #~3 коричневый
                 '#FF7F27', #~4 оранжевый
                 '#00A2E8', #~5 голубой
                 '#A349A4', #~6 фиолетовый
                 '#FFF200', #~7 желтый
                 '#FFAEC9', #~8 светло-красный 
                 '#B5E61D', #~9 светло-зеленый
                 '#7092BE', #~10 светло-синий
                 '#B97A57', #~11 светло-коричневый
                 '#FFC90E', #~12 светло-оранжевый
                 '#99D9EA', #~13 светло-голубой
                 '#C8BFE7', #~14 светло-фиолетовый
                 '#EFE4B0' #~15 светло-желтый
                 ]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ график
    #~~~~~~~~~~~~~~~~~~~~~~~~
    fig = plt.figure(figsize=(10, 6))
    #~~~~~~~~~~~~~~~~~~~~~~~~
    plt.xlabel('Эпохи')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 0: mAP50
    #~ 1: mAP50-95
    #~ 2: precision
    #~ 3: recall
    if 0 == graphic_mode:
      plt.title('График mAP50')
      plt.ylabel('mAP50')
      dst_fname22 = dst_fname2 + '_mAP50.png'
    elif 1 == graphic_mode:
      plt.title('График mAP50-95')
      plt.ylabel('mAP50-95')
      dst_fname22 = dst_fname2 + '_mAP50_95.png'
    elif 2 == graphic_mode:
      plt.title('График Precision')
      plt.ylabel('Precision')
      dst_fname22 = dst_fname2 + '_precision.png'
    elif 3 == graphic_mode:
      plt.title('График Recall')
      plt.ylabel('Recall')
      dst_fname22 = dst_fname2 + '_recall.png'
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] dst_fname22: `{dst_fname22}`')
    if os.path.exists(dst_fname22):
      os.remove(dst_fname22)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ считываю данные из каждого текстового файла
    graphic_count = len(src_fname_lst)
    for i in range(graphic_count):
      print(f'[INFO] {i}-{graphic_count-1}:')
      print(f'[INFO]   {graphic_name_lst[i]}:  {src_fname_lst[i]}:')
      epoch_lst,val_lst = self.read_data(graphic_mode, src_fname_lst[i])
      # print(f'[INFO]   epoch_lst: len: {len(epoch_lst)}, {epoch_lst}')
      # print(f'[INFO]   val_lst: len: {len(val_lst)}, {val_lst}')
      #~ преобразуем данные в массивы NumPy для удобства работы
      xN = np.array(epoch_lst)
      yN = np.array(val_lst)
      # print(f'[INFO]   xN.shape: {xN.shape}, xN: {xN}')
      # print(f'[INFO]   yN.shape: {yN.shape}, yN: {yN}')
      # plt.plot(xN, yN, color='blue', linewidth=2, label='Синусоида')
      plt.plot(xN, yN,
               color=color_lst[i],
               linewidth=1,
               label=graphic_name_lst[i])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    plt.legend()
    plt.grid(which = 'major', color = 'black', alpha = 0.3)
    plt.minorticks_on()
    plt.grid(which = 'minor', color = 'gray', linestyle = '--', alpha = 0.3)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ сохранение графика в файл PNG
    #~ - **outputpath**: Полный путь до файла, куда будет сохранён график.
    #~ - **dpi**: Число точек на дюйм (dots per inch), определяющее разрешение изображения. Чем больше значение dpi, 
    #~  тем выше качество изображения, но и размер файла также увеличивается. Значение по умолчанию часто составляет 100, 
    #~  но для лучшего качества можно использовать значения около 300 или выше.
    #~ - **bboxinches**: Этот параметр управляет обрезанием рамки вокруг графика при сохранении. По умолчанию используется 
    #~  'standard', что приводит к тому, что внешние границы графика обрезаются до A4 страницы. 
    #~  'tight' включает весь график, но оставляет достаточно места для осей и меток.
    plt.savefig(dst_fname22, dpi=300, bbox_inches='tight')
    plt.close(fig)


  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def draw_graphicN(self, src_fname_lst: list[str], graphic_name_lst: list[str], dst_fname2: str):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ режим/тип формирования графика
    #~ epoch, mAP50, mAP50-95, precision, recall
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 0: mAP50
    #~ 1: mAP50-95
    #~ 2: precision
    #~ 3: recall
    for i in range(4):
      self.draw_graphic1(i, src_fname_lst, graphic_name_lst, dst_fname2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloResultGraphic ver.2024.10.21')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ список с путями файлам с данными
  src_fname_lst = []
  #~ список с подписями к графикам
  graphic_name_lst = []
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ пути к файлам с входными данными и подписям графиков в легенде
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # #~ 1
  # src_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241023_small_401epochs_optimizer_auto/results_20241023_small_401epochs_optimizer_auto_5.txt'
  # graphic_name1 = 'YOLOv8s optimizer: auto'
  # src_fname_lst.append(src_fname1)
  # graphic_name_lst.append(graphic_name1)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # #~ 2
  # src_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241024_small_462epochs_optimizer_SGD/results_20241024_small_462epochs_optimizer_SGD_5.txt'
  # graphic_name2 = 'YOLOv8s optimizer: SGD'
  # src_fname_lst.append(src_fname2)
  # graphic_name_lst.append(graphic_name2)
  # #~~~~~~~~~~~~~~~~~~~~~~~~
  # #~ 3
  # src_fname3 = 'c:/my_campy/smart_city/weights_car_accident/20241029_small_827epochs_optimizer_Adam/results_20241029_small_827epochs_optimizer_Adam_5.txt'
  # graphic_name3 = 'YOLOv8s optimizer: Adam'
  # src_fname_lst.append(src_fname3)
  # graphic_name_lst.append(graphic_name3)
  # #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 4
  src_fname4 = 'c:/my_campy/smart_city/weights_car_accident/20241030_small_551epochs_optimizer_AdamW/results_20241030_small_551epochs_optimizer_AdamW_5.txt'
  graphic_name4 = 'YOLOv8s optimizer: AdamW'
  src_fname_lst.append(src_fname4)
  graphic_name_lst.append(graphic_name4)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # #~ 5
  # src_fname5 = 'c:/my_campy/smart_city/weights_car_accident/20241031_small_101epochs_optimizer_RMSProp/results_20241031_small_101epochs_optimizer_RMSProp_5.txt'
  # graphic_name5 = 'YOLOv8s optimizer: RMSProp'
  # src_fname_lst.append(src_fname5)
  # graphic_name_lst.append(graphic_name5)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ model mode
  #~ YOLOv8n -> nano
  #~ YOLOv8s -> small
  #~ YOLOv8m -> medium
  #~ YOLOv8l -> large
  #~ YOLOv8x -> extra large
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ пути к выходным файлам-графикам
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/nano500epochs_small401epochs_medium412epochs_large394epochs'
  # dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241101_small_optimizer_auto_SGD_Adam_AdamW_RMSProp'
   # dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241101_0808_small_optimizer_auto_SGD_Adam_AdamW'
  # dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241101_0808_small_optimizer_auto'
  # dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241101_0808_small_optimizer_SGD'
  # dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241101_0808_small_optimizer_Adam'
  dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241101_0808_small_optimizer_AdamW'
  #~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  yrg_obj = YoloResultGraphic()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  yrg_obj.draw_graphicN(src_fname_lst, graphic_name_lst, dst_fname2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  yrg_obj.timer_obj.elapsed_time()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # cd c:\my_campy
  # .\camenv8\Scripts\activate
  # cd c:\my_campy\smart_city\preparation_utils
  # python yolo_result_graphic.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
