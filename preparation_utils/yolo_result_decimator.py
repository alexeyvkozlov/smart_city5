#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_result_decimator.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class YoloResultDecimator:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_fname1: str, dst_fname2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.src_fname1 = src_fname1
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.dst_fname2 = dst_fname2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.src_fname1: `{self.src_fname1}`')
    print(f'[INFO] self.dst_fname2: `{self.dst_fname2}`')

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def result_decimate(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    dir_filer = DirectoryFileWorker()
    dir_filer.remove_file(self.dst_fname2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ проверяем существует ли выходной файл
    if not dir_filer.file_exists(self.src_fname1):
      print(f'[WARNING] file is not exists: `{self.src_fname1}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ открываем исходный файл для чтения
    lines1 = []
    input_file1 = open(self.src_fname1, 'r', encoding='utf-8')
    lines1 = input_file1.readlines()
    input_file1.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ и открываем результирующий файл для записи
    #~ все записи
    dst_fname2_5 = self.dst_fname2 + '_5.txt'
    dst_fname2_11 = self.dst_fname2 + '_11.txt'
    #~ каждая десятая
    dst_fname2_5_10 = self.dst_fname2 + '_5_10.txt'
    dst_fname2_11_10 = self.dst_fname2 + '_11_10.txt'
    #~ и открываем файлы для записи
    output_file2_5 = open(dst_fname2_5, 'w', encoding='utf-8')
    output_file2_11 = open(dst_fname2_11, 'w', encoding='utf-8')
    output_file2_5_10 = open(dst_fname2_5_10, 'w', encoding='utf-8')
    output_file2_11_10 = open(dst_fname2_11_10, 'w', encoding='utf-8')
#~~~~~~~~~~~~~~~~~~~~~~~~
#                 0-epoch,       1-train/box_loss,       2-train/cls_loss,       3-train/dfl_loss, 4-metrics/precision(B),    5-metrics/recall(B),     6-metrics/mAP50(B),  7-metrics/mAP50-95(B),         8-val/box_loss,         9-val/cls_loss,        10-val/dfl_loss,              11-lr/pg0,              12-lr/pg1,              13-lr/pg2
# ===============================================================================================================================================================================================================================================================================================================================================
#                   epoch,         train/box_loss,         train/cls_loss,         train/dfl_loss,   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B),           val/box_loss,           val/cls_loss,           val/dfl_loss,                 lr/pg0,                 lr/pg1,                 lr/pg2
    #~ 0-epoch,
    #~ 1-train/box_loss,
    #~ 2-train/cls_loss,
    #~ 3-train/dfl_loss,
    #~ 4-metrics/precision(B),
    #~ 5-metrics/recall(B),
    #~ 6-metrics/mAP50(B),
    #~ 7-metrics/mAP50-95(B),
    #~ 8-val/box_loss,
    #~ 9-val/cls_loss,
    #~ 10-val/dfl_loss,
    #~ 11-lr/pg0,
    #~ 12-lr/pg1,
    #~ 13-lr/pg2
#~~~~~~~~~~~~~~~~~~~~~~~~
    #~ 0-epoch,
    #~ 6-metrics/mAP50(B),
    #~ 7-metrics/mAP50-95(B),
    #~ 4-metrics/precision(B),
    #~ 5-metrics/recall(B),
    #~
    #~ 1-train/box_loss,
    #~ 2-train/cls_loss,
    #~ 3-train/dfl_loss,
    #~
    #~ 8-val/box_loss,
    #~ 9-val/cls_loss,
    #~ 10-val/dfl_loss,
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~          0      1      2         3          4  
    fline2_5 = 'epoch, mAP50, mAP50-95, precision, recall\n'
    #~           0      1      2         3          4       5               6               7               8             9             10 
    fline2_11 = 'epoch, mAP50, mAP50-95, precision, recall, train_box_loss, train_cls_loss, train_dfl_loss, val_box_loss, val_cls_loss, val_dfl_loss\n'
    #~~~~~~~~~~~~~~~~~~~~~~~~
    output_file2_5.write(fline2_5)
    output_file2_11.write(fline2_11)
    output_file2_5_10.write(fline2_5)
    output_file2_11_10.write(fline2_11)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ оставляем только не пустые строки
    for i in range(1,len(lines1)):
      #~ удаляем пробелы в начале и конце строки
      line2 = lines1[i].strip()
      if len(line2) < 1:
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      fields14 = line2.split(',')
      if not 14 == len(fields14):
        continue
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ 0-epoch,
      #~ 6-metrics/mAP50(B),
      #~ 7-metrics/mAP50-95(B),
      #~ 4-metrics/precision(B),
      #~ 5-metrics/recall(B),
      epoch_str = fields14[0].strip()
      # print(f'[INFO] fields14[0]: `{fields14[0]}`, epoch_str: `{epoch_str}`')
      mAP50_str = fields14[6].strip()
      # print(f'[INFO] fields14[6]: `{fields14[6]}`, mAP50_str: `{mAP50_str}`')
      mAP50_95_str = fields14[7].strip()
      precision_str = fields14[4].strip()
      recall_str = fields14[5].strip()
      try:
        epoch_int = int(epoch_str)
      except ValueError as e:
        print(f'[ERROR] произошла ошибка при преобразовании строки в число: `{line2}`, : {e}')
        continue
      # print(f'[INFO] self.src_fname1: `{self.src_fname1}`')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~            0      1      2         3          4  
      # fline2_5 = 'epoch, mAP50, mAP50-95, precision, recall'
      fline2_5 = f'{epoch_str},{mAP50_str},{mAP50_95_str},{precision_str},{recall_str}\n'
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~
      #~ 1-train/box_loss,
      #~ 2-train/cls_loss,
      #~ 3-train/dfl_loss,
      train_box_loss_str = fields14[1].strip()
      train_cls_loss_str = fields14[2].strip()
      train_dfl_loss_str = fields14[3].strip()
      #~
      #~ 8-val/box_loss,
      #~ 9-val/cls_loss,
      #~ 10-val/dfl_loss,
      val_box_loss_str = fields14[8].strip()
      val_cls_loss_str = fields14[9].strip()
      val_dfl_loss_str = fields14[10].strip()
      #~
      #~             0      1      2         3          4       5               6               7               8             9             10 
      # fline2_11 = 'epoch, mAP50, mAP50-95, precision, recall, train_box_loss, train_cls_loss, train_dfl_loss, val_box_loss, val_cls_loss, val_dfl_loss'
      fline2_11 = f'{epoch_str},{mAP50_str},{mAP50_95_str},{precision_str},{recall_str},{train_box_loss_str},{train_cls_loss_str},{train_dfl_loss_str},{val_box_loss_str},{val_cls_loss_str},{val_dfl_loss_str}\n'
      #~~~~~~~~~~~~~~~~~~~~~~~~
      output_file2_5.write(fline2_5)
      output_file2_11.write(fline2_11)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ и выполняем прореживание - оставляем только каждую десятую эпоху
      if epoch_int % 10 == 0 or epoch_int == 1:
        #~ первая эпоха или эпоха кратна 10
        # output_file21.write(fline21 + '\n')
        output_file2_5_10.write(fline2_5)
        output_file2_11_10.write(fline2_11)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ закрываем файл с результатами
    output_file2_5.close()
    output_file2_11.close()
    output_file2_5_10.close()
    output_file2_11_10.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloResultDecimator ver.2024.10.16')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ путь к входному файлу
  src_fname1 = 'c:/my_campy/smart_city/weights_car_accident/20241031_small_101epochs_optimizer_RMSProp/train7/results.csv'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ путь к выходному файлу
  dst_fname2 = 'c:/my_campy/smart_city/weights_car_accident/20241031_small_101epochs_optimizer_RMSProp/results_20241031_small_101epochs_optimizer_RMSProp'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  yrd_obj = YoloResultDecimator(src_fname1, dst_fname2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  yrd_obj.result_decimate()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  yrd_obj.timer_obj.elapsed_time()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # cd c:\my_campy
  # .\camenv8\Scripts\activate
  # cd c:\my_campy\smart_city\preparation_utils
  # python yolo_result_decimator.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
