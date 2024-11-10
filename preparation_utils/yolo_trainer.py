#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python yolo_trainer.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
from ultralytics import YOLO

from task_timer import TaskTimer

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class YoloTrainer:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self,
               src_dir1: str,
               model_mode1: str,
               epochs_count1: int,
               batch_count1: int,
               img_size1: int,
               optimizer1: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.yaml_fname1 = os.path.join(src_dir1, 'data.yaml')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model_mode0 = 'm'
    if 'nano' == model_mode1:
      model_mode0 = 'n'
    elif 'small' == model_mode1:
      model_mode0 = 's'
    elif 'medium' == model_mode1:
      model_mode0 = 'm'
    elif 'large' == model_mode1:
      model_mode0 = 'l'
    elif 'extra large' == model_mode1:
      model_mode0 = 'x'
    self.model_yaml_name1 = f'yolov8{model_mode0}.yaml'
    self.model_pretrained_name1 = f'yolov8{model_mode0}.pt'
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.epochs_count1 = epochs_count1
    self.batch_count1 = batch_count1
    self.img_size1 = img_size1
    self.optimizer1 = optimizer1

    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] self.yaml_fname1: `{self.yaml_fname1}`')
    print(f'[INFO] self.model_yaml_name1: `{self.model_yaml_name1}`')
    print(f'[INFO] self.model_pretrained_name1: `{self.model_pretrained_name1}`')
    print(f'[INFO] self.epochs_count1: {self.epochs_count1}')
    print(f'[INFO] self.batch_count1: {self.batch_count1}')
    print(f'[INFO] self.img_size1: {self.img_size1}')
    print(f'[INFO] self.optimizer1: `{self.optimizer1}`')

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def calc_weights(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ https://docs.ultralytics.com/ru/modes/train/#key-features-of-train-mode
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ build from YAML and transfer weights
    model = YOLO(self.model_yaml_name1).load(self.model_pretrained_name1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ train the model
    #~ results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
    #~ Чтобы тренироваться с 2 GPU, CUDA-устройствами 0 и 1, используй следующие команды. 
    #~ По мере необходимости расширяйся на дополнительные GPU.
    #~ Train the model with 2 GPUs
    #~ results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('~'*70)
    print('[INFO] start train...')
    print('~'*70)
    # if -1 == self.img_size1:
    #   results = model.train(data=self.yaml_fname1, epochs=self.epochs_count1, batch=self.batch_count1)
    # else
    #   results = model.train(data=self.yaml_fname1, epochs=self.epochs_count1, batch=self.batch_count1, imgsz=self.img_size1)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ EarlyStopping: Training stopped early as no improvement observed in last 100 epochs.
    #~ Best results observed at epoch 162, best model saved as best.pt.
    #~ To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` 
    #~ to disable EarlyStopping.
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # model.train(data=self.yaml_fname1, epochs=self.epochs_count1, batch=self.batch_count1, imgsz=self.img_size1, patience=0)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model.train(data=self.yaml_fname1,
                epochs=self.epochs_count1,
                batch=self.batch_count1,
                imgsz=self.img_size1,
                optimizer=self.optimizer1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] YoloTrainer ver.2024.10.23')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ путь к обучающим данным: test train valid: images labels
  src_dir1 = 'c:/dataset_car_accident4'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ model mode
  #~ YOLOv8n -> nano
  #~ YOLOv8s -> small
  #~ YOLOv8m -> medium
  #~ YOLOv8l -> large
  #~ YOLOv8x -> extra large
  # model_mode1 = 'nano'
  model_mode1 = 'small'
  # model_mode1 = 'medium'
  # model_mode1 = 'large'
  # model_mode1 = 'extra large'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ epochs -	количество эпох для обучения: 1, 300, 500, 1000
  epochs_count1 = 1000
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ batch - количество изображений в одной партии: 16
  #~ batch = -1 -> для автопартии
  batch_count1 = 16
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ imgsz - размер входных изображений в виде целого числа: 640
  #~ imgsz = -1 -> для автоматического указания размера изображения -> по умолчанию -1
  img_size1 = 640
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ optimizer	'auto'	Выбери оптимизатор для тренировки. Варианты включают в себя 
  #~  SGD, Adam, AdamW, NAdam, RAdam, RMSProp и т.д., или auto для автоматического 
  #~ выбора на основе конфигурации модели. Влияет на скорость сходимости и стабильность.
  optimizer1 = 'auto'
  # optimizer1 = 'SGD'
  # optimizer1 = 'Adam'
  # optimizer1 = 'AdamW'
   # optimizer1 = 'NAdam'
   # optimizer1 = 'RAdam'
  # optimizer1 = 'RMSProp'
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ytrn_obj = YoloTrainer(src_dir1,
                         model_mode1,
                         epochs_count1,
                         batch_count1,
                         img_size1,
                         optimizer1
                         )
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ytrn_obj.calc_weights()
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  ytrn_obj.timer_obj.elapsed_time()
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ model mode
  #~ YOLOv8n -> nano
  #~ YOLOv8s -> small
  #~ YOLOv8m -> medium
  #~ YOLOv8l -> large
  #~ YOLOv8x -> extra large
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # cd c:\my_campy
  # .\camenv8\Scripts\activate
  # cd c:\my_campy\smart_city\preparation_utils
  # python yolo_trainer.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
