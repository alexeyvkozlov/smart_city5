#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python keras_automl_trainer.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import numpy as np #работа с массивами
import cv2
import datetime #работа с датой

# import tensorflow as tf
# from tensorflow.keras.models import Sequential  #подключение класса создания модели Sequential
# from tensorflow.keras import layers #подключение слоев
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import utils #утилиты для подготовки данных

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping
from tensorflow.keras import utils #утилиты для подготовки данных

import matplotlib.pyplot as plt

import autokeras as ak

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class KerasAutoMLTrainer:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, src_dir1: str, class_labels_lst1: list[str], dst_dir2: str):
    #~ засекаем время выполнения всего процесса
    self.timer_obj = TaskTimer()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ директории с датасетом
    self.train_dir1 = os.path.join(src_dir1, 'train')
    self.valid_dir1 = os.path.join(src_dir1, 'valid')
    self.test_dir1 = os.path.join(src_dir1, 'test')
    #~ список категорий классов
    self.classes_lst1 = []
    for i in range(len(class_labels_lst1)):
      self.classes_lst1.append(class_labels_lst1[i])
    #~ директория с результатами
    self.dst_dir2 = dst_dir2
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ объект для работы с файлами, папками и т.д.
    self.dir_filer = DirectoryFileWorker()
    #~ удаляю результирующую директорию с весами, графиками и т.д.
    self.dir_filer.remove_create_directory(self.dst_dir2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'[INFO] src_dir1: `{src_dir1}`')
    print(f'[INFO]  train_dir1: `{self.train_dir1}`')
    print(f'[INFO]  valid_dir1: `{self.valid_dir1}`')
    print(f'[INFO]  test_dir1: `{self.test_dir1}`')
    print(f'[INFO] classes_lst1: len: {len(self.classes_lst1)}, `{self.classes_lst1}`')
    print(f'[INFO] dst_dir2: `{self.dst_dir2}`')

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ функция для загрузки изображений и меток
  def load_images_labels(self, data_dir1: str, target_imgwh2: int):
    images = []
    labels = []
    #~ ширина-высота сжатого изображения для нейронки
    #~ target_imgwh2 - target_img_width_height2
    target_size2 = (target_imgwh2, target_imgwh2)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(len(self.classes_lst1)):
      print(f'[INFO] {i}: {self.classes_lst1[i]}')
      category_dir1 = os.path.join(data_dir1, self.classes_lst1[i])
      # print(f'[INFO]  category_dir1: {category_dir1}')
      # [INFO] Test
      # [INFO] 0: fire
      # [INFO]  category_dir1: d:/dataset_fire4\test\fire
      # [INFO]  img_lst1: len: 13
      # [INFO] 1: non-fire
      # [INFO]  category_dir1: d:/dataset_fire4\test\non-fire
      # [INFO]  img_lst1: len: 13
      #~~~~~~~~~~~~~~~~~~~~~~~~
      img_lst1 = self.dir_filer.get_image_list(category_dir1)
      img_lst_len1 = len(img_lst1)
      if img_lst_len1 < 1:
        print('[WARNING]  img_lst1 is empty')
        continue
      # print(f'[INFO]  img_lst1: len: {img_lst_len1}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ побежали по изображениям в списке
      for j in range(img_lst_len1):
        print(f'[INFO]  {j}->{img_lst_len1-1}: `{img_lst1[j]}`')
        # [INFO]  12->12: `nf_01182-fcf9ab75-7110-11ef-a68f-bcee7b784ecb.jpg`
        img_fname1 = os.path.join(category_dir1, img_lst1[j])
        # print(f'[INFO]  2-> img_fname1: `{img_fname1}`')
        # [INFO]  2-> img_fname1: `d:/dataset_fire4\test\non-fire\nf_01182-fcf9ab75-7110-11ef-a68f-bcee7b784ecb.jpg`
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ открываем изображение с использованием библиотеки opencv
        img_cv = cv2.imread(img_fname1)
        # print(f'[INFO]  3-> type(img_cv): {type(img_cv)}, img_cv.shape: {img_cv.shape}')
        # print(f'[INFO]  4-> type(img_cv[0][0][0]): {type(img_cv[0][0][0])}, img_cv[0][0][0]: {img_cv[0][0][0]}')
        # [INFO]  3-> type(img_cv): <class 'numpy.ndarray'>, img_cv.shape: (224, 224, 3)
        # [INFO]  4-> type(img_cv[0][0][0]): <class 'numpy.uint8'>, img_cv[0][0][0]: 0
        #~~~~~~~~~~~~~~~~~~~~~~~~
        img_width = 0
        img_height = 0
        try:
          img_width = img_cv.shape[1]
          img_height = img_cv.shape[0]
          # print(f'[INFO]  5-> img_width: {img_width}, img_height: {img_height}, target_imgwh2: {target_imgwh2}, target_size2: {target_size2}')
          # [INFO]  5-> img_width: 224, img_height: 224, target_imgwh2: 224, target_size2: (224, 224)
        except:
          print(f'[WARNING] corrupted image: {img_fname1}')
          continue
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ сжимаем изображение
        if img_width != target_imgwh2 or img_height != target_imgwh2:
          img_cv = cv2.resize(img_cv, target_size2, interpolation=cv2.INTER_AREA)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ и нормализуем значения пикселей - делим на 255, приводим к диапазону [0..1]
        img_cv = img_cv/255.0
        # print(f'[INFO]  6-> type(img_cv): {type(img_cv)}, img_cv.shape: {img_cv.shape}')
        # print(f'[INFO]  7-> type(img_cv[0][0][0]): {type(img_cv[0][0][0])}, img_cv[0][0][0]: {img_cv[0][0][0]}')
        # [INFO]  6-> type(img_cv): <class 'numpy.ndarray'>, img_cv.shape: (224, 224, 3)
        # [INFO]  7-> type(img_cv[0][0][0]): <class 'numpy.float64'>, img_cv[0][0][0]: 0.0        
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ добавляем нормализованную картинку массив в список
        images.append(img_cv)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        labels.append(i)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # return np.array(images), np.array(labels)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ для уменьшения расходования памяти приводим тип данных от по умолчанию float64 к float32
    #~ и форма хранения данных - массив - <class 'numpy.ndarray'>
    images32 = np.array(images).astype(np.float32)
    labels32 = np.array(labels).astype(np.float32)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    return images32, labels32

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ загружаем train-valid-test массивы-данных
  def load_train_valid_test_data(self, target_imgwh2: int):
    print('~'*70)
    print('[INFO] load train-valid data...')
    print(f'[INFO] target image width and height: {target_imgwh2}')
    # [INFO] target image width and height: 224
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ train
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('-'*50)
    print('[INFO] Train')
    self.X_train, self.y_train = self.load_images_labels(self.train_dir1, target_imgwh2)
    # #~~~ X_train
    # print(f'[INFO] type(self.X_train): {type(self.X_train)}')
    # print(f'[INFO] self.X_train.shape: {self.X_train.shape}')
    # print(f'[INFO] self.X_train.dtype: {self.X_train.dtype}')
    # [INFO] type(self.X_train): <class 'numpy.ndarray'>
    # [INFO] self.X_train.shape: (1654, 224, 224, 3)
    # [INFO] self.X_train.dtype: float32
    # #~~~ y_train
    # print(f'[INFO] type(self.y_train): {type(self.y_train)}')
    # print(f'[INFO] self.y_train.shape: {self.y_train.shape}')
    # print(f'[INFO] self.y_train.dtype: {self.y_train.dtype}')
    # print(f'[INFO] self.y_train: {self.y_train}')
    # [INFO] type(self.y_train): <class 'numpy.ndarray'>
    # [INFO] self.y_train.shape: (1654,)
    # [INFO] self.y_train.dtype: float32
    # [INFO] self.y_train: [0. 0. 0. ... 1. 1. 1.]
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ valid
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('-'*50)
    print('[INFO] Valid')
    self.X_valid, self.y_valid = self.load_images_labels(self.valid_dir1, target_imgwh2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ test
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('-'*50)
    print('[INFO] Test')
    self.X_test, self.y_test = self.load_images_labels(self.test_dir1, target_imgwh2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ отображаю размеры и размерности сформированных массивов
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('-'*50)
    print(f'[INFO] self.X_train: shape: {self.X_train.shape}, dtype: {self.X_train.dtype}')
    print(f'[INFO] self.y_train: shape: {self.y_train.shape}, dtype: {self.y_train.dtype}')
    print(f'[INFO] self.X_valid: shape: {self.X_valid.shape}, dtype: {self.X_valid.dtype}')
    print(f'[INFO] self.y_valid: shape: {self.y_valid.shape}, dtype: {self.y_valid.dtype}')
    print(f'[INFO] self.X_test: shape: {self.X_test.shape}, dtype: {self.X_test.dtype}')
    print(f'[INFO] self.y_test: shape: {self.y_test.shape}, dtype: {self.y_test.dtype}')
    # [INFO] self.X_train: shape: (1654, 224, 224, 3), dtype: float32
    # [INFO] self.y_train: shape: (1654,), dtype: float32
    # [INFO] self.X_valid: shape: (684, 224, 224, 3), dtype: float32
    # [INFO] self.y_valid: shape: (684,), dtype: float32
    # [INFO] self.X_test: shape: (26, 224, 224, 3), dtype: float32
    # [INFO] self.y_test: shape: (26,), dtype: float32

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ создание модели нейронной сети Keras/TensorFlow для классификации изображений
  #~ с использованием сверточных нейронных сетей (CNN)
  #~ и обучение модели - расчет весов
  def train_model(self, tuner_type2: str, max_trials2: int, batch_size2: int, epochs2: int, patience2: int):
    print('~'*70)
    print('[INFO] train neural network model...')
    print('-'*50)
    print(f'[INFO] tuner-type: `{tuner_type2}`')
    print(f'[INFO] max-trials: {max_trials2}')
    print(f'[INFO] batch size: {batch_size2}')
    print(f'[INFO] epochs: {epochs2}')
    print(f'[INFO] patience: {patience2}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ путь к папке, где “AutoKeras” будет сохранять результаты (по умолчанию “auto-keras”).
    dst_autodir2 = os.path.join(self.dst_dir2, 'auto-keras')
    print(f'[INFO] dst-autodir: `{dst_autodir2}`')
    #~ создаю директорию
    self.dir_filer.create_directory(dst_autodir2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ запускаем AutoKeras на подбор модели
    #~ cоздаем экземпляр классификатора
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # Инициализация классификатора AutoKeras
    # clf = ak.ImageClassifier(
    #   max_trials=2,
    #   objective='val_accuracy',
    #   directory='pavement_crack',
    #   overwrite=True)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # # Initialize the image classifier.
    # clf = ak.ImageClassifier(overwrite=True, max_trials=2, objective='val_accuracy', tuner='greedy') 
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ cоздаем экземпляр классификатора
    clf = ak.ImageClassifier(
      tuner=tuner_type2,
      max_trials=max_trials2,
      loss='binary_crossentropy',
      metrics=['accuracy'],  # Отслеживаем accuracy во время обучения
      objective='val_accuracy',  # Оптимизируем по val_accuracy
      overwrite=True,
      directory=dst_autodir2)

    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ create an EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience2, restore_best_weights=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ обучаем модель
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # clf.fit(X_train, y_train, epochs=8, validation_data=(X_val, y_val))
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # # Feed the image classifier with training data.
    # clf.fit(x_train, y_train, epochs=10, batch_size=64)  # Change no of epochs to improve the model
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ self.X_train - обучающая выборка параметров
    #~ self.y_train - обучающая выборка меток класса
    #~ self.X_valid, self.y_valid - валидационные данные
    history = clf.fit(
        self.X_train,
        self.y_train,
        validation_data=(self.X_valid, self.y_valid),
        epochs=epochs2,
        batch_size=batch_size2,  # Указываем размер пакета
        callbacks=[early_stopping]
    )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ экспортируем лучшую модель
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # # Export as a Keras Model.
    # model = clf.export_model()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    best_model = clf.export_model()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ вывод структуры модели в виде графического файла
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print(best_model.summary())
    #~ и сохраняем ее графический файл
    model_summary_fname = os.path.join(self.dst_dir2, 'best_model_summary.png')
    utils.plot_model(best_model, to_file=model_summary_fname, show_shapes=True, show_layer_names=True, dpi=300)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ вывод структуры модели в виде текстового файла
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ bml: best model layer
    bml_fname = os.path.join(self.dst_dir2, 'best_model_layers.txt')
    bml_file = open(bml_fname, 'w', encoding='utf-8')
    #~ печатаем заголовок
    bml_file.write("Layer details:\n")
    #~ проходимся по каждому слою модели
    for layer in best_model.layers:
      #~ записываем имя слоя
      bml_file.write(f"Layer name: {layer.name}\n")
      #~ записываем тип слоя
      bml_file.write(f"Layer type: {type(layer)}\n")
      #~ получаем конфигурацию слоя
      config = layer.get_config()
      #~ записываем конфигурацию слоя
      bml_file.write("Layer configuration:\n")
      for key, value in config.items():
        bml_file.write(f"  {key}: {value}\n")
      #~ делаем пустую строку между слоями
      bml_file.write("\n")
    #~~~~~~~~~~~~~~~~~~~~~~~~
    bml_optimizer = best_model.optimizer
    bml_learning_rate = bml_optimizer.learning_rate
    #~ пишем название оптимизатора
    bml_file.write(f"\nOptimizer: {type(bml_optimizer).__name__}\n")
    #~ пишем значение learning_rate
    bml_file.write(f"Learning rate: {bml_learning_rate}\n")
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ закрываем файл
    bml_file.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Обучение модели завершено
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ сохраняем рассчитанные веса в файл
    print('-'*50)
    print('[INFO] Обучение модели нейронной сети завершено:')
    event_datetime = datetime.datetime.now()
    # model_fname = f'model{event_datetime.strftime("%Y.%m.%d %H:%M:%S")}.h5'
    model_fname = f'model{event_datetime.strftime("%Y%m%d")}.h5'
    # print(f'[INFO] model_fname: `{model_fname}`')
    model_fpath = os.path.join(self.dst_dir2, model_fname)
    print(f'[INFO]   saved model: `{model_fpath}`')
    #~ cохраняем только веса
    # model.save_weights('model.h5')
    #~ cохраняем модель и веса
    best_model.save(model_fpath)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ визуализация потерь (loss)
    self.plot_loss(history)
    #~ визуализация метрики (accuracy)
    self.plot_accuracy(history)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ оценка и предсказание модели на тестовом наборе
    print('-'*50)
    self.model_evaluation(best_model)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ предсказание модели на тестовых изображениях
    # print('-'*50)
    # results = clf.predict(x_test)
    self.detect_draw_test_images(best_model)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ визуализация потерь (loss)
  def plot_loss(self, history):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('График потерь (Loss)')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    # plt.show()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ сохранение графика в файл PNG
    fname = f'loss_curve.png'
    loss_fname = os.path.join(self.dst_dir2, fname)
    #~ - **outputpath**: Полный путь до файла, куда будет сохранён график.
    #~ - **dpi**: Число точек на дюйм (dots per inch), определяющее разрешение изображения. Чем больше значение dpi, 
    #~  тем выше качество изображения, но и размер файла также увеличивается. Значение по умолчанию часто составляет 100, 
    #~  но для лучшего качества можно использовать значения около 300 или выше.
    #~ - **bboxinches**: Этот параметр управляет обрезанием рамки вокруг графика при сохранении. По умолчанию используется 
    #~  'standard', что приводит к тому, что внешние границы графика обрезаются до A4 страницы. 
    #~  'tight' включает весь график, но оставляет достаточно места для осей и меток.
    plt.savefig(loss_fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ визуализация метрики (accuracy)
  def plot_accuracy(self, history):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('График точности (Accuracy)')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    # plt.show()
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ сохранение графика в файл PNG
    fname = f'accuracy_curve.png'
    accuracy_fname = os.path.join(self.dst_dir2, fname)
    plt.savefig(accuracy_fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ оценка и предсказание модели на тестовом наборе
  def model_evaluation(self, model):
    evaluation_fname = os.path.join(self.dst_dir2, 'model_evaluation.txt')
    #~ оценка модели на тестовых данных
    test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
    evaluation_str = f'Точность на тестовом наборе: {round(test_acc*100, 1)}%' 
    print(f'[INFO] {evaluation_str}')
    #~ cохранение точности в текстовый файл
    with open(evaluation_fname, 'w', encoding='utf-8') as f:
      f.write(evaluation_str)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ предсказание модели на тестовых изображениях
  def detect_draw_test_images(self, model):
    predict_dir = os.path.join(self.dst_dir2, 'predict')
    self.dir_filer.create_directory(predict_dir)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ количество изображений
    X_test_shape = self.X_test.shape
    y_test_shape = self.y_test.shape
    img_count = X_test_shape[0]
    lbl_count = y_test_shape[0]
    # print(f'[INFO] img_count: {img_count}, lbl_count: {lbl_count}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(lbl_count):
      # print(f'[INFO] i: {i}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      Ximg = self.X_test[i]
      #~ добавление одной оси в начале, чтобы нейронка могла распознать пример
      Xexd = np.expand_dims(Ximg, axis=0)
      #~ распознавание примера изображения - определение его класса 
      prediction = model.predict(Xexd, verbose=0) 
      # pred = prediction[0][0]
      # print(f'[INFO]  ===>class: {int(self.y_test[i])} -> prediction: {prediction}')
      # print(f'[INFO]   ==>type(prediction): {type(prediction)}, shape: {prediction.shape}, dtype: {prediction.dtype}')
      # print(f'[INFO]    =>pred: {pred}')
      pred_inx = 0
      if prediction[0][0] > 0.5:
        pred_inx = 1
      # print(f'[INFO]  ===>class: {int(self.y_test[i])} -> prediction: {pred_inx} ({prediction[0][0]})')
      # [INFO]  ===>class: 0 -> prediction: 0 (6.1594523660464736e-18)
      # [INFO]  ===>class: 1 -> prediction: 1 (0.9999997615814209)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ сохраняем распознанные изображения из тестовой папки
      fname2 = f'original{int(self.y_test[i])}_predict{pred_inx}_inx{i}.jpg'
      img_fname2 = os.path.join(predict_dir, fname2)
      # print(f'[INFO]  img_fname2: `{img_fname2}`')
      class_inx = int(self.y_test[i])
      class_lbl = f'original: {self.classes_lst1[class_inx]}'
      pred_lbl = f'predict: {self.classes_lst1[pred_inx]}'
      #~~~~~~~~~~~~~~~~~~~~~~~~
      class_color = (0, 0, 255)
      if 1 == class_inx:
        class_color = (255, 0, 0)
      pred_color = (0, 0, 255)
      if 1 == pred_inx:
        pred_color = (255, 0, 0)
      #~~~~~~~~~~~~~~~~~~~~~~~~
      Ximg255 = (Ximg*255.0).astype(np.uint8)
      cv2.rectangle(Ximg255, (0,0), (168,48), (255,255,255), -1)
      cv2.putText(Ximg255, class_lbl, (2,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 1, cv2.LINE_AA)
      cv2.putText(Ximg255, pred_lbl, (2,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 1, cv2.LINE_AA)
      cv2.imwrite(img_fname2, Ximg255)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  print('~'*70)
  print('[INFO] KerasTrainer ver.2024.11.05')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ исходные изображения, список классов и директория для результаттов
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с датасетом
  src_dir1 = 'c:/dataset_fire4'
  #~ список категорий классов
  class_labels_lst1 = ['fire', 'non-fire']
  #~ директория с результатами
  dst_dir2 = 'c:/dataset_fire_20241105_res'
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ настраиваемые парметры для расчета модели
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ ширина-высота сжатого изображения для нейронки
  #~ target_imgwh2 - target_img_width_height2
  target_imgwh2 = 224
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ tuner (тюнер): оптимизатор для подбора архитектуры модели.
  #~ можно выбрать “random”, “bayesian”, “hyperband” или “greedy”. 
  #~ random
  #~ bayesian
  #~ hyperband
  #~ greedy
  tuner_type2 = 'random'
  # tuner_type2 = 'bayesian'
  # tuner_type2 = 'hyperband'
  # tuner_type2 = 'greedy'
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ max_trials2: максимальное количество раз, которое “AutoKeras” будет искать модели (по умолчанию 100).
  max_trials2 = 1 #2 1
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ размер пакета во время обучения
  batch_size2 = 32
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ epochs: количество эпох для обучения модели
  epochs2 = 10 #10, 20, 150
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ patience(терпение): определяет количество эпох без улучшения валидационной потери перед остановкой обучения (ранняя остановка).
  patience2 = 3
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ktrn_obj = KerasAutoMLTrainer(src_dir1, class_labels_lst1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ загружаем train-valid-test массивы-данных
  ktrn_obj.load_train_valid_test_data(target_imgwh2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ создание модели нейронной сети Keras/TensorFlow для классификации изображений
  #~ с использованием сверточных нейронных сетей (CNN)
  #~ и обучение модели - расчет весов
  ktrn_obj.train_model(tuner_type2, max_trials2, batch_size2, epochs2, patience2)

  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  ktrn_obj.timer_obj.elapsed_time()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python keras_automl_trainer.py
#~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~