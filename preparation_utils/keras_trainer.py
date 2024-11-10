#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\preparation_utils
# cd d:\my_campy\smart_city\preparation_utils
#~~~~~~~~~~~~~~~~~~~~~~~~
# python keras_trainer.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import os
import numpy as np #работа с массивами
import cv2
import matplotlib.pyplot as plt #отрисовка изображений
import datetime #работа с датой
import tensorflow as tf
from tensorflow.keras.models import Sequential  #подключение класса создания модели Sequential
from tensorflow.keras import layers #подключение слоев
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils #утилиты для подготовки данных

from task_timer import TaskTimer
from dirfile_worker import DirectoryFileWorker

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class KerasTrainer:
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
      print(f'[INFO]  category_dir1: {category_dir1}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      img_lst1 = self.dir_filer.get_image_list(category_dir1)
      img_lst_len1 = len(img_lst1)
      if img_lst_len1 < 1:
        print('[WARNING]  img_lst1 is empty')
        continue
      print(f'[INFO]  img_lst1: len: {img_lst_len1}')
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ побежали по изображениям в списке
      for j in range(img_lst_len1):
        # print(f'[INFO]   {j}->{img_lst_len1-1}: `{img_lst1[j]}`')
        img_fname1 = os.path.join(category_dir1, img_lst1[j])
        # print(f'[INFO]     img_fname1: `{img_fname1}`')
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ открываем изображение с использованием библиотеки opencv
        img_cv = cv2.imread(img_fname1)
        # print(f'[INFO] 8=>type(img_cv): {type(img_cv)}, img_cv.shape: {img_cv.shape}')
        # print(f'[INFO] 9=>type(img_cv[0][0][0]): {type(img_cv[0][0][0])}, img_cv[0][0][0]: {img_cv[0][0][0]}')
        # [INFO] 8=>type(img_cv): <class 'numpy.ndarray'>, img_cv.shape: (640, 640, 3)
        # [INFO] 9=>type(img_cv[0][0][0]): <class 'numpy.uint8'>, img_cv[0][0][0]: 174
        #~~~~~~~~~~~~~~~~~~~~~~~~
        img_width = 0
        img_height = 0
        try:
          img_width = img_cv.shape[1]
          img_height = img_cv.shape[0]
          # print(f'[INFO]  img_width: {img_width}, img_height: {img_height}, target_imgwh2: {target_imgwh2}, target_size2: {target_size2}')
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
        # print(f'[INFO] 12=>type(img_cv): {type(img_cv)}, img_cv.shape: {img_cv.shape}')
        # print(f'[INFO] 13=>type(img_cv[0][0][0]): {type(img_cv[0][0][0])}, img_cv[0][0][0]: {img_cv[0][0][0]}')
        # [INFO] 12=>type(img_cv): <class 'numpy.ndarray'>, img_cv.shape: (64, 64, 3)
        # [INFO] 13=>type(img_cv[0][0][0]): <class 'numpy.float64'>, img_cv[0][0][0]: 0.5490196078431373
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
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ train
    #~~~~~~~~~~~~~~~~~~~~~~~~
    print('-'*50)
    print('[INFO] Train')
    self.X_train, self.y_train = self.load_images_labels(self.train_dir1, target_imgwh2)
    #~~~ X_train
    # print(f'[INFO] type(self.X_train): {type(self.X_train)}')
    # print(f'[INFO] self.X_train.shape: {self.X_train.shape}')
    # print(f'[INFO] self.X_train.dtype: {self.X_train.dtype}')
    #~~~ y_train
    # print(f'[INFO] type(self.y_train): {type(self.y_train)}')
    # print(f'[INFO] self.y_train.shape: {self.y_train.shape}')
    # print(f'[INFO] self.y_train.dtype: {self.y_train.dtype}')
    # print(f'[INFO] self.y_train: {self.y_train}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # [INFO] type(self.X_train): <class 'numpy.ndarray'>
    # [INFO] self.X_train.shape: (1654, 224, 224, 3)
    # [INFO] self.X_train.dtype: float64
    # [INFO] type(self.y_train): <class 'numpy.ndarray'>
    # [INFO] self.y_train.shape: (1654,)
    # [INFO] self.y_train.dtype: int32
    # [INFO] self.y_train: [0 0 0 ... 1 1 1]
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # [INFO] type(self.X_train): <class 'numpy.ndarray'>
    # [INFO] self.X_train.shape: (1654, 224, 224, 3)
    # [INFO] self.X_train.dtype: float32
    # [INFO] type(self.y_train): <class 'numpy.ndarray'>
    # [INFO] self.y_train.shape: (1654,)
    # [INFO] self.y_train.dtype: float32
    # [INFO] self.y_train: [0. 0. 0. ... 1. 1. 1.]

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

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ создание модели нейронной сети Keras/TensorFlow для классификации изображений
  #~ с использованием сверточных нейронных сетей (CNN)
  #~ и обучение модели - расчет весов
  def train_model(self, learning_rate2: float, target_imgwh2: int, batch_size2: int, epochs2: int, verbose2: int):
    print('~'*70)
    print('[INFO] train neural network model...')
    print('-'*50)
    print(f'[INFO] learning rate: {learning_rate2}')
    print(f'[INFO] target image width and height: {target_imgwh2}')
    print(f'[INFO] batch size: {batch_size2}')
    print(f'[INFO] epochs: {epochs2}')
    print(f'[INFO] verbose: {verbose2}')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ cоздание модели
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model = Sequential()

    # model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(target_imgwh2, target_imgwh2, 3)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    # model.add(layers.BatchNormalization())

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Компиляция модели
    #~~~~~~~~~~~~~~~~~~~~~~~~
    # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate2),
                  metrics=['accuracy'])
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Вывод структуры модели
    print(model.summary())
    #~ и сохраняем ее графический файл
    model_summary_fname = os.path.join(self.dst_dir2, 'model_summary.png')
    utils.plot_model(model, to_file=model_summary_fname, show_shapes=True, show_layer_names=True, dpi=300)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Обучение модели
    #~~~~~~~~~~~~~~~~~~~~~~~~
    history = model.fit(x=self.X_train,                               #Обучающая выборка параметров
                        y=self.y_train,                               #Обучающая выборка меток класса
                        batch_size=batch_size2,                       #Размер батча (пакета)
                        epochs=epochs2,                               #Количество эпох обучения
                        validation_data=(self.X_valid, self.y_valid), #Валидационные данные
                        verbose=verbose2,                             #Отображение хода обучения
                        shuffle=True)                                 #Перемешивание перед каждой эпохой

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
    model.save(model_fpath)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ визуализация потерь (loss)
    self.plot_loss(history)
    #~ визуализация метрики (accuracy)
    self.plot_accuracy(history)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ оценка и предсказание модели на тестовом наборе
    print('-'*50)
    self.model_evaluation(model)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ предсказание модели на тестовых изображениях
    # print('-'*50)
    self.detect_draw_test_images(model)

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
  print('[INFO] KerasTrainer ver.2024.09.26')
  print('~'*70)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ директория с датасетом
  src_dir1 = 'c:/dataset_fire'
  #~ список категорий классов
  class_labels_lst1 = ['fire', 'non-fire']
  #~ директория с результатами
  dst_dir2 = 'c:/dataset_fire_20240926_res_10epochs'
  #~ ширина-высота сжатого изображения для нейронки
  #~ target_imgwh2 - target_img_width_height2
  # target_imgwh2 = 64
  target_imgwh2 = 224
  #~ переменные-настройки-параметры-для-обучения-сети
  learning_rate2 = 1e-4 #1e-4, 1e-3 
  batch_size2 = 32
  epochs2 = 10 #10, 20, 150
  verbose2 = 1 #0, 1

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~
  # python keras_trainer.py
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~
  ktrn_obj = KerasTrainer(src_dir1, class_labels_lst1, dst_dir2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ загружаем train-valid-test массивы-данных
  ktrn_obj.load_train_valid_test_data(target_imgwh2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ создание модели нейронной сети Keras/TensorFlow для классификации изображений
  #~ с использованием сверточных нейронных сетей (CNN)
  #~ и обучение модели - расчет весов
  ktrn_obj.train_model(learning_rate2, target_imgwh2, batch_size2, epochs2, verbose2)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ отображаем время выполнения программы
  ktrn_obj.timer_obj.elapsed_time()