import time
  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ менеджер для измерения времени операций
class TaskTimer:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self):
    self.init_timer()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ инициализация таймера
  def init_timer(self):
    #~ фиксация времени старта процесса
    self.t = time.time()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def calculate_execution_time(self):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ pасчет времени выполнения
    result_time = time.time()-self.t
    result_hour = int(result_time//3600)
    result_min = int(result_time//60)-result_hour*60
    result_sec = int(round(result_time%60))
    result_msec = round(1000*result_time%60)
    #~~~~~~~~~~~~~~~~~~~~~~~~
    execution_time = ''
    if result_hour > 0:
      execution_time = f'Время обработки: {result_hour} час. {result_min} мин.'
    elif result_min > 0:
      execution_time = f'Время обработки: {result_min} мин. {result_sec} сек.'
    elif result_sec > 0:
      execution_time = f'Время обработки: {result_sec} сек.'
    else:
      execution_time = f'Время обработки: {result_msec} мсек.'
    return execution_time

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ затраченное время на выполнение всей операции/процесса
  def elapsed_time(self):
    execution_time = self.calculate_execution_time()
    print('='*70)
    print(execution_time)
    print('='*70)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ затраченное время на выполнение всей операции/процесса
  #~ добавляем сообщение
  def message_elapsed_time(self, message_time: str):
    execution_time = self.calculate_execution_time()
    print(f'{message_time}{execution_time}')