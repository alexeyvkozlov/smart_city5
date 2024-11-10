#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ import the necessary packages
import configparser
import os


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SettingsReader:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, data_path: str):
    config_filename = os.path.join(data_path, 'settings.ini')
    print(f'[INFO.SettingsReader] config filename: `{config_filename}`')
    if os.path.exists(config_filename):
      if not os.path.isfile(config_filename):
        print(f'[WARNING.SettingsReader] config file is not file: `{config_filename}`')
    else:
      print(f'[WARNING.SettingsReader] config filename is not exists: `{config_filename}`')
    self.config = configparser.ConfigParser()
    self.config.read(config_filename)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_camera_id(self) -> int:
    retVal = self.config.getint('CAMERA', 'id')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_camera_name(self) -> str:
    retVal = self.config.get('CAMERA', 'name')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_camera_url(self) -> str:
    retVal = self.config.get('CAMERA', 'url')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_interval_ms(self) -> int:
    retVal = self.config.getint('CAMERA', 'frame_interval_ms')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_report_dir(self) -> str:
    retVal = self.config.get('CAMERA', 'report_dir')
    return retVal

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fire_model_weights(self) -> str:
    retVal = self.config.get('FIRE_DETECTOR', 'model_weights')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fire_model_image_size(self) -> int:
    retVal = self.config.getint('FIRE_DETECTOR', 'model_image_size')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fire_predict_threshold(self) -> float:
    retVal = self.config.getfloat('FIRE_DETECTOR', 'predict_threshold')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fire_alarm_count(self) -> int:
    retVal = self.config.getint('FIRE_DETECTOR', 'alarm_count')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fire_alarm_time(self) -> float:
    retVal = self.config.getfloat('FIRE_DETECTOR', 'alarm_time')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fire_siren_time(self) -> float:
    retVal = self.config.getfloat('FIRE_DETECTOR', 'siren_time')
    return retVal

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_accident_weights(self) -> str:
    retVal = self.config.get('ACCIDENT_DETECTOR', 'weights')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_accident_yolo_image_size(self) -> int:
    retVal = self.config.getint('ACCIDENT_DETECTOR', 'yolo_image_size')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_accident_confidence(self) -> float:
    retVal = self.config.getfloat('ACCIDENT_DETECTOR', 'feature_confidence')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_accident_alarm_count(self) -> int:
    retVal = self.config.getint('ACCIDENT_DETECTOR', 'alarm_count')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_accident_alarm_time(self) -> float:
    retVal = self.config.getfloat('ACCIDENT_DETECTOR', 'alarm_time')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_accident_siren_time(self) -> float:
    retVal = self.config.getfloat('ACCIDENT_DETECTOR', 'siren_time')
    return retVal

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_yolo_model(self) -> str:
    retVal = self.config.get('FALL_DETECTOR', 'yolo_model')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_yolo_image_size(self) -> int:
    retVal = self.config.getint('FALL_DETECTOR', 'yolo_image_size')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_yolo_confidence(self) -> float:
    retVal = self.config.getfloat('FALL_DETECTOR', 'yolo_confidence')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_person_ratio(self) -> float:
    retVal = self.config.getfloat('FALL_DETECTOR', 'person_ratio')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_pipe_confidence(self) -> float:
    retVal = self.config.getfloat('FALL_DETECTOR', 'pipe_confidence')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_footgrav_deg(self) -> int:
    retVal = self.config.getint('FALL_DETECTOR', 'footgrav_deg')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_alarm_count(self) -> int:
    retVal = self.config.getint('FALL_DETECTOR', 'alarm_count')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_alarm_time(self) -> float:
    retVal = self.config.getfloat('FALL_DETECTOR', 'alarm_time')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_fall_siren_time(self) -> float:
    retVal = self.config.getfloat('FALL_DETECTOR', 'siren_time')
    return retVal
  
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def get_telegram_is_active(self) -> bool:
    retVal = self.config.getboolean('TELEGRAM_BOT', 'is_active')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_telegram_url(self) -> str:
    retVal = self.config.get('TELEGRAM_BOT', 'bot_url')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_telegram_token(self) -> str:
    retVal = self.config.get('TELEGRAM_BOT', 'token')
    return retVal
  #~~~~~~~~~~~~~~~~~~~~~~~~
  def get_telegram_chat_id(self) -> str:
    retVal = self.config.get('TELEGRAM_BOT', 'chat_id')
    return retVal