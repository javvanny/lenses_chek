# YOLO Dataset Preparation and Training Script

Этот проект включает скрипт для подготовки данных и обучения модели сегментации с использованием YOLO. Скрипт выполняет следующие действия:
1. Разделяет изображения и маски на обучающую и валидационную выборки.
2. Преобразует маски в формат YOLO.
3. Сохраняет данные в соответствующей структуре директорий.
4. Подготавливает файл `data.yaml` для конфигурации обучения.
5. Загружает предварительно обученную модель YOLO для последующего использования.

## Установка

Перед запуском убедитесь, что у вас установлены следующие библиотеки Python:

- `ultralytics`
- `numpy`
- `opencv-python`
- `torch`
- `matplotlib`
- `sklearn`

Установить их можно с помощью команды:

```bash
pip install ultralytics numpy opencv-python torch matplotlib scikit-learn


## Входные данные

- train_dataset/img/: Директория с изображениями.
- train_dataset/msk/: Директория с масками (формат .png).

