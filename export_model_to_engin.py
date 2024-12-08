
from ultralytics import YOLO
import torch
import torchvision
from onnx2torch import convert




# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
device = 'cpu'

# Путь к предварительно обученной модели YOLO в формате .pt
model_path = r'./baseline.pt'

# Загрузка модели YOLO
yolo_model = YOLO(model_path)

# Экспорт модели в формат .engine
#yolo_model.export(format="engine", half=True, dynamic=True, int8=True)

# yolo_model.export(format="onnx", half=True, dynamic=True, int8=True, device='cpu')
#
# yolo_model_onnx = './baseline.onnx'
# pytorch_model = convert(yolo_model_onnx)


# model_dynamic_quantized = torch.quantization.quantize_dynamic(yolo_model, qconfig_spec={torch.nn.modules.conv.Conv2d,torch.nn.modules.container.Sequential},
#                                                               dtype=torch.qint8)
# Загружаем модель
# Определяем типы слоев для квантования
quantizable_layers = {
    torch.nn.Conv2d,
    torch.nn.Linear,
    torch.nn.BatchNorm2d,
}

# Применяем динамическое квантование
#yolo_model = torch.load('./baseline.pt')
torch.load('./baseline.pt', map_location=device)

quantized_model = torch.quantization.quantize_dynamic(
    yolo_model,
    qconfig_spec={torch.nn.Conv2d, torch.nn.Linear,torch.nn.BatchNorm2d},  # Указываем слои для квантования
    dtype=torch.qint8  # Используем квантование до 8 бит
)

# Сохраняем квантованную модель
torch.save(quantized_model, 'quantized_model.pt')



# backend = "qnnpack"
# yolo_model.qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend
# model_static_quantized = torch.quantization.prepare(yolo_model, inplace=False)
# model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
# model_static_quantized.save('./q_baseline.pt')
#for name, module in yolo_model.named_modules():
#    print(name, type(module))
