# Стилизация видео с помощью нейросетей (ControlNet Video-to-Video)

## Описание проекта

Проект посвящен преобразованию обычного домашнего видео (с играющим котом) в стилизованную визуальную сцену с использованием генеративных моделей. В основе системы лежит метод покадровой обработки Video-to-Video, объединяющий возможности Stable Diffusion и ControlNet. Бот или скрипт не просто накладывает фильтры, а полностью перерисовывает каждый кадр по текстовому запросу (например, превращая обычного кота в киберпанк-робота), жестко сохраняя контуры, движение и исходное поведение объекта. Проект демонстрирует, как нейросети позволяют переосмыслить любые видеоматериалы.

## Используемые технологии

- **Генеративные модели:** Stable Diffusion v1.5, ControlNet (Canny Edge)
    
- **Библиотеки:** `diffusers`, `transformers`, `accelerate`, `opencv-python` (cv2), `controlnet_aux`, `xformers`
    
- **Среда:** Google Colab, GPU (CUDA)
    
- **Данные:** Исходный видеофайл (`cat.mp4`)
    

## Структура проекта

|**Компонент / Файл**|**Назначение**|
|---|---|
|`cat.ipynb`|Главный Jupyter-ноутбук: препроцессинг видео, пайплайн SD + ControlNet, рендеринг|
|`Описание.txt`|Концептуальное описание проекта и идеи|
|`cat.mp4`|Входное домашнее видео с котом (исходные данные)|
|`output.mp4`|Итоговое видео после нейросетевой стилизации|

## Схема пайплайна

Процесс обработки видео пользователя проходит через несколько этапов:

1. **Препроцессинг (Extraction):** Загрузка видео, извлечение кадров, прореживание (изменение шага FPS) и приведение их к размеру 512x512 для оптимизации.
    
2. **Анализ (Edge Detection):** Обработка каждого кадра через алгоритм Canny Detector для выделения жестких контуров и силуэтов движения.
    
3. **Генерация (Style Transfer):** Передача контурного изображения в пайплайн Stable Diffusion + ControlNet с текстовым промптом для генерации стилизованного кадра.
    
4. **Постпроцессинг (Compilation):** Склейка массива сгенерированных изображений обратно в видеоформат (`mp4`) с помощью OpenCV.
    

## Демонстрация работы алгоритма

Ниже представлены примеры того, как трансформируется визуальный ряд:

**Скриншот исходного кадра:**

<img width="1282" height="723" alt="изображение" src="https://github.com/user-attachments/assets/97dd4384-85c0-4791-b935-728b8ba6e6c0" />

**Скриншот стилизованного кадра (Киберпанк):**

<img width="512" height="512" alt="изображение" src="https://github.com/user-attachments/assets/a8076fd2-0be0-465e-9171-4776bc6e2625" />

## Этапы реализации (Детальный разбор кода)

Процесс функционирования нейросети в `cat.ipynb` можно разделить на три технологических этапа:

### 1. Подготовка данных и препроцессинг видео

На этом этапе устанавливается работа с исходным файлом. Мы используем библиотеку OpenCV (`cv2`) для манипуляций с видеопотоком.

- **Что происходит:** Скрипт читает `cat.mp4`, извлекает кадры, берет каждый второй кадр (для ускорения обработки) и меняет их разрешение на 512x512 пикселей. Затем кадры конвертируются в формат PIL для работы с диффузионными моделями.
    
- **Ключевой код:**
    
```Python
import cv2

video_path = "cat.mp4"
cap = cv2.VideoCapture(video_path)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.resize(frame, (512, 512)))

cap.release()
```

### 2. Инициализация моделей (Stable Diffusion + ControlNet)

Этот блок - ядро стилизации. Поскольку обычная Stable Diffusion изменила бы композицию до неузнаваемости, мы подключаем ControlNet.

- **Что происходит:** Загружается предварительно обученная модель ControlNet для работы с краями (`canny`). Затем она интегрируется в общий пайплайн Stable Diffusion v1.5. Для предотвращения нехватки видеопамяти (OOM) активируется оптимизация `xformers`.
    
- **Ключевой код:**

```Python
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
canny = CannyDetector()
```

### 3. Покадровая генерация и сборка видео

Здесь замыкается цепочка: нейросеть применяет заданный стиль к каждому отдельному кадру, сохраняя форму из оригинального видео, после чего кадры объединяются обратно.

- **Что происходит:** Цикл проходит по всем кадрам. `CannyDetector` снимает контуры кота. Затем диффузионная модель по промпту ("cyberpunk robotic cat, neon lights...") генерирует новое изображение. Готовые картинки сшиваются в `output.mp4` через `cv2.VideoWriter`.
    
- **Ключевой код:**

```Python
prompt = "cyberpunk robotic cat, neon lights, glowing eyes, highly detailed"
processed_frames = []

for frame in pil_frames:
    # 1. Извлечение контуров
    canny_image = canny(frame)
    
    # 2. Генерация нового стиля
    result = pipe(
        prompt=prompt,
        image=canny_image,
        num_inference_steps=20,
        guidance_scale=7,
        generator=generator
    ).images[0]
    
    processed_frames.append(result)

# Сборка обратно в видео
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (512, 512))
```

## Как запустить проект

1. Откройте `cat.ipynb` в Google Colab.
    
2. Перейдите во вкладку «Среда выполнения» -> «Сменить среду выполнения» и выберите **T4 GPU** (обязательно для генерации).
    
3. Установите необходимые зависимости через `%pip install`.
    
4. Загрузите исходное видео с названием `cat.mp4` в корневую директорию Colab.
    
5. Последовательно запустите все ячейки. В конце скрипт сгенерирует файл `output.mp4` и предложит его скачать.
