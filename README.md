# OpenEyesClassificator
Тестовое для VisionLabs на CV engineer

Отчет о проделанной работе: 

Создайте виртуальную среду:
```bash
python -m venv venv
```

Активируйте ее:
```bash
source venv/bin/activate
```

Установите необходимые для инференса библиотеки:
```bash
pip install -r requirements.txt
```

Чтобы использовать классифкатор импортируйте из файла OpenEyesClassificator класс OpenEyesClassificator. Также убедитесь, что рядом с файлом `OpenEyesClassificator.py` лежат веса модели ``open_eyes_classifier.pth``.
```python
from OpenEyesClassificator import OpenEyesClassificator
```

Затем вы можете его иницилизовать и использовать его метод predict
```
model = OpenEyesClassificator()
model.predict(inpIm)
```
