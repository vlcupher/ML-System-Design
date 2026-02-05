# system_ml_v4

Используется **Qwen3-VL-2B**.

- Препроцессинг → тайлинг → извлечение по тайлам → two-pass (extract → merge → structure) → ответ API.
- Типы диаграмм: BPMN, UML, C4, OTHER.
- Модель: Qwen3-VL-2B (по умолчанию полная `Qwen/Qwen3-VL-2B-Instruct` с 4-bit через BitsAndBytes — все веса, включая `lm_head`, загружаются корректно).

## Запуск

```bash
cd system_ml_v4
pip install -r requirements-cpu.txt
export PYTHONPATH=src
python -m diagram_service.main
```

- API: http://localhost:8000  
- UI: http://localhost:8000/ui  
- Метрики: http://localhost:8000/metrics  

Переменные окружения: `DEVICE` (cpu/auto), `QWEN3VL_MODEL_ID`, `QWEN3VL_PROCESSOR_ID`, `QWEN3VL_4BIT` (1 = 4-bit на GPU, 0 = полная точность), `PORT`, `TWO_PASS`.

**CPU:** при `DEVICE=cpu` всегда грузится полная модель (без 4-bit). Если в `QWEN3VL_MODEL_ID` указан 4-bit чекпоинт, на CPU автоматически подставляется полная `Qwen/Qwen3-VL-2B-Instruct`, иначе возможны пустой вывод и предупреждения bitsandbytes.

## Docker

### CPU

```bash
docker build -f docker/Dockerfile.cpu -t diagram-v4:cpu .
docker run -p 8000:8000 -e DEVICE=cpu diagram-v4:cpu
```

### GPU

Требуется [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Образ с PyTorch CUDA 12.1:

```bash
docker build -f docker/Dockerfile.gpu -t diagram-v4:gpu .
docker run --gpus all -p 8000:8000 diagram-v4:gpu
```
