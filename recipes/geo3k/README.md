# Geo3k

Multimodal geometry-reasoning recipe (Geometry-3K dataset). Image-conditioned, single-turn, `\boxed{}` answer. Targets Qwen2.5-VL.

```bash
cd recipes/geo3k/
python data.py
./train_fsdp.sh        # or ./train_megatron.sh
```
