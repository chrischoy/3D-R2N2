# Configurations

For configurations that are not explicitly set in the main.py, you can add the configurations to the arguments by feeding the configuration files to the main file.

## Example

```
python main.py \
  --batch-size 24 \
  --iter 40000 \
  --cfg experiments/cfgs/config.yaml \
  --cfg experiments/cfgs/config2.yaml \
```
