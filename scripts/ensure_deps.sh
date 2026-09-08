#!/bin/bash
# 沙箱环境重置后一键恢复依赖（沙箱专用，不入库需求）
pip install -q ruamel.yaml einops gymnasium pyyaml pytest 2>&1 | grep -v notice
pip install -q -e /mnt/agents/output/uwm --no-deps --no-build-isolation 2>&1 | grep -v notice
python3 -c "import ruamel.yaml, einops, gymnasium, uwm; print('deps OK')"
