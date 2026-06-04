# StoryGrid_Studio
A LangGraph-based multi-agent system that converts scripts into structured story grids and generates multi-shot long-form videos.

## 环境配置

### 1. clone 后使用 uv 配置环境

项目使用 `pyproject.toml` 管理依赖，要求 Python 3.12 及以上。

```powershell
git clone <your-repo-url>
cd StoryGrid_Studio

# 同步虚拟环境
uv lock --upgrade
uv sync
```

激活虚拟环境：

```bash
source .venv/bin/activate
```

如果只想直接通过 uv 运行命令，也可以不手动激活环境：

```powershell
uv run main.py --host 127.0.0.1 --port 8890
```

### 2. 配置 API Key

项目中的 OpenAI 相关逻辑默认读取 [config/openai/config.yaml](config/openai/config.yaml) 里的配置。运行前请先把其中的 `api_key` 改成自己的可用密钥，例如：

```yaml
api_key: your_openai_api_key
model: gpt-5
timeout: 30
max_retries: 3
```

还会用到其他提供方，也需要同步检查并填写对应配置文件中的密钥，例如：

- [config/doubao/config.yaml](config/doubao/config.yaml)
- [config/vidu/config.yaml](config/vidu/config.yaml)

建议不要把真实密钥提交到仓库。
