# StoryGrid_Studio

A LangGraph-based multi-agent system that converts scripts into structured story grids and generates multi-shot long-form videos.

项目分为两部分：

- `backend/`：基于 FastAPI + LangGraph 的多智能体后端服务。
- `frontend/`：基于 React 19 + Vite 的 Web 前端。

---

## 一、环境要求

- Python ≥ 3.12（后端，由 [backend/pyproject.toml](backend/pyproject.toml) 锁定）
- Node.js ≥ 20 LTS、npm ≥ 10（前端，Vite 8 / React 19 要求）
- [uv](https://docs.astral.sh/uv/)（用于管理 Python 虚拟环境与依赖）
- 可访问 OpenAI / 豆包 / Vidu 等服务的网络（国内通常需要代理）

---

## 二、后端准备（backend）

### 1. 安装依赖

后端依赖通过 `pyproject.toml` 管理，虚拟环境位于 `backend/.venv`。

```bash
git clone <your-repo-url>
cd StoryGrid_Studio/backend

uv lock --upgrade
uv sync
```

激活虚拟环境：

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

也可以不激活，直接用 uv 运行：

```powershell
uv run main.py --host 127.0.0.1 --port 8890
```

### 2. 配置 API Key

在运行前填好下列配置文件中的密钥：

- [backend/config/openai/config.yaml](backend/config/openai/config.yaml)
- [backend/config/doubao/config.yaml](backend/config/doubao/config.yaml)
- [backend/config/vidu/config.yaml](backend/config/vidu/config.yaml)

`backend/config/openai/config.yaml` 示例：

```yaml
api_key: your_openai_api_key
model: gpt-5
timeout: 30
max_retries: 3
# 可选：显式上游代理；留空则不走代理
proxy: http://127.0.0.1:7890
# 可选：是否信任系统 / 环境的 HTTP(S)_PROXY，默认 false
trust_env: false
```

> OpenAI HTTP 客户端的代理与超时行为由 [backend/utils/http_client.py](backend/utils/http_client.py) 统一注入，所有 `ChatOpenAI` / `OpenAI` 调用都走该配置，不会被 Windows IE / 系统代理状态影响。建议不要把真实密钥提交到仓库。

### 3. 启动后端

```powershell
cd backend
uv run main.py --host 127.0.0.1 --port 8890
```

服务默认监听 `http://127.0.0.1:8890`，业务路由前缀为 `/api`。

---

## 三、前端准备（frontend）

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 开发模式

```bash
npm run dev
```

Vite 默认监听 `http://127.0.0.1:5173`，并把 `/api` 反向代理到后端 `http://127.0.0.1:8890`（见 [frontend/vite.config.js](frontend/vite.config.js)）。因此本地开发时**先启动后端，再启动前端**。

### 3. 生产构建

```bash
npm run build
```

构建产物会直接输出到 `backend/dist/www`，由后端的 FastAPI StaticFiles 提供静态服务——也就是说生产模式下只需要启动后端进程，无需再单独跑前端。

### 4. 其他常用命令

| 命令 | 说明 |
| --- | --- |
| `npm run preview` | 本地预览构建产物 |
| `npm run lint` | 运行 ESLint 检查 |

---

## 四、目录速览

```
StoryGrid_Studio/
├── backend/                FastAPI + LangGraph 多智能体服务
│   ├── agents/             supervisor / character_generation / shot_generation 等子智能体
│   ├── routers/            FastAPI 路由（包含 SSE 接口）
│   ├── tools/              图像 / 视频 / 提示词等工具集
│   ├── utils/http_client.py 统一构造 OpenAI httpx 客户端（代理 / trust_env）
│   ├── config/             各服务商配置 (openai / doubao / vidu / prompt)
│   ├── main.py             FastAPI 入口
│   └── pyproject.toml
└── frontend/               React 19 + Vite 8 前端
    ├── src/
    ├── vite.config.js      dev 代理 + 构建输出到 backend/dist/www
    └── package.json
```

---

## 五、常见问题

- **`openai.APIConnectionError: Connection error.`**
  通常是本地代理（如 7890）没启动或线路不通。先确认 `Test-NetConnection 127.0.0.1 -Port 7890` 通过，或在 `backend/config/openai/config.yaml` 中调整 `proxy` / `trust_env`。
- **前端访问 `/api/...` 404**
  开发模式下请确保后端 `8890` 已启动；生产模式下需先 `npm run build`，由后端统一提供静态资源。
