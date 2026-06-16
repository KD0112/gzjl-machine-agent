# GitHub + Streamlit Cloud 部署说明

目标网站形态：

```text
https://gzjl-machine-agent.streamlit.app
```

## 重要原则

- `.env` 只能留在本地，不能上传 GitHub。
- GitHub 只放代码、依赖文件、知识库文档。
- DeepSeek API Key 放到 Streamlit Cloud 的 Secrets 里。
- `chroma_db/` 不上传，云端首次启动会根据 `docs/` 自动建立向量库。

## 推荐仓库设置

- Repository name：`gzjl-machine-agent`
- Visibility：Private 或 Public 都可以。实习展示更方便可以用 Public，但不要上传 `.env`。
- Main file path：`app.py`
- Python version：`runtime.txt` 已指定 `python-3.11`

## Streamlit Cloud Secrets

在 Streamlit Cloud 的 App settings -> Secrets 中填写：

```toml
DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
```

## 部署流程

1. 把当前 `day1` 目录推送到 GitHub 仓库 `gzjl-machine-agent`。
2. 打开 Streamlit Community Cloud。
3. 选择 New app。
4. 选择 GitHub 仓库 `gzjl-machine-agent`。
5. Main file path 填 `app.py`。
6. App URL 填 `gzjl-machine-agent`。
7. 添加 Secrets 后点击 Deploy。

首次启动会下载 embedding 模型并建立向量库，可能会慢一些。后续访问会直接读取云端缓存，速度会更稳定。
