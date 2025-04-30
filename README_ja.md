# AutoGen-MiniHack-Agent

## 概要

このプロジェクトは、AIエージェントフレームワークであるAutoGenを使用して、NetHackの強化学習環境であるMiniHack上でキャラクターを自動操作する実験プロジェクトです。  
[こちら](https://github.com/Mega-Gorilla/Discovery)のMinecraft自動操作エージェントに触発されて作成しました。

## インストール方法

以下については Python 3.10 で動作確認済みです。

### 準備

1. uv をインストールする
2. MiniHack のインストール準備（[NLE のインストール手順](https://github.com/heiner/nle#installation)）  
   - Mac:
     ```bash
     brew install cmake
     ```
   - Linux:
     ```bash
     sudo apt-get install -y build-essential autoconf libtool pkg-config \
     python3-dev python3-pip python3-numpy git flex bison libbz2-dev cmake
     ```

### パッケージインストール

1. 本プロジェクトをクローンする
    ```bash
    git clone git@github.com:endo5501/AutoNetHack.git
    ```
2. プロジェクトのディレクトリに移動し、以下を実行して仮想環境を作成する
    ```bash
    uv venv
    source ./.venv/bin/activate
    ```
3. 以下を実行して、必要なパッケージをインストール
    ```bash
    uv sync
    ```

### 設定ファイル（.env）について

`.env` ファイルを用意し、以下のように設定してください：

```
AUTOGEN_USE_DOCKER=False
OPENAI_API_KEY=sk-...
```

## 実行方法

### 監視用サーバを起動

エージェントの行動や状態をブラウザで確認できる Web サーバを起動します。  
サーバは [http://localhost:8000](http://localhost:8000) でアクセスできます。

```bash
python src/server.py
```

### OpenAI の API を使用して実行

以下を実行すると、5 x 5 の部屋が生成され、対角上の階段 `>` へ移動するタスクが作成されます。

```bash
python src/minihack_agent.py
```

他のタスクを実行したい場合は、[MiniHack Environment Zoo](https://minihack.readthedocs.io/en/latest/envs/index.html) に記載されている Environment Name を指定してください。

例：[ランダムに生成された部屋の一つにある階段を探して移動する](https://minihack.readthedocs.io/en/latest/envs/navigation/corridor.html)
```bash
python src/minihack_agent.py --env_name MiniHack-Corridor-R2-v0
```

> **注意：**
> - OpenAI の gpt-4o などでも地図の読み取りに失敗してウロウロし続けることがあるため、API 使用料に注意してください。
> - "Navigation Tasks" 以下のタスクは階段まで移動するだけですが、それ以外は特定の行動（例：りんごを食べる）が必要です。目的を設定する必要があります（基本的に未実装です）。

例：
```bash
python src/minihack_agent.py --env_name MiniHack-Eat-v0 --goal "eat an apple"
```

### 手動で MiniHack Environment Zoo のタスクを確認する

タスクをエージェントに実行させる前に内容を確認したい場合は、以下を実行してマニュアル操作で動作を確認できます。

例：
```bash
python src/manual_test.py --env_name MiniHack-Corridor-R2-v0
```

### Ollama を利用して実行

引数 `--llm` を指定することで、Ollama のローカルモデルを利用可能です。

例：
```bash
python src/minihack_agent.py --llm Ollama/192.168.2.100:11434/qwen3:14b
```

> **注意：**
> 使用できるのは function calling に対応したモデルに限られます。
>
> 対応モデル例：
> - qwen3:14b  
> - phi4-mini:latest

