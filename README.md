# AutoGen-MiniHack-Agent

## Overview

This project is an experimental system that uses the AutoGen framework to autonomously control a character in the MiniHack environment, which is based on the classic roguelike NetHack and tailored for reinforcement learning.  
It was inspired by [Discovery: Customizable Minecraft Agent with AutoGen](https://github.com/Mega-Gorilla/Discovery).

## Installation

Tested on Python 3.10.

### Prerequisites

1. Install `uv`
2. Prepare the MiniHack environment (follow the [NLE installation instructions](https://github.com/heiner/nle#installation))

- macOS:
    ```bash
    brew install cmake
    ```
- Linux:
    ```bash
    sudo apt-get install -y build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git flex bison libbz2-dev cmake
    ```

### Installing Packages

1. Clone this repository:
    ```bash
    git clone git@github.com:endo5501/AutoNetHack.git
    ```
2. Move into the project directory and create a virtual environment:
    ```bash
    uv venv
    source ./.venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    uv sync
    ```

### Environment Configuration (.env)

Create a `.env` file and set the following environment variables:

```
AUTOGEN_USE_DOCKER=False
OPENAI_API_KEY=sk-...
```

## How to Run

### Launch Monitoring Server

Start a web server to view the agent's state and behavior in real-time.  
Access it at [http://localhost:8000](http://localhost:8000).

```bash
python src/server.py
```

### Run with OpenAI API

Run the following to generate a 5x5 room with a goal to move to the downward staircase `>`:

```bash
python src/minihack_agent.py
```

To use a different task, specify an environment name from the [MiniHack Environment Zoo](https://minihack.readthedocs.io/en/latest/envs/index.html).

Example: [Navigate through randomly generated rooms to find a staircase](https://minihack.readthedocs.io/en/latest/envs/navigation/corridor.html)
```bash
python src/minihack_agent.py --env_name MiniHack-Corridor-R2-v0
```

> **Note:**
> - Even with models like OpenAI's `gpt-4o`, agents might fail to parse the map correctly and wander endlessly, so keep an eye on your API usage.
> - Tasks listed under "Navigation Tasks" only require movement to a staircase, but others (e.g. "Eat an apple") may require goal setting. These are generally not implemented yet.

Example:
```bash
python src/minihack_agent.py --env_name MiniHack-Eat-v0 --goal "eat an apple"
```

### Test Environments Manually

To inspect a task manually before assigning it to an agent, use:

```bash
python src/manual_test.py --env_name MiniHack-Corridor-R2-v0
```

### Run with Ollama (Local LLM)

Specify the `--llm` option to use an Ollama-hosted model:

Example:
```bash
python src/minihack_agent.py --llm Ollama/192.168.2.100:11434/qwen3:14b
```

> **Note:**
> Only models that support function calling can be used.
>
> Example compatible models:
> - qwen3:14b  
> - phi4-mini:latest

