import argparse
import asyncio
import json
import os
from typing import Dict

import websockets
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

from wrapper import MiniHackWrapper


# ─── WebSocket notifier ───
async def send_state(state: Dict):
    try:
        async with websockets.connect("ws://localhost:8000/ws") as ws:
            await ws.send(json.dumps(state))
    except Exception:
        pass


# ─── MiniHack × AutoGen  ───
class MiniHackAgentSystem:
    """Planner → Executor → Introspection を 1 周ずつ回して行動決定"""

    def __init__(self, env_name: str, goal: str, llm: str):
        self.wrapper = MiniHackWrapper(env_name)
        self.goal = goal
        self.is_debug = False
        self.obs = []
        self._build_team(llm)

    def _get_client_mode(self, llm: str):
        """モデルクライアントを返す

        Args:
            llm (str): llmモデル種類(例:OpenAI/gpt-4o, Ollama/0.0.0.0:11434/gemma3:27b)

        Returns:
            ChatCompletionClient: LLMモデルクライアント
        """
        model_client = None

        if llm == "Debug":
            return None

        llminfo = llm.split("/")
        if len(llminfo) < 2:
            raise ValueError(f"Invalid llm format: {llm}")
        else:
            if llminfo[0] == "OpenAI":
                model_name = llminfo[1]
                model_client = OpenAIChatCompletionClient(
                    model=model_name,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0.3,
                    max_tokens=2048,
                )
            elif llminfo[0] == "Ollama":
                host = f'http://{llminfo[1]}/'
                model_name = llminfo[2]
                print(f'host:{host}, model:{model_name}')

                model_info = {
                    "name": model_name,
                    "json_output": False,
                    "function_calling": True,
                    "family": 'unknown',
                    "vision": False,
                }
                model_client = OllamaChatCompletionClient(
                    model=model_name, host=host, temperature=0.3, model_info=model_info
                )

        return model_client

    def _build_team(self, llm: str):
        self.load_tool()
        self.model_client = self._get_client_mode(llm)
        if self.model_client == None:
            self.is_debug = True
            return
        self.planner = AssistantAgent(
            name="MissionPlanner",
            tools=[
                self.get_game_goal_tool,
                self.get_game_map_tool,
                self.get_game_message_tool,
                self.get_symbol_description_list_tool,
            ],
            description="An agent that decides what to do to achieve the game goal.",
            system_message="""
            You are an agent responsible for planning actions in the game "NetHack".
            /no_think
            Your main role is to analyze the game goal, the game map, past execution history, and game messages, and then propose the **next action** to the ActionExecuter agent.

            🗺️ Game Map:
            The game map is a 7x7 ASCII grid where each character represents a game object (player, monsters, staircase, etc.). 
            If you do not know map symbols, you must use 'get_symbol_description_list_tool'.
            The top of the grid is north, the bottom is south, the left is west, and the right is east. Interpret spatial layout carefully and visually.

            🧠 Required Thought Process (must be explicitly written before giving your final suggestion):

            1. **Situation Analysis**  
                - Determine what is missing or what obstacles are present in achieving the final goal.
                - If you notice repeated actions at the same location, consider asking the `Introspecter` agent to analyze and suggest alternatives.

            2. **Goal Breakdown and Strategy**  
                - Break the final goal into smaller, achievable subgoals (or select the next one if already broken down).  
                - For the current subgoal, brainstorm several possible strategies or approaches.  
                - Consider the risks and assumptions of each, and select the most efficient and safe plan.
                - If you want to attack enemy, go to enemy.
                - Don't stay in one place, always move around.

            3. **Specific Task Decision**  
                - Based on your chosen strategy, determine **one specific next action** to be sent to the `ActionExecuter`.

            🚫 Important Restrictions:
            - You are a **strategic planner**. You **MUST NOT** use any tools or attempt to call any tool functions directly.
            - You **must NOT** mention or suggest specific function names (e.g., `move()`, `search_area()`).
            - Your job is to describe *what* to do, not *how* to do it. The `ActionExecuter` will handle the execution details.
            - If you mention tools or attempt to use them, your response will be ignored.
            """,
            model_client=self.model_client,
        )
        self.executor = AssistantAgent(
            name="ActionExecuter",
            tools=[
                self.get_game_map_tool,
                self.execute_action_tool,
                self.get_available_actions_tool,
            ],
            description="An agent that play game.",
            system_message="""
            You are an adventurer agent in the world of NetHack.
            /no_think
            Your task is to interpret the action intention given by the `MissionPlanner` agent, analyze the current 7x7 game map, and choose the most appropriate executable action from the available action list.

            🗺️ Game Map:
            The game map is a 7x7 ASCII grid. Each character represents the position of the player, items, enemies, staircases, etc.
            North is at the top, South is at the bottom, West is to the left, and East is to the right. Pay close attention to spatial interpretation.

            🎯 Your Goal:
            - Given a high-level instruction (e.g., "descend the staircase"), examine the current map and available actions.
            - From the available actions list, choose the best single action that matches the planner's intent.
            - The available actions are in the form of a numbered list like:  
            `0: Move North`  
            `1: Search Area`  
            ...
            - Execute the appropriate action by calling `execute_action_tool` with the number of the selected action (e.g., `"0"`).

            📌 Example:
            If the planner says "Move north toward the staircase" and the action list includes `0: Move North`, then:
            → You should call `execute_action_tool("0")`.

            ⚠️ Important Rules:
            - Use `get_available_actions_tool` at first
            - You must only select actions from the available action list.
            - Do not hallucinate or invent actions.
            - Always make sure your selected action logically matches the intention of the planner.
            - Do not make strategic decisions yourself. Your role is tactical execution based on existing commands.
            """,
            model_client=self.model_client,
        )
        self.introspector = AssistantAgent(
            name="Introspecter",
            tools=[
                self.get_game_goal_tool,
                self.get_game_map_tool,
                self.get_game_message_tool,
                self.get_all_game_map_tool,
                self.get_symbol_description_list_tool,
            ],
            description="An agent that reflects on recent actions",
            system_message="""
            You are the inner thought agent of an adventurer in the game NetHack.

            Your main role is to analyze the situation by looking at:
            - The past execution history
            - The current 7x7 game map
            - Game messages and events

            Your goal is to determine whether the agent is stuck in a loop, performing ineffective actions, or otherwise encountering a problem in achieving the game goal.

            🧠 If a problem is detected:
            - Clearly describe what the problem is
            - Suggest a potential solution or a change in strategy
            - Report the issue back to the `MissionPlanner` agent

            ✅ If no problem is detected:
            - Report that no issues have been found

            📏 Special tool usage:
            You may use the `get_all_game_map_tool` to obtain a full view of the game world **only if the limited 7x7 map is insufficient to identify the problem**. This tool is resource-intensive and should only be used when necessary.

            🎯 Remember:
            You are not responsible for planning or taking action — your role is **analysis and reflection**.
            You do not directly modify behavior, but report findings to the MissionPlanner for decision-making.
            """,
            model_client=self.model_client,
        )

    async def run(self):
        self.obs, done = self.wrapper.reset()
        await send_state(self.obs)

        if self.is_debug:
            print("--DEBUG--")
            print(self.get_game_map())
            # print(self.get_all_game_map())
            # print(self.execute_action(0))
            return

        selector_prompt = """You are the team leader responsible for selecting which agent should perform the next task.

        {roles}

        Conversation Context:
        {history}

        Based on the above conversation and the current situation, select **one** agent from among {participants} to perform the next task.

        🧠 Important:
        - Ensure that the `MissionPlanner` agent assigns the task **before** other agents begin their work.
        - Choose **only one agent** for the next step.
        /no_think
        """

        termination = TextMentionTermination("TASK FINISHED!")
        team = SelectorGroupChat(
            participants=[self.planner, self.executor, self.introspector],
            selector_prompt=selector_prompt,
            termination_condition=termination,
            # allow_repeated_speaker=True,
            model_client=self.model_client,
        )
        await Console(team.run_stream(task=self.goal))

    # ------- Tool -------
    def load_tool(self) -> None:
        self.get_game_goal_tool = FunctionTool(
            self.get_game_goal, description="Return game goal"
        )
        self.get_game_map_tool = FunctionTool(
            self.get_game_map,
            description="Returns a 7x7 map of the game and the positon of each entity.",
        )
        self.get_all_game_map_tool = FunctionTool(
            self.get_all_game_map,
            description="Returns the overall map of the game and the positon of each entity",
        )
        self.get_available_actions_tool = FunctionTool(
            self.get_available_actions,
            description="Return action number and description list.",
        )
        self.execute_action_tool = FunctionTool(
            self.execute_action,
            description="Play one turn of the game. Pass a action number (int) as the argument.",
        )
        self.get_game_message_tool = FunctionTool(
            self.get_game_message, description="Return game message."
        )
        self.get_symbol_description_list_tool = FunctionTool(
            self.get_symbol_description_list, description="Return game symbol list."
        )

    def get_game_goal(self) -> str:
        """現在のゲーム目的を返す

        Returns:
            str: ゲーム目的
        """
        return self.goal

    def get_game_map(self) -> str:
        """7x7 周囲のゲームマップとプレイヤー・階段の座標を返す"""
        board_text = self.obs["board_text"]
        board_lines = board_text.splitlines()
        pos_list = self._get_player_pos(board_text)
        coord_info = self._get_cord_info(pos_list)
        px, py = self.obs["my_pos"]
        cropped = []
        for dy in range(-3, 4):
            y = py + dy
            if 0 <= y < len(board_lines):
                row = board_lines[y]
                x_start = max(0, px - 3)
                x_end = px + 4
                cropped.append(row[x_start:x_end])

        return "Game board:\n```\n" + "\n".join(cropped) + "\n```\n\n" + coord_info

    def _get_player_pos(self, board_text) -> list:
        """ゲーム盤面上の@などの存在の座標値を返す

        Args:
            board_text (str): 画面盤面

        Returns:
            dict: Dict[key=名前,value=(x,y)]
        """
        board_lines = board_text.splitlines()
        pos_list = []

        target_list = []
        for symbol in self.wrapper.get_symbol_descriptions():
            count = sum(row.count(symbol) for row in board_text)
            if count < 4:
                target_list.append(symbol)

        for y, row in enumerate(board_lines):
            for x, symbol in enumerate(row):
                if symbol in target_list:
                    pos_list.append((symbol, (x, y)))

        return pos_list

    def get_all_game_map(self) -> str:
        board_text = self.obs["board_text"]
        pos_list = self._get_player_pos(board_text)
        coord_info = self._get_cord_info(pos_list)
        return "Game board:\n```\n" + board_text + "\n```\n\n" + coord_info

    def get_available_actions(self) -> str:
        """選択可能な行動リストを返す

        Returns:
            str: 行動リスト
        """
        return self.wrapper.get_action_description_list()

    async def execute_action(self, action_idx: int) -> str:
        """ゲームを実行する

        Args:
            action_idx (int): 行動番号

        Returns:
            str: "CONTINUE:message" or "TASK FINISHED!"
        """
        self.obs, done = self.wrapper.step(action_idx)
        await send_state(self.obs)

        return (
            f'CONTINUE:\n{self.get_game_map()}\nMessage:{self.obs["message"]}'
            if not done
            else "TASK FINISHED!"
        )

    def get_game_message(self) -> str:
        """ゲームメッセージを返す

        Returns:
            str: ゲームメッセージ
        """
        return f'Msg: {self.obs["message"]}'

    def get_symbol_description_list(self) -> str:
        """シンボルのリストを返す

        Returns:
            str: シンボルリスト
        """
        return self.wrapper.get_symbol_description_list()

    def _rel_dir(self, dx: int, dy: int) -> str:
        """Δx,Δy から方位文字 N/E/S/W/NE... を返す"""
        if dx == dy == 0:
            return "HERE"
        dir_x = "E" if dx > 0 else "W" if dx < 0 else ""
        dir_y = "S" if dy > 0 else "N" if dy < 0 else ""
        return dir_y + dir_x  # 例: "N", "SE"

    def _get_cord_info(self, pos_list: list) -> str:
        coord_info = ""
        px, py = self.obs["my_pos"]

        info_parts = []
        info_parts.append(f'HP:{self.obs["hp"]}/{self.obs["max_hp"]}')
        info_parts.append(f"@:HERE(x={px},y={py})")

        for name, (x, y) in pos_list:
            dx, dy = x - px, y - py
            if dx == 0 and dy == 0:
                continue
            info_parts.append(f"{name}:{self._rel_dir(dx,dy)}(dx={dx},dy={dy})")
        coord_info = " ".join(info_parts)

        return coord_info


# ─── entry ───
if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description='Resolve MiniHack program by AutoGen')
    parser.add_argument(
        '-e',
        '--env_name',
        help='MiniHack Environment Zoo environment name',
        default=os.getenv("MINIHACK_TASK", "MiniHack-Room-5x5-v0"),
    )
    parser.add_argument(
        '--llm',
        help='LLM model(example:OpenAI/gpt-4o, Ollama/192.168.2.100:11434/qwen3:14b, default: OpenAI/gpt-4o)',
        default="OpenAI/gpt-4o",
    )
    parser.add_argument(
        '--goal',
        help='Environment goal. Default:"Find and descend the staircase (>)."',
        default="Find and descend the staircase (>).",
    )
    args = parser.parse_args()

    game = MiniHackAgentSystem(args.env_name, args.goal, args.llm)
    asyncio.run(game.run())
