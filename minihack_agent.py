from wrapper import MiniHackWrapper
import json, os, asyncio, websockets
from typing import Dict
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_agentchat.ui import Console

# ─── WebSocket notifier ───
async def send_state(state: Dict):
    try:
        async with websockets.connect("ws://localhost:8000/ws") as ws:
            await ws.send(json.dumps(state))
    except Exception:
        pass

# ─── MiniHack × AutoGen (RoundRobin only, no Conversation) ───
class MiniHackAgentSystem:
    """Planner → Executor → Introspection を 1 周ずつ回して行動決定"""

    def __init__(self, env_id: str, goal: str):
        self.wrapper = MiniHackWrapper(env_id)
        self.goal = goal
        self.obs = []
        self._build_team()

    def _build_team(self):
        self.load_tool()
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o", 
            api_key=os.getenv("OPENAI_API_KEY"), 
            temperature=0,
            max_tokens=2048)
        self.planner = AssistantAgent(
            name = "MissionPlanner", 
            tools=[self.get_game_goal_tool,
                   self.get_game_map_tool,
                   self.get_game_message_tool],
            description = "An agent that decides what to do to achieve the game goal.",
            system_message=""""
            あなたは、NetHackというゲーム上の行動の計画を立てるエージェントです。
            あなたの主な役割は、ゲーム目標("game goal")とゲームマップ("game map")、ゲームメッセージ("game message")
            を元に次に行う行動を`ActionExecuter`へ提案することです。

            ゲームマップは、文字の位置がそのままプレイヤーや階段の位置を示しています。
            上が北、下が南、左が西、右が東です。視覚的に正しく位置を判断してください。

            **タスク立案の思考プロセス (必須):**
            提案を行う前に、必ず以下の思考プロセスを経過し、その内容を明示的に記述してください。

            1. **現状分析:**
                * `Introspecter`に、目標を達成する上で問題が発生していないか問い合わせる
                * 最終目標達成に向けて、現在何が不足しているか、どのような課題があるかを明確にする
            2. **目標分解と戦略:**
                *   最終目標を達成可能な、より小さなサブゴールに分解する（すでに分解されていれば、次のサブゴールを特定する）。
                *   現在のサブゴールを達成するための、いくつかの可能な戦略やアプローチを検討する
                *   各戦略のリスクや前提条件を考慮し、最も効率的で安全と思われる戦略を選択する
            3.  **具体的タスクの決定:**
                *   選択した戦略に基づいて、次に実行すべき**単一の具体的行動**を決定する。
                *   その行動が、なぜ現時点で最適だと判断したかの根拠を簡潔に述べる。
            決定した行動は`ActionExecuter'へ通知する
            """,
            model_client=self.model_client
        )
        self.executor = AssistantAgent(
            name = "ActionExecuter", 
            tools=[self.get_game_goal_tool,
                   self.get_game_map_tool,
                   self.execute_action_tool,
                   self.get_available_actions_tool],
            description = "An agent that play game.",
            system_message="""
            あなたは、NetHack世界の冒険者エージェントです。
            `MissionPlanner`エージェントからの行動指示、ゲームマップ("game map")から、次に行う行動を決定し、
            利用可能なアクションリスト(available action list)から、その行動を行うためのアクションを選択してください。
            そのアクションの先頭の数字を`execute_action_tool`に対して実行してください。
            ゲームマップは、文字の位置がそのままプレイヤーや階段の位置を示しています。
            上が北、下が南、左が西、右が東です。視覚的に正しく位置を判断してください。
            例:
                `北へ移動する`->`0: Move North`->'0'を使用して`execute_action_tool`を実行
            """, 
            model_client=self.model_client
        )
        self.introspector = AssistantAgent(
            name = "Introspecter", 
            tools=[self.get_game_goal_tool,
                   self.get_game_map_tool],
            description = "An agent that reflects on recent actions",
            system_message="""           
            あなたはNetHackというゲームの冒険者の内心を担当するエージェントです。
            あなたの主な役割は、**過去の実行履歴**、ゲームマップ("game map")、ゲームメッセージ("game message")、
            を元に、スタックするなどの問題が発生していないかを解析し、問題があればその問題の内容を`MissionPlanner`へ連絡してください。
            問題がない場合も、問題は無いことを連絡してください。
            """, 
            model_client=self.model_client
        )

    async def run(self):
        self.obs, done = self.wrapper.reset()
        await send_state(self.obs)
        selector_prompt = """あなたは優秀なリーダーとして、タスクを実行するエージェントを選択してください。

        {roles}

        現在の会話コンテキスト:
        {history}

        上記の会話を読み、{participants}の中から次のタスクを実行するエージェントを選択してください。
        MissionPlannerが他のエージェントの作業開始前にタスクを割り当てていることを確認してください。
        エージェントは1つだけ選択してください。
        """

        termination = TextMentionTermination("TASK FINISHED!")
        team = SelectorGroupChat(
            participants=[self.planner, self.executor, self.introspector],
            selector_prompt=selector_prompt,
            termination_condition=termination,
            allow_repeated_speaker=True,
            model_client=self.model_client,
        )
        await Console(
            team.run_stream(task=self.goal)
        )

    # ------- Tool -------
    def load_tool(self) -> None:
        self.get_game_goal_tool = FunctionTool(
            self.get_game_goal,
            description ="Return game goal"
        )
        self.get_game_map_tool = FunctionTool(
            self.get_game_map,
            description ="Return game map. @ is you."
        )
        self.get_available_actions_tool = FunctionTool(
            self.get_available_actions,
            description ="Return action number and description list."
        )
        self.execute_action_tool = FunctionTool(
            self.execute_action,
            description ="""Play one turn of the game. Pass a action number (int) as the argument."""
        )
        self.get_game_message_tool = FunctionTool(
            self.get_game_message,
            description ="Return game message."
        )

    def get_game_goal(self) -> str:
        """現在のゲーム目的を返す

        Returns:
            str: ゲーム目的
        """
        return self.goal

    def get_game_map(self) -> str:
        """7x7 周囲のゲームマップとプレイヤー・階段の座標差分を返す"""
        board_text = self.obs["board_text"]
        board_lines = board_text.splitlines()
        player = stair = (-1, -1)
        for y, row in enumerate(board_lines):
            for x, ch in enumerate(row):
                if ch == "@":
                    player = (x, y)
                elif ch == ">":
                    stair = (x, y)

        dx, dy = stair[0] - player[0], stair[1] - player[1]
        coord_info = f"Player at {player}, Stair at {stair}, Δ(dx,dy)=({dx},{dy})"

        px, py = player
        cropped = []
        for dy in range(-3, 4):
            y = py + dy
            if 0 <= y < len(board_lines):
                row = board_lines[y]
                x_start = max(0, px - 3)
                x_end = px + 4
                cropped.append(row[x_start:x_end])

        return (
            "Game board:\n```\n"
            + "\n".join(cropped)
            + "\n```\n\n"
            + coord_info
        )

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
            str: "CONTINUE" or "TASK FINISHED!"
        """
        self.obs, done = self.wrapper.step(action_idx)
        await send_state(self.obs)
        return "CONTINUE" if not done else "TASK FINISHED!"

    def get_game_message(self) -> str:
        """ゲームメッセージを返す

        Returns:
            str: ゲームメッセージ
        """
        return f'Msg: {self.obs["message"]}'


# ─── entry ───
if __name__ == "__main__":
    load_dotenv()
    env_id = os.getenv("MINIHACK_TASK", "MiniHack-Room-5x5-v0")
    goal = "Find and descend the staircase (>)."
    asyncio.run(MiniHackAgentSystem(env_id, goal).run())
