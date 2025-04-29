import asyncio
import websockets
import json

from wrapper import MiniHackWrapper


async def send_game_state(state_dict):
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        await websocket.send(json.dumps(state_dict))


def main():
    minihack = MiniHackWrapper("MiniHack-Eat-v0")
    print("How to use:")
    print(minihack.get_action_description_list())
    data, done = minihack.reset()
    asyncio.run(send_game_state(data))

    while not done:
        id = int(input())
        data, done = minihack.step(id)
        asyncio.run(send_game_state(data))


if __name__ == "__main__":
    main()
