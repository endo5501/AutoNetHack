import json, os, asyncio, websockets, argparse
from dotenv import load_dotenv
from wrapper import MiniHackWrapper


async def send_game_state(state_dict):
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        await websocket.send(json.dumps(state_dict))


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description='MiniHack program by Manual[DEBUG]')
    parser.add_argument('-e', '--env_name', 
                        help='MiniHack Environment Zoo environment name',
                        default = os.getenv("MINIHACK_TASK", "MiniHack-Room-5x5-v0")
                        )
    args = parser.parse_args()

    minihack = MiniHackWrapper(args.env_name)
    print("Action list:")
    print(minihack.get_action_description_list())
    data, done = minihack.reset()
    asyncio.run(send_game_state(data))

    while not done:
        id = int(input())
        data, done = minihack.step(id)
        asyncio.run(send_game_state(data))


if __name__ == "__main__":
    main()
