from wrapper import MiniHackWrapper

def main():
    print("Hello from auto-nethack!")
    minihack = MiniHackWrapper()

    print(minihack.get_action_description_list())


if __name__ == "__main__":
    main()
