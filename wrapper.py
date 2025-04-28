import gymnasium as gym
import minihack
import nle.nethack.actions as actions
import numpy as np

ACTION_LOOKUP = {
    actions.CompassDirection.N : "Move North",
    actions.CompassDirection.E : "Move East",
    actions.CompassDirection.S : "Move South",
    actions.CompassDirection.W : "Move West",
    actions.CompassDirection.NW : "Move Northwest",
    actions.CompassDirection.SE : "Move Southeast",
    actions.CompassDirection.SW : "Move Southwest",
    actions.CompassDirection.NE : "Move Northeast",
    actions.CompassDirectionLonger.N : "Select target to the North",
    actions.CompassDirectionLonger.E : "Select target to the East",
    actions.CompassDirectionLonger.S : "Select target to the South",
    actions.CompassDirectionLonger.W : "Select target to the East",
    actions.CompassDirectionLonger.NW : "Select target to the Northwest",
    actions.CompassDirectionLonger.SE : "Select target to the Southeast",
    actions.CompassDirectionLonger.SW : "Select target to the Southwest",
    actions.CompassDirectionLonger.NE : "Select target to the Northeast",
    actions.MiscDirection.DOWN : "Go up a staircase",
    actions.MiscDirection.UP : "Go down a staircase",
    actions.MiscDirection.WAIT : "Rest one move while doing nothing / Apply to self",
    actions.Command.EXTCMD : "Perform an extended command",
    actions.Command.EXTLIST : "List all extended commands",
    actions.Command.ADJUST : "Adjust inventory letters",
    actions.Command.ANNOTATE : "Name current level",
    actions.Command.APPLY : "Apply (use) a tool (pick-axe, key, lamp...)",
    actions.Command.ATTRIBUTES : "Show your attributes",
    actions.Command.AUTOPICKUP : "Toggle the pickup option on/off",
    actions.Command.CALL : "Call (name) something",
    actions.Command.CAST : "Zap (cast) a spell",
    actions.Command.CHAT : "Talk to someone",
    actions.Command.CLOSE : "Close a door",
    actions.Command.CONDUCT : "List voluntary challenges you have maintained",
    actions.Command.DIP : "Dip an object into something",
    actions.Command.DROP : "Drop an item",
    actions.Command.DROPTYPE : "Drop specific item types",
    actions.Command.EAT : "Eat something",
    actions.Command.ENGRAVE : "Engrave writing on the floor",
    actions.Command.ENHANCE : "Advance or check weapon and spell skills",
    actions.Command.ESC : "Escape from the current query/action",
    actions.Command.FIGHT : "Force fight even if you don't see a monster",
    actions.Command.FIRE : "Fire ammunition from quiver",
    actions.Command.FORCE : "Force a lock",
    actions.Command.GLANCE : "Show what type of thing a map symbol corresponds to",
    actions.Command.HISTORY : "Show long version and game history",
    actions.Command.INVENTORY : "Show your inventory",
    actions.Command.INVENTTYPE : "Inventory specific item types",
    actions.Command.INVOKE : "Invoke an object's special powers",
    actions.Command.JUMP : "Jump to another location",
    actions.Command.KICK : "Kick something",
    actions.Command.KNOWN : "Show what object types have been discovered",
    actions.Command.KNOWNCLASS : "Show discovered types for one class of objects",
    actions.Command.LOOK : "Look at what is here",
    actions.Command.LOOT : "Loot a box on the floor",
    actions.Command.MONSTER : "Use monster's special ability",
    actions.Command.MOVE : "Move without picking up objects/fighting",
    actions.Command.MOVEFAR : "Run without picking up objects/fighting",
    actions.Command.OFFER : "Offer a sacrifice to the gods",
    actions.Command.OPEN : "Open a door",
    actions.Command.OPTIONS : "Show option settings, possibly change them",
    actions.Command.OVERVIEW : "Show a summary of the explored dungeon",
    actions.Command.PAY : "Pay your shopping bill",
    actions.Command.PICKUP : "Pick up things at the current location",
    actions.Command.PRAY : "Pray to the gods for help",
    actions.Command.PUTON : "Put on an accessory (ring, amulet, etc)",
    actions.Command.QUAFF : "Quaff (drink) something",
    actions.Command.QUIT : "Exit without saving current game",
    actions.Command.QUIVER : "Select ammunition for quiver",
    actions.Command.READ : "Read a scroll or spellbook",
    actions.Command.REDRAW : "Redraw screen",
    actions.Command.REMOVE : "Remove an accessory (ring, amulet, etc)",
    actions.Command.RIDE : "Mount or dismount a saddled steed",
    actions.Command.RUB : "Rub a lamp or a stone",
    actions.Command.RUSH : "Rush until something interesting is seen",
    actions.Command.RUSH2 : "Rush until something interesting is seen",
    actions.Command.SAVE : "Save the game and exit",
    actions.Command.SEARCH : "Search for traps and secret doors",
    actions.Command.SEEALL : "Show all equipment in use",
    actions.Command.SEEAMULET : "Show the amulet currently worn",
    actions.Command.SEEARMOR : "Show the armor currently worn",
    actions.Command.SEEGOLD : "Count your gold",
    actions.Command.SEERINGS : "Show the ring(s) currently worn",
    actions.Command.SEESPELLS : "List and reorder known spells",
    actions.Command.SEETOOLS : "Show the tools currently in use",
    actions.Command.SEETRAP : "Show the type of adjacent trap",
    actions.Command.SEEWEAPON : "Show the weapon currently wielded",
    actions.Command.SHELL : "Do a shell escape (not enabled in NLE build)",
    actions.Command.SIT : "Sit down",
    actions.Command.SWAP : "Swap wielded and secondary weapons",
    actions.Command.TAKEOFF : "Take off one piece of armor",
    actions.Command.TAKEOFFALL : "Remove all armor",
    actions.Command.TELEPORT : "Teleport around the level",
    actions.Command.THROW : "Throw something",
    actions.Command.TIP : "Empty a container",
    actions.Command.TRAVEL : "Travel to a specific location on the map",
    actions.Command.TURN : "Turn undead away",
    actions.Command.TWOWEAPON : "Toggle two-weapon combat",
    actions.Command.UNTRAP : "Untrap something",
    actions.Command.VERSION : "List compile time options for this version of NetHack",
    actions.Command.VERSIONSHORT : "Show version",
    actions.Command.WEAR : "Wear a piece of armor",
    actions.Command.WHATDOES : "Tell what a command does",
    actions.Command.WHATIS : "Show what type of thing a symbol corresponds to",
    actions.Command.WIELD : "Wield (put in use) a weapon",
    actions.Command.WIPE : "Wipe off your face",
    actions.Command.ZAP : "Zap a wand",
}



class MiniHackWrapper:
    def __init__(self, env_id="MiniHack-Room-5x5-v0"):
        self.env = gym.make(env_id, observation_keys=("glyphs", "message", "blstats", "chars", "inv_letters", "inv_strs"))
        self.last_obs = None

    # ゲームを1ターン進める
    def step(self, action_idx: int):
        obs, reward, done, truncated, info = self.env.step(action_idx)
        self.last_obs = obs
        return self._make_dict(obs, reward), done

    # ゲームを初期化
    def reset(self):
        obs, _ = self.env.reset()
        self.last_obs = obs
        return self._make_dict(obs, 0)

    # ---------- 内部ヘルパ ----------

    def _make_dict(self, obs, reward):
        """glyphs→ascii にして周囲を文字列化"""
        chars = obs["chars"]  # (height, width)のnp.array
        # 文字列に変換
        board_text = "\n".join(
            "".join(chr(c) for c in row)
            for row in chars
        )
        level = obs["blstats"][2] # プレイヤーレベル
        gold = obs["blstats"][3]  # 所持金
        #strength = obs["blstats"][4] # 筋力
        ac = obs["blstats"][5] # AC
        exp = obs["blstats"][6] # 経験値
        hp = obs["blstats"][10] # HP
        max_hp = obs["blstats"][11] # 最大HP
        pw = obs["blstats"][12] # 魔力
        max_pw = obs["blstats"][13] # 最大魔力
        humger = obs["blstats"][14] # 満腹度
        turn = obs["blstats"][18] # ターン数

        # メッセージ
        msg_bytes = bytes(obs["message"])
        msg = msg_bytes.decode("utf-8").rstrip("\x00")

        inv_items=""
        inv_strs = obs.get("inv_strs",[])
        inv_letters = obs.get("inv_letters",[])
        for index, item_text in enumerate(inv_strs):
            item = bytes(item_text)
            item_str = item.decode("utf-8").rstrip("\x00")
            if item_str == "":
                break
            inv_items += '{}: {}\n'.format(chr(inv_letters[index]), item_str)

        return{
            "level" : level.item(),
            "gold": gold.item(),
            "exp": exp.item(),
            "board_text": board_text,
            "message" : msg,
            "hp" : hp.item(),
            "max_hp" : max_hp.item(),
            "pw" : pw.item(),
            "max_pw" : max_pw.item(),
            "ac" : ac.item(),
            "hunger" : humger.item(),
            "turn" : turn.item(),
            "inventry" : inv_items,
            "reward": reward
        }

    def valid_actions(self):
        return list(range(self.env.action_space.n))

    def get_action_descriptions(self):
        real_env = self.env.unwrapped
        return {
            idx: ACTION_LOOKUP.get(
                real_env.actions[idx],
                f'Unknown Action: {str(real_env.actions[idx])}'
            )
            for idx in self.valid_actions()
        }
    def get_action_description_list(self) -> str:
        desp_list=""
        descriptions = self.get_action_descriptions()
        for idx in descriptions:
            desp_list += f'{idx}: {descriptions[idx]}\n'
        return desp_list