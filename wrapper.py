import gymnasium as gym
import minihack
import nle.nethack.actions as actions
import numpy as np

ACTION_LOOKUP = {
    actions.CompassDirection.N : "Move North",
    actions.CompassDirection.E : "Move East",
    actions.CompassDirection.S : "Move South",
    actions.CompassDirection.W : "Move East",
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
        self.env = gym.make(env_id, observation_keys=("glyphs", "message", "blstats"))
        self.last_obs = None

    # ゲームを1ターン進める
    def step(self, action_idx: int):
        obs, reward, done, truncated, info = self.env.step(action_idx)
        self.last_obs = obs
        return self._to_text(obs), reward, done

    # ゲームを初期化
    def reset(self):
        obs, _ = self.env.reset()
        self.last_obs = obs
        return self._to_text(obs)

    # ---------- 内部ヘルパ ----------
    def _to_text(self, obs) -> str:
        """glyphs→ascii にして周囲を文字列化、メッセージ・HP なども追記"""
        desc = self.env.render(mode="ansi")  # 1 行文字列
        hp, maxhp = obs["blstats"][10], obs["blstats"][11]
        msg = obs["message"].decode()
        return f"{desc}\nHP:{hp}/{maxhp}\nMsg:{msg}"

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