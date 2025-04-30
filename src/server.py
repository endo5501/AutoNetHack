from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import json
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

clients = set()          # つながってるブラウザを全部保持
current_state = {}       # 最新盤面を保存（任意）

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            data = await ws.receive_text()           # ← ゲームランナーから届く
            current_state.update(json.loads(data))   # (任意) 最新を保存
            # -------- ブラウザへ即プッシュ --------
            dead = []
            for c in clients:
                try:
                    await c.send_text(json.dumps(current_state))
                except:
                    dead.append(c)
            for d in dead:
                clients.discard(d)
    except:
        clients.discard(ws)

@app.get("/")
async def get():
    # HTMLページ返す
    # current_stateから盤面を描画
    return HTMLResponse(content=render_game_html(current_state))

def render_game_html(state):
    board = state.get("board_text", "(no board)")
    level = state.get("level", "??")
    hunger = state.get("hunger", "??")
    hp = state.get("hp", "??")
    max_hp = state.get("max_hp", "??")
    pw = state.get("pw", "??")
    max_pw = state.get("max_pw", "??")
    ac = state.get("ac", "??")
    msg = state.get("message", "")
    turn = state.get("turn", "")
    reward = state.get("reward", "")
    inventry = state.get("inventry", "")

    return f"""
<html>
  <head>
    <title>MiniHack Viewer</title>
    <script>
      let socket = new WebSocket("ws://localhost:8000/ws");

        socket.onmessage = (ev) => {{
            const st = JSON.parse(ev.data);
            document.getElementById("level").innerText  = 
                "LEVEL: "+ (st.level ?? "") + " Hunger: " + (st.hunger ?? "");
            document.getElementById("hp").innerText = 
                "HP: " + (st.hp ?? "") + "(" + (st.max_hp ?? "") + "), " +
                "Pw: " + (st.pw ?? "") + "(" + (st.max_pw ?? "") + "), " +
                "AC: " + (st.ac ?? "");
            document.getElementById("turn").innerText = 
                "Turn: " + (st.turn ?? "") + ", Reward: " + (st.reward ?? "");
            document.getElementById("msg").innerText  = "Msg: "+ (st.message ?? "");
            document.getElementById("board").innerText= st.board_text ?? "";
            document.getElementById("inventry").innerText= st.inventry ?? "";
        }};
    </script>
  </head>
  <body style="font-family:monospace; white-space:pre;">
    <div id="level">LEVEL: {level}, Hunger; {hunger}</div>
    <div id="hp">HP: {hp}({max_hp}), Pw: {pw}({max_pw}), AC:{ac}</div>
    <div id="turn">Trun: {turn}, Reward: {reward}</div>
    <div id="msg">Msg: {msg}</div>
    <pre id="board">{board}</pre>
    <pre id="inventry">{inventry}</pre>
  </body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
