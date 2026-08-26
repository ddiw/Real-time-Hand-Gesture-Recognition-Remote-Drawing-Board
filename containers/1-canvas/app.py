"""Canvas service on 8770. Consumes C command JSON without changing gesture rules."""
from __future__ import annotations
import json,os,struct
import cv2,websockets
from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from drawing_canvas import DrawingCanvas
OUTPUT=os.getenv("WEB_CANVAS_OUTPUT_URL","ws://127.0.0.1:8000/ws/canvas-output/{session_id}")
app=FastAPI(title="Drawing Canvas"); canvases={}
def pack(meta,data):
    raw=json.dumps(meta,separators=(",",":")).encode(); return struct.pack(">I",len(raw))+raw+data
def point(packet):
    p=packet.get("index_tip") or packet.get("pointer")
    if not isinstance(p,dict) or not all(isinstance(p.get(k),(int,float)) for k in ("x","y")): return None
    x,y=p["x"],p["y"]; return (round(x*359),round(y*639)) if 0<=x<=1 and 0<=y<=1 else (round(x),round(y))
@app.get("/health")
async def health(): return {"status":"ok","sessions":len(canvases)}
@app.websocket("/commands/{session_id}")
async def commands(ws:WebSocket,session_id:str):
    await ws.accept(); canvas=canvases.setdefault(session_id,DrawingCanvas(360,640))
    try:
        async with websockets.connect(OUTPUT.format(session_id=session_id),max_size=None) as output:
            while True:
                packet=json.loads(await ws.receive_text()); command=str(packet.get("command","IDLE"))
                # EXPORT is a monitor request, not a gesture. It returns the stored
                # drawing at full quality, without the zoom label or the cursor.
                if command=="EXPORT":
                    ok,png=cv2.imencode(".png",canvas.export())
                    if ok: await output.send(pack({"kind":"export","session_id":session_id,"zoom":round(canvas.zoom,3)},png.tobytes()))
                    continue
                p=point(packet)
                d=packet.get("index_direction") or {}; direction=(float(d["x"]),float(d["y"])) if "x" in d and "y" in d else None
                canvas.apply(command,p,direction); ok,jpeg=cv2.imencode(".jpg",canvas.render(),[cv2.IMWRITE_JPEG_QUALITY,88])
                if ok: await output.send(pack({"kind":"canvas","session_id":session_id,"frame_id":packet.get("frame_id"),"seq":packet.get("seq"),"command":command,"mode":packet.get("mode","IDLE"),"zoom":round(canvas.zoom,3),"inference_ms":packet.get("inference_ms"),"finger_count":packet.get("finger_count"),"landmarks":packet.get("landmarks")},jpeg.tobytes()))
    except WebSocketDisconnect: canvas.hide_cursor()
