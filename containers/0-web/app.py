"""Container A: mobile/web gateway. Inference belongs to container B."""
from __future__ import annotations
import io, json, os, struct, uuid
from pathlib import Path
import cv2, numpy as np, qrcode, websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT=Path(__file__).resolve().parent; WEB_DIR=ROOT/"web"
VISION_URL=os.getenv("VISION_ANALYSIS_WS_URL","ws://127.0.0.1:8760/ingest/{session_id}")
TOKEN=os.getenv("SESSION_TOKEN","hand-board"); PUBLIC=os.getenv("PUBLIC_BASE_URL","").rstrip("/")
clients: dict[str,set[WebSocket]]={}

def pack(meta:dict,data:bytes)->bytes:
    raw=json.dumps(meta,ensure_ascii=False,separators=(",",":")).encode(); return struct.pack(">I",len(raw))+raw+data
def unpack(payload:bytes)->tuple[dict,bytes]:
    if len(payload)<5: raise ValueError("short frame")
    n=struct.unpack(">I",payload[:4])[0]
    if n<=0 or 4+n>=len(payload): raise ValueError("invalid header length")
    return json.loads(payload[4:4+n]),payload[4+n:]
async def publish(session:str,payload:bytes)->None:
    stale=[]
    for ws in set(clients.get(session,set())):
        try: await ws.send_bytes(payload)
        except Exception: stale.append(ws)
    for ws in stale: clients.get(session,set()).discard(ws)

app=FastAPI(title="Remote Drawing Web Gateway")
@app.get("/",include_in_schema=False)
@app.get("/index.html",include_in_schema=False)
@app.get("/monitor",include_in_schema=False)
async def monitor_page(): return FileResponse(WEB_DIR/"index.html")
@app.get("/capture.html",include_in_schema=False)
async def capture_page(): return FileResponse(WEB_DIR/"capture.html")
@app.get("/api/config",include_in_schema=False)
async def config(): return JSONResponse({"session_token":TOKEN,"public_base_url":PUBLIC})
@app.get("/qr",include_in_schema=False)
async def qr(u:str):
    if not u.startswith(("http://","https://")): raise HTTPException(400,"invalid URL")
    image=qrcode.make(u); out=io.BytesIO(); image.save(out,"PNG"); return Response(out.getvalue(),media_type="image/png")
@app.get("/health")
async def health(): return {"status":"ok","role":"A-web-gateway"}

@app.websocket("/ws/camera")
async def camera(ws:WebSocket,t:str=""):
    if t!=TOKEN: await ws.close(code=1008,reason="invalid token"); return
    await ws.accept(); vision_url=VISION_URL.format(session_id=t)
    try:
        async with websockets.connect(vision_url,max_size=None) as vision:
            while True:
                meta,jpeg=unpack(await ws.receive_bytes())
                frame=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                if frame is None: continue
                h,w=frame.shape[:2]; frame_id=str(meta.get("frame_id") or uuid.uuid4())
                header={
                    "session_id":t,
                    "frame_id":frame_id,
                    "seq":int(meta.get("seq",0)),
                    "capture_ts":int(meta.get("captured_at_ms",0)),
                    "width":w,
                    "height":h,
                    "format":"bgr8",
                    "rotation":0,
                    "mirrored":False,
                }
                # A→B: exactly one TEXT JSON header, then one BINARY raw BGR frame.
                await vision.send(json.dumps(header,separators=(",",":"))); await vision.send(frame.tobytes(order="C"))
                await publish(t,pack({**header,"kind":"source"},jpeg))
    except WebSocketDisconnect: pass
    except Exception as exc: await ws.send_text(json.dumps({"error":str(exc)}))

@app.websocket("/ws/canvas-output/{session_id}")
async def canvas_output(ws:WebSocket,session_id:str):
    await ws.accept()
    try:
        while True: await publish(session_id,await ws.receive_bytes())
    except WebSocketDisconnect: pass
@app.websocket("/ws/monitor")
async def monitor(ws:WebSocket,t:str=""):
    if t!=TOKEN: await ws.close(code=1008,reason="invalid token"); return
    await ws.accept(); clients.setdefault(t,set()).add(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: clients.get(t,set()).discard(ws)
