import os, base64

from llm import get_agent_response

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

with open('static/index.html', 'r') as file:
    html = file.read()

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        image_path = None
        while True:
            data = await websocket.receive_text()
            if data.startswith("data:image/"):
                image_data = data.split(",")[1]
                filename = "query.png"
                image_path = os.path.join('tmp', filename)

                # Decode the base64 string and save the image
                with open(image_path, "wb") as image_file:
                    image_file.write(base64.b64decode(image_data))

                # Send back the path or URL of the saved image
                response = {"type": "image", "content": image_path}
            else:
                prompt = {
                    'input': str(data)
                }
                if image_path:
                    image_path = image_path.replace('\\', '/')
                    prompt['input'] += f"Image file: {image_path}"
                    print(prompt)
                response = get_agent_response(prompt)
                image_path = ''
                await websocket.send_text(response)
    except WebSocketDisconnect:
        for file in os.listdir('./tmp'):
            os.remove(os.path.join('./tmp', file))
        print("Client disconnected")