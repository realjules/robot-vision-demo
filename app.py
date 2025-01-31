import cv2
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
from datetime import datetime
from PIL import Image
import io
import base64
from transformers import AutoProcessor, LlavaForConditionalGeneration

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class RobotVisionSystem:
    def __init__(self):
        # Initialize LLaVA model
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        self.prompt_template = """Analyze this scene and generate a robot navigation command. 
        Focus on identifying objects, their distances, and safe paths. 
        Generate a JSON response with the following structure:
        {
            "velocity_command": {
                "linear_velocity_mps": float (-1.0 to 1.0),
                "angular_velocity_radps": float (-1.0 to 1.0)
            },
            "gait_mode": string ("trotting", "walking", "standing"),
            "reasoning": string (explanation of the command)
        }
        """

    def process_image(self, image, query=None):
        # Convert CV2 image to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        query = query if query else self.prompt_template
        inputs = self.processor(images=image, text=query, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            command = json.loads(response[json_start:json_end])
            
            # Add timestamp
            command['timestamp'] = datetime.utcnow().isoformat() + "Z"
            
            return command
        except:
            # Return safe command if parsing fails
            return {
                "velocity_command": {
                    "linear_velocity_mps": 0.0,
                    "angular_velocity_radps": 0.0
                },
                "gait_mode": "standing",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "reasoning": "Failed to parse response. Defaulting to safe stop."
            }

# Initialize vision system
vision_system = RobotVisionSystem()

@app.get("/")
async def get_index():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive base64 image from client
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(',')[1])
            
            # Convert to CV2 image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process image
            command = vision_system.process_image(image)
            
            # Send response
            await websocket.send_json(command)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)