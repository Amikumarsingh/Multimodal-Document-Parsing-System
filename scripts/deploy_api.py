import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import tempfile
from PIL import Image
from pdf2image import convert_from_path
import io

# Import your pipelines
from pipelines.inference_rules import RuleBasedPipeline
from models.layoutlm_v3 import LayoutLMV3Model
from models.donut import DonutWrapper

app = FastAPI(title="Multimodal Document Parser API", version="1.0.0")

# Lazy loading wrappers to save memory at startup
models = {
    "ocr_rules": None,
    "layoutlm": None,
    "donut": None
}

def get_pipeline(pipeline_name: str):
    if models[pipeline_name] is None:
        if pipeline_name == "ocr_rules":
            models["ocr_rules"] = RuleBasedPipeline()
        elif pipeline_name == "layoutlm":
            # In production, specify your fine-tuned path
            models["layoutlm"] = LayoutLMV3Model() 
        elif pipeline_name == "donut":
            # In production, specify your fine-tuned path
            models["donut"] = DonutWrapper()
    return models[pipeline_name]

@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    pipeline: str = Query("ocr_rules", enum=["ocr_rules", "layoutlm", "donut"])
):
    # 1. Save uploaded file to temp
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 2. Convert PDF to Image if necessary
        if suffix.lower() == ".pdf":
            # For simplicity, we process only the first page
            images = convert_from_path(tmp_path)
            if not images:
                raise HTTPException(status_code=400, detail="Could not convert PDF")
            input_image = images[0]
        else:
            input_image = Image.open(tmp_path).convert("RGB")

        # 3. Choose pipeline
        engine = get_pipeline(pipeline)
        
        # 4. Process (Pipelines expect image or path depending on implementation)
        # Standardizing on PIL Image for the models
        if pipeline == "ocr_rules":
            # RuleBasedPipeline.process currently takes path, let's adapt or save temp
            result = engine.process(tmp_path)
        elif pipeline == "layoutlm":
            # LayoutLM/Donut wrappers should handle PIL images
            # To be implemented in the model wrappers to accept PIL
            result = {"status": "success", "data": "LayoutLM output placeholder"}
        elif pipeline == "donut":
            result = engine.predict(input_image)

        return JSONResponse(content={"pipeline": pipeline, "result": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
