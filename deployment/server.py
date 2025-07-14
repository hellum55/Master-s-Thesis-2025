import os, io, torch
from torch_snippets import P, makedir
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from n1 import CableCabinetClassifier  # Make sure this contains your wrapper class

# ----- Setup device and model -----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load("n1.weights.pth", map_location=device, weights_only=False)
model.eval()


# Mapping from indices to class names
idx_to_class = {
    0: 'BC', 1: 'BC K.M.skab', 2: 'CP1', 3: 'CP3', 4: 'CP4', 5: 'CP6', 6: 'KSE09', 7: 'KSE12', 8: 'KSE15',
    9: 'KSE18', 10: 'KSE21', 11: 'KSE27', 12: 'KSE36/45', 13: 'Kabeldon CDC440/460/420',
    14: 'Kabeldon KSIP423', 15: 'Kabeldon KSIP433', 16: 'Kabeldon KSIP443',
    17: 'Kabeldon KSIP463/KSIP483', 18: 'Kombimodul 2M', 19: 'Kombimodul 3M',
    20: 'Kombimodul 4M', 21: 'MEL1', 22: 'MEL2', 23: 'MEL3', 24: 'MEL4', 25: 'NU',
    26: 'PK20', 27: 'PK35', 28: 'PK48', 29: 'SC'
}

classifier = CableCabinetClassifier(model=model, device=device)

# ----- Setup FastAPI app -----
server_root = P('/tmp')
templates = './templates'
static = server_root / 'server/static'
files = server_root / 'server/files'
for fldr in [static, files]:
    makedir(fldr)

app = FastAPI()
app.mount("/static", StaticFiles(directory=static), name="static")
app.mount("/files", StaticFiles(directory=files), name="files")
templates = Jinja2Templates(directory=templates)

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post('/uploaddata/')
async def upload_file(request: Request, file: UploadFile = File(...)):
    content = file.file.read()
    saved_filepath = f'{files}/{file.filename}'
    with open(saved_filepath, 'wb') as f:
        f.write(content)
    output = classifier.predict_from_path(saved_filepath)
    payload = {
        'request': request,
        "filename": file.filename,
        'output': output
    }
    return templates.TemplateResponse("home.html", payload)

@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content))
    output = classifier.predict_from_image(image)
    return output