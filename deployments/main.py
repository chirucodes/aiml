import logging
import os
import sys
from fastapi import FastAPI
from fastapi import Response
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import status
from fastapi import File
from fastapi import UploadFile

from fastapi.templating import Jinja2Templates

FORMAT = '[%(asctime)-15s] %(filename)s:%(lineno)d %(levelname)-8s- %(message)s'
log_file_handler = logging.FileHandler('./logs/application.log')
console_handler = logging.StreamHandler()
logging.basicConfig(handlers=[log_file_handler, console_handler], format=FORMAT, level=logging.DEBUG)

root_dir = os.getcwd()
sys.path.append(root_dir+"/src")

from src import libs as ml_libs
from src import trained_transformer as ml_trans_libs

cap_gen = ml_libs.CaptionGenerator()
cap_gen.setup()

# cap_tran_gen = ml_trans_libs.CaptionGeneratorTransformer1()
# cap_tran_gen.setup()

app = FastAPI()

app.mount("/static", StaticFiles(directory=root_dir+"/src/ui/static"), name="static")
templates = Jinja2Templates(directory=root_dir+"/src/ui")

@app.get('/root1')
def root(response: Response):
    response.status_code = status.HTTP_200_OK
    logging.info("Root url")
    return {"welcome_message":"Welcome to Automatic Image Captioning project by Group-4 (Cohort-18)"}


@app.post('/caption')
async def get_caption(my_file: UploadFile = File()):
    """
    Uses a model with Attention of optimizer enabled
    """
    logging.info("Storing the input image")

    file_location = f"{root_dir}/input_files/{my_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(my_file.file.read())

    logging.info("Predicting a caption for given image")
    saved_file = root_dir+"/input_files/"+my_file.filename
    caption = cap_gen.get_caption(input_image=saved_file)
    logging.info("Caption generated: %s" % caption)

    return {
        "file": my_file,
        "caption": caption
    }


@app.post('/caption_tf')
async def get_caption_transformer(my_file: UploadFile = File()):
    """
    Uses a model with Transformer
    """
    logging.info("Storing the input image")

    file_location = f"{root_dir}/input_files/{my_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(my_file.file.read())

    logging.info("Predicting a caption for given image")
    saved_file = root_dir+"/input_files/"+my_file.filename
    caption = cap_tran_gen.get_caption(input_image=saved_file)
    logging.info("Caption generated: %s" % caption)

    return {
        "file": my_file,
        "caption": caption
    }

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    logging.info("Storing the input image")
    file_location = f"{root_dir}/input_files/{file.filename}"

    try:
        contents = file.file.read()
        with open(file_location, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    logging.info("Storing the input image")

    logging.info("Predicting a caption for given image")
    saved_file = root_dir+"/input_files/"+file.filename
    caption = cap_gen.get_caption(input_image=saved_file)

    logging.info("Caption generated: %s" % caption)
    file_name1 = file.filename
    file.file.close()

    return {"message": f"Caption: {caption}"}


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
