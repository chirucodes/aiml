import logging
import os
import sys
from fastapi import FastAPI
from fastapi import Response
from fastapi import status
from fastapi import File
from fastapi import UploadFile

FORMAT = '[%(asctime)-15s] %(filename)s:%(lineno)d %(levelname)-8s- %(message)s'
log_file_handler = logging.FileHandler('./logs/application.log')
console_handler = logging.StreamHandler()
logging.basicConfig(handlers=[log_file_handler, console_handler], format=FORMAT, level=logging.DEBUG)

root_dir = os.getcwd()
sys.path.append(root_dir+"/src")

from src import libs as ml_libs

cap_gen = ml_libs.CaptionGenerator()
cap_gen.setup()

app = FastAPI()


@app.get('/')
def root(response: Response):
    response.status_code = status.HTTP_200_OK
    logging.info("Root url")
    return {"welcome_message":"Welcome to Automatic Image Captioning project by Group-4 (Cohort-18)"}


@app.post('/file')
def get_caption():
    """
        my_file: UploadFile = File(...),
        caption: str = "caption here",
        # first: str = Form(...),
        # second: str = Form("default value  for second"),
        ):
    """
    logging.info("Predicting a caption for given image")

    my_file = root_dir+"\\src\\2090339522_d30d2436f9.jpg"
    caption = cap_gen.get_caption(input_image=my_file)
    logging.info("Caption generated: %s" % caption)

    return {
        "file_name": "my_file.filename",
        "file": my_file,
        "caption": caption
        # "second": second
    }


from starlette.responses import HTMLResponse


@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text Emotion to be tested" />
        <input type="submit" />'''
