from fastapi import  FastAPI
from fastapi import  Response
from fastapi import  status
from fastapi import File
from fastapi import UploadFile

app = FastAPI()


@app.get('/')
def root(response: Response):
    response.status_code = status.HTTP_200_OK
    print("Root url")
    return {"welcome_message":"Welcome to Automatic Image Captioning project by Group-4 (Cohort-18)"}


@app.post('/file')
def get_caption(
        my_file: UploadFile = File(...),
        caption: str = "caption here",
        # first: str = Form(...),
        # second: str = Form("default value  for second"),
        ):
    print("Predicting the caption for given image")
    return {
        "file_name": my_file.filename,
        "file": my_file,
        "caption": caption
        # "second": second
    }