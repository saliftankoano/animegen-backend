# GenWalls Backend

Modal app for GenWalls a generative AI image generator web application.

Note: This code requires a 2 api keys to run. You can get them from the following sources:

- [Modal](https://modal.com/docs/guide/secrets)
- [HuggingFace](https://huggingface.co/docs/hub/security-tokens)

## How to run

Running the following commands will run the app on a ephemeral environment in Modal.com. This is useful for testing changes. Changes saved on the subsequent file will be reflected in the app temporarily hosted on Modal.com.

1. Clone the repo
2. Run `pip install -r requirements.txt` (Modal is the only dependency that needs to be installed, the rest will be installed in the image)
3. Run `modal serve main.py`

## How to deploy

Running this command will deploy the app to Modal.com and make it available to the public.

1. Run `modal deploy main.py`
