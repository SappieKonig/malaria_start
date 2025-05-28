# Introduction of the new team to competitions

We're going to hold a mini-competition, which is just a copy of the original malaria competition
You're actually going to compete, one team against the other, but the goal is to learn as much as possible about how to do competitions.
We're gonna go through all the steps, inspecting data, training models, submitting, and optimizing the results.

## Code structure

First, run
```bash
uv sync
```
to install the dependencies.
If you don't have uv, you can find installation instructions here:
https://docs.astral.sh/uv/getting-started/installation/

## Downloading and preparing data

If you run
```bash
uv run setting_it_up.py
```
it will download the data and prepare it for you. (Preparing meaning converting to YOLO format)

## Training a model

If you run
```bash
uv run train.py
```
it will train a YOLO model on the prepared data.

DON'T TRY TO MAKE YOUR OWN TORCH MODEL OR SOME SHIT, IT WILL TAKE TOO LONG.

## Submitting a prediction

You can test the trained model on the validation set using
```bash
uv run evaluate.py
```

And you can also predict on the test set using
```bash
uv run predict.py
```
and then submit the predictions using
```bash
uv run submit.py
```
Which will send the predictions to a server that I host, and will return the score.
Before running submit.py, make sure to set the NGROK_ID and TEAM_NAME environment variables in .env

