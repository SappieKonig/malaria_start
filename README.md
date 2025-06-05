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

## Sweeping

An incredibly important part of machine learning is sweeping. A different learning rate, batch size, or set of augmentation parameters can make the difference between a meh model and 1st place.
The standard sweeping setup we have is configs using hydra, and sweeping using wandb.

### Hydra

Hydra is a config management tool that allows you to manage different configurations for your code.
It's main feature is the ability to instantiate objects from a config file (which will be more important as your code gets more complex).

But for now we just use it as a simple way to pass arguments to our training script. If you look in train.py, you see we pass the config path using
```python
@hydra.main(config_path="configs", config_name="base_config")
def train_model(config: DictConfig):
```

Which gets the base_config.yaml from configs/base_config.yaml. If you look in there, you see the parameter definitions.
Now, say you want to run train, but as your in a testing phase, you don't need 10 epochs, but only 2.
You can do this by running
```bash
uv run train.py training.num_epochs=2
```

Which will run the training script with num_epochs set to 2.
You can also do this for multiple parameters, for example
```bash
uv run train.py training.num_epochs=2 training.batch_size=8 training.learning_rate=0.0001
```

### WandB

It's inconvenient to try different configurations manually. This is where sweeping (and wandb) come in.
Wandb is a tool that allows you to sweep over different configurations and track the results.
In configs/sweep_config.yaml, you can see the sweep configuration.

To run a sweep, you first have to run
```bash
uv run wandb sweep configs/sweep_config.yaml
```

This will define the sweep and return a sweep command like
```bash
wandb agent <user>/<project_name>/<sweep_id>
```

You can then run this command after prepending `uv run` to start a sweep agent.
This will run the training script with different configurations, and track the results in wandb.

This will go horribly wrong. Because we said to optimize `val_accuracy`, but we never log `val_accuracy`. In the competition, we're optimizing mAP50. So we should probably log that.

Now, the most important thing. Clicking the sweep link (it was posted just after we ran `uv run sweep configs/sweep_config.yaml`) will open the sweep page in wandb, and now we sit back and wait.

## Ensembling with TTA

Once you have trained a few models you can combine their strengths by averaging
their predictions. This is called *ensembling* and usually leads to a small but
reliable performance boost because different models make different mistakes.

Another easy trick is *test time augmentation* (TTA). The idea is simple: run a
model on slightly modified versions of the same image and merge the results. In
this repository we only use flips, because they are fast and deterministic. An
image is predicted four times: as is, horizontally flipped, vertically flipped
and flipped both ways. The boxes are un-flipped back to the original coordinate
frame and written to a single CSV file.

Run the script with
```bash
uv run tta_ensemble.py
```
and edit the `models` list inside `tta_ensemble.py` to point at the model files
you want to ensemble. By default the combined detections are stored in
`tta_submission.csv`.
