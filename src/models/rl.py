""" Testing the A2C Model for the Bank Environment."""


from src.data.definitions import MODEL_PATH, TENSORBOARD_LOGS


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""

    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0."""
        return progress_remaining * initial_value

    return func


def train_model(env, steps=60_000, log_dir="tmp/", ent_coef=0.01, model_name="A2C"):
    """Train the model"""

    model = A2C(
        "MlpPolicy",
        env,
        device="cpu",
        tensorboard_log=TENSORBOARD_LOGS,
        verbose=0,
        n_steps=60,
        learning_rate=linear_schedule(0.001),
        ent_coef=ent_coef,
    )

    model.learn(
        total_timesteps=steps,
        tb_log_name=model_name,
    )

    modelpath = Path(
        MODEL_PATH,
        model_name + "_" + str(steps) + ".zip",
    )
    model.save(modelpath)
    return env, model  # remove to demonstrate saving and loading
