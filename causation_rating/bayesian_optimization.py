import optuna
from transformers import TrainingArguments
from sklearn.model_selection import StratifiedKFold

import numpy as np
import time


from causation_rating.custom_dataset import CausalDataset
from causation_rating.text_augmenter import augment_text
from causation_rating.custom_callback import TransformersPruningCallback
from causation_rating.classifier import BERTClassifier
from causation_rating.read_data import load_dataset, split_dataset
from causation_rating.model_eval import final_model_best_param
from causation_rating import constants



def objective(trial: optuna.trial, X, y, prune_threshold=70, prune_step=20):
    # Set up the hyperparameters to search
    lr = trial.suggest_float("lr", constants.LR_SEARCH[0], constants.LR_SEARCH[1], log=True)
    weight_decay = trial.suggest_float("weight_decay", constants.WEIGHT_DECAY_SEARCH[0], constants.WEIGHT_DECAY_SEARCH[1])
    warmup_ratio = trial.suggest_float("warmup_ratio", constants.WARMUP_RATIO_SEARCH[0], constants.WARMUP_RATIO_SEARCH[1])
    lr_scheduler_power = trial.suggest_float(
        "lr_scheduler_power", constants.LR_SCHEDULER_POWER_SEARCH[0], constants.LR_SCHEDULER_POWER_SEARCH[1]
    )
    oll_power = trial.suggest_categorical("oll_power", constants.OLL_POWER_SEARCH)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=None,
        num_train_epochs=constants.EPOCH,
        per_device_train_batch_size=constants.BATCH_SIZE,
        per_device_eval_batch_size=constants.BATCH_SIZE,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        learning_rate=lr,
        lr_scheduler_type="polynomial",
        lr_scheduler_kwargs={"power": lr_scheduler_power, "lr_end": 1e-8},
        logging_dir=None,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        run_name=None,
        seed=constants.SEED,
    )

    print(f"\n*** Trial {trial.number} starts with param: ***\n   ----- Warmup R:{warmup_ratio:.4f}; WeightDecay:{weight_decay:.4f}; LR: {lr}; Power: {lr_scheduler_power:.4f}; Loss power: {oll_power}\n ****  ***\n")
    
    #prepare the entire dataset for 4-fold CV. Due to runtime limitation.
    #Augment the training data
    print(f"-- starting data augmentation...\n")
    time0 = time.time()
    dataset = CausalDataset(X, y)
    aug_dataset = augment_text(dataset)
    print(f"--- data augmentation time elapsed: {time.time()-time0}")
    
    
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=constants.SEED)
    scores = []
    fold_idx = 0
    step_offset = 0  # Initialize step_offset

    for train_index, val_index in kf.split(aug_dataset.texts, aug_dataset.labels):
        fold_idx += 1
        X_train, X_val = [aug_dataset.texts[i] for i in train_index], [aug_dataset.texts[i] for i in val_index]
        y_train, y_val = [aug_dataset.labels[i] for i in train_index], [aug_dataset.labels[i] for i in val_index]
        
        train_dataset = CausalDataset(X_train, y_train)
        eval_dataset = CausalDataset(X_val, y_val)
        
        # Initialize model
        model = BERTClassifier(
            model_path=constants.MODEL_PATH_PRE, training_args=training_args, loss_err_power=oll_power
        )

        # Custom callback for pruning with step_offset
        pruning_callback = TransformersPruningCallback(
            trial,
            'eval_off_by_1_acc',
            prune_step,
            prune_threshold,
            step_offset=step_offset
        )
        callbacks = [pruning_callback]

        #wandb.init(
        #    project=WANDB_PROJ,
        #    job_type="train",
        #    name=f'trial_{trial.number}_fold_{fold_idx}',
        #) # if using wandb, please uncomment this line

        # Fit model
        model.fit(
            train_dataset, eval_dataset,
            output_dir=f'./results/trial_{trial.number}_fold_{fold_idx}',
            logging_dir=f'./logs/trial_{trial.number}_fold_{fold_idx}',
            run_name=f'trial_{trial.number}_fold_{fold_idx}',
            callbacks=callbacks
        )

        # Predict
        predictions = model.predict(eval_dataset)

        # Compute metrics
        metric = predictions.metrics
        score = metric['test_off_by_1_acc']
        scores.append(score)
        print(f"Trial {trial.number} Fold {fold_idx} Scores: {metric}\n")
        #wandb.finish()

        # Report intermediate value to Optuna with unique step
        intermediate_value = np.mean(scores)
        step = step_offset + 10000 + fold_idx  # Ensure uniqueness
        trial.report(intermediate_value, step=step)

        # Handle pruning
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at fold {fold_idx}")
            raise optuna.exceptions.TrialPruned()

        # Update step_offset with total steps taken in this fold
        fold_total_steps = model.trainer.state.global_step
        step_offset += fold_total_steps

    # Return the average score over all folds
    return np.mean(scores)

def BayesianOptimization(objective, study_name, n_trials=50):
    X, y = load_dataset(random_sample = False)
    X_splitted, y_splitted = split_dataset(X, y, test_size=0.25)
   
    study = optuna.create_study(direction='maximize',study_name=study_name)
    study.optimize(lambda trial: objective(trial, X_splitted, y_splitted, prune_threshold = 60, prune_step = 60), 
                   n_trials=n_trials, timeout=41400)
    
    final_metric = final_model_best_param(
        study.best_params, X_splitted, y_splitted, test_dt=constants.TEST_DATA_NAME
    )

    return study.best_params, final_metric