from transformers import TrainingArguments
import torch
import time

from causation_rating.custom_dataset import CausalDataset
from causation_rating.text_augmenter import augment_text
from causation_rating.classifier import BERTClassifier
from causation_rating.read_data import load_dataset, write_dataset
from causation_rating import constants


def final_model_training(path_dataset, path_pred_dataset, best_params, run_name = 'DR1'):
    
    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=None,
        num_train_epochs=constants.EPOCH,
        per_device_train_batch_size=constants.BATCH_SIZE,
        per_device_eval_batch_size=constants.BATCH_SIZE,
        warmup_ratio=best_params['warmup_ratio'],
        weight_decay=best_params['weight_decay'],
        learning_rate=best_params['lr'],
        lr_scheduler_type="polynomial",
        lr_scheduler_kwargs={"power": best_params['lr_scheduler_power'], "lr_end": 1e-8},
        logging_dir=None,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="no",
        run_name=None,
        seed=constants.SEED,
    )
    
    print(f"\n*** Final train starts! ***\n")
    
    X, y = load_dataset(path = path_dataset)
    dataset = CausalDataset(X, y)
    
    X_pred, y_pred = load_dataset(path = path_pred_dataset)
    pred_dataset = CausalDataset(X_pred, y_pred)
    
    print(f"-- starting data augmentation...\n")
    time0 = time.time()
    aug_dataset = augment_text(dataset)
    print(f"--- data augmentation time elapsed: {time.time()-time0}")
        
    model = BERTClassifier(
            model_path=constants.MODEL_PATH_PRE, training_args=training_args, loss_err_power=best_params['oll_power']
    )
    
    #wandb.init(
    #        project=WANDB_PROJ,
    #        job_type="train",
    #        name=f'final_model_{run_name}'
    #)
    # Fit model
    model.fit(
        aug_dataset, dataset,
        output_dir=f'./results/final_{run_name}',
        logging_dir=f'./logs/final_{run_name}',
        run_name=f'final_model_{run_name}'
    )
    
    
    
    # Save model
    model.save_model(f'./results/saved_model_{run_name}')
    
    # Predict
    predictions = model.predict(pred_dataset)
    pred_labels = torch.argmax(torch.tensor(predictions.predictions),dim = -1).numpy()
    write_dataset(X_pred, pred_labels, path = f'predicted_{run_name}.csv')
    
    #wandb.finish()
    
    return None
