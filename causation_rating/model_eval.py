import time

from transformers import TrainingArguments
from causation_rating.custom_dataset import CausalDataset
from causation_rating.text_augmenter import augment_text
from causation_rating.classifier import BERTClassifier
from causation_rating.read_data import load_dataset
from causation_rating import constants



def final_model_best_param(best_params, X_traineval, y_traineval, test_dt=constants.TEST_DATA_NAME):
    
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
    
    print(f"\n*** Final eval starts! ***\n")
    
    #prepare the entire dataset for 4-fold CV. Due to runtime limitation.
    #Augment the training data
    print(f"-- starting data augmentation...\n")
    time0 = time.time()
    dataset = CausalDataset(X_traineval, y_traineval)
    aug_dataset = augment_text(dataset)
    print(f"--- data augmentation time elapsed: {time.time()-time0}")
    
    X_test, y_test = load_dataset(path = test_dt)
    test_dataset = CausalDataset(X_test, y_test)
    
    model = BERTClassifier(
            model_path=constants.MODEL_PATH_PRE, training_args=training_args, loss_err_power=best_params['oll_power']
    )
    
    #wandb.init(
    #        project=WANDB_PROJ,
    #        job_type="eval",
    #        name='Final_test'
    #)
    # Fit model
    model.fit(
        aug_dataset, test_dataset,
        output_dir='./results/final',
        logging_dir='./logs/final',
        run_name='Final_test'
    )

    # Predict
    predictions = model.predict(test_dataset)
    
    # Compute metrics
    metric = predictions.metrics
    print(f"Final metrics: {metric}\n")
    #wandb.finish()
    
    return metric