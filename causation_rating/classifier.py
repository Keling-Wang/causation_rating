from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments
from sklearn.base import BaseEstimator, ClassifierMixin

from causation_rating.oll_loss_trainer import OLLTrainer
from causation_rating.metrics import compute_metrics
from causation_rating import constants

import os, time

# Modify BERTClassifier to accept callbacks
class BERTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_path, num_labels=5, aug_factor=7, training_args=None, loss_err_power=2.0):
        self.model_path = model_path
        self.num_labels = num_labels
        self.aug_factor = aug_factor
        self.training_args = training_args if training_args is not None else TrainingArguments(output_dir="./results")
        self.loss_err_power = loss_err_power

    def fit(self, train_dataset, eval_dataset, output_dir=None, logging_dir=None, run_name=None, callbacks=None):
        # Initialize model and tokenizer
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path, num_labels=self.num_labels
        ) 
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model.dist_matrix = constants.DIST_MATRIX.to(self.model.device)

        # Prepare dataset
        #train_dataset = CausalDataset(
        #    X_t, y_t, self.tokenizer, num_classes=self.num_labels)
        #eval_dataset = CausalDataset(
        #    X_e, y_e, self.tokenizer, num_classes=self.num_labels)
        
        # Augment the training data
        #print(f"-- starting data augmentation...\n")
        #time0 = time.time()
        #augmented_dataset = augment_text(train_dataset, aug_factor=self.aug_factor, model_path=MODEL_PATH_AUG)
        #print(f"--- data augmentation time elapsed: {time.time()-time0}")
        self.training_args.output_dir = output_dir if self.training_args.output_dir is None else self.training_args.output_dir
        self.training_args.logging_dir = logging_dir if self.training_args.logging_dir is None else self.training_args.logging_dir
        self.training_args.run_name = run_name if self.training_args.run_name is None else self.training_args.run_name

        # Initialize trainer
        self.trainer = OLLTrainer(
            err_power=self.loss_err_power,
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks  # Pass the callbacks here
        )
        
        time0 = time.time()
        # Train the model
        self.trainer.train()
        print(f"--- This run time elapsed: {time.time()-time0}")
        return self

    def predict(self, eval_dataset):
        # Prepare dataset for prediction
        #dataset = CausalDataset(
        #    X, y, self.tokenizer, num_classes=self.num_labels
        #)

        # Make predictions
        #trainer = OLLTrainer(err_power=self.loss_err_power,
        #                     model=self.model, 
        #                     args=TrainingArguments(output_dir="./tmp"),
        #                     tokenizer=self.tokenizer,
        #                     compute_metrics=compute_metrics)
        
        predictions = self.trainer.predict(eval_dataset)
        return predictions
    
    def save_model(self, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        # Save model and tokenizer
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}")