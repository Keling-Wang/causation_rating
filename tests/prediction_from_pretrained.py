import causation_rating
from causation_rating.predict_from_pretrained import predict_fromPretrained

import torch

# You always need to set constants before using the model. For details, see the constants.py file and README.md.
causation_rating.set_config(DEVICE = torch.device("cuda"), 
                            #BATCH_SIZE = 128,
                            MODEL_PATH_FINALUSE = 'kelingwang/bert-causation-rating-dr2',
                            )
file_for_pred = 'https://huggingface.co/datasets/kelingwang/causation_strength_rating/resolve/main/all_sentence.csv'
data = predict_fromPretrained(model_path=causation_rating.MODEL_PATH_FINALUSE, 
                       file_for_pred = file_for_pred,
                       write_csv = False,  path_to_save= None)

print(data)