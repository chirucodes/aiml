import os

pwd = os.getcwd()

model1_weights = pwd+"/src/models/model_30.h5"
# model1_weights = pwd+"/src/models/model_final_with_attention_with_optimiser.h5"
# o\aiml\deploy_19aug_04\src\models\model_final_with_attention_with_optimiser.h5
# model1_weights = pwd+"/src/models/model_final_with_attention_with_optimiser.h5"
transformer_model1_weights = pwd+"/src/models/save_weights_image_caption_transformer_resnet50.h5"

print("Initiated model1_weights")