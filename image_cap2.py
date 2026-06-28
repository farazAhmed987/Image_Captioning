

# #importing the libraries
# # Run this in a Python shell or script
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')

# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow
# from tqdm import tqdm
# from nltk.translate.bleu_score import sentence_bleu
# from numpy import array
# import pickle
# import numpy as np
# import os
# import cv2

# from tensorflow.keras.preprocessing.text import Tokenizer # Using Keras Tokenizer can be more convenient
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.callbacks import ModelCheckpoint

# # ## Data Visualization and Preprocessing

# # Define paths
# captions_base_path = "all_captions/"
# images_base_path = "all_images/Flicker8k_Dataset/"
# pickled_features_path = "pickled_features/"
# model_checkpoint_path = "model_checkpoints/"

# # Create directories if they don't exist
# os.makedirs(pickled_features_path, exist_ok=True)
# os.makedirs(model_checkpoint_path, exist_ok=True)


# #converting the text files to pandas dataframe
# try:
#     image_tokens=pd.read_csv(os.path.join(captions_base_path, "Flickr8k.lemma.token.txt"),sep='\t',names=["img_id","img_caption"])
#     train_image_names=pd.read_csv(os.path.join(captions_base_path, "Flickr_8k.trainImages.txt"),names=["img_id"])
#     test_image_names=pd.read_csv(os.path.join(captions_base_path, "Flickr_8k.testImages.txt"),names=["img_id"])
#     val_image_names=pd.read_csv(os.path.join(captions_base_path, "Flickr_8k.devImages.txt"),names=["img_id"])
# except FileNotFoundError as e:
#     print(f"Error: Caption file not found. Make sure '{captions_base_path}' directory is correctly populated.")
#     print(e)
#     exit()

# #just checking the number of images in the dataset
# try:
#     l=os.listdir(images_base_path)
#     print(f"Number of images found: {len(l)}")
# except FileNotFoundError:
#     print(f"Error: Image directory '{images_base_path}' not found. Please check the path.")
#     exit()



# #visualising one of the train images
# if not train_image_names.empty:
#     try:
#         img_path_to_show = os.path.join(images_base_path, train_image_names.img_id[0])
#         img_show = cv2.imread(img_path_to_show)
#         if img_show is not None:
#             plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
            
#             plt.savefig("sample_train_image.png")
#             print("Saved sample_train_image.png")
#             plt.close()
#         else:
#             print(f"Warning: Could not read sample image at {img_path_to_show}")
#     except Exception as e:
#         print(f"Error displaying sample image: {e}")
# else:
#     print("No training images to display.")





# #removing the #0,#1,#2,#3,#5 from the image ids
# image_tokens["img_id"]=image_tokens["img_id"].map(lambda x: x.split('#')[0]) # More robust way to remove suffix





# image_tokens["img_caption"]=image_tokens["img_caption"].map(lambda x: "<start> " + x.strip() + " <end>")





# #head of the image_tokens dataframe
# print("Image Tokens Head:\n", image_tokens.head())


# # In[ ]:


# #head of the train_image_names dataframe
# print("Train Image Names Head:\n", train_image_names.head())


# # ### Creating dictionaries to map image_id and their corresponding captions



# def create_captions_dictionary(image_names_df, all_image_tokens_df):
#     captions_dict = {}
#     for i in tqdm(range(len(image_names_df)), desc="Creating Captions Dictionary"):
#         img_id = image_names_df["img_id"].iloc[i]
#         captions_list = list(all_image_tokens_df[all_image_tokens_df["img_id"] == img_id].img_caption)
#         captions_dict[img_id] = captions_list
#     return captions_dict

# train_captions = create_captions_dictionary(train_image_names, image_tokens)
# test_captions = create_captions_dictionary(test_image_names, image_tokens)
# validation_captions = create_captions_dictionary(val_image_names, image_tokens)


# # ### ResNet50 model for encoding images




# feature_extractor_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))
# feature_extractor_model.summary() 


# # ### Encoding images and forming dictionaries containing mapping of image_id to image encodings


# def encode_images(captions_dict, feature_extractor, image_dir_path, pkl_filename):
#     pickled_file_path = os.path.join(pickled_features_path, pkl_filename)
#     if os.path.exists(pickled_file_path):
#         print(f"Loading encoded features from {pickled_file_path}...")
#         with open(pickled_file_path, "rb") as pickle_f:
#             encoded_features = pickle.load(pickle_f)
#         return encoded_features

#     print(f"Encoding images and saving to {pickled_file_path}...")
#     encoded_features = {}
#     for image_name in tqdm(captions_dict.keys(), desc=f"Encoding {pkl_filename.split('_')[0]}"):
#         img_path = os.path.join(image_dir_path, image_name)
#         try:
#             img = image.load_img(img_path, target_size=(224, 224))
#             x = image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = preprocess_input(x)
#             features = feature_extractor.predict(x, verbose=0)
#             encoded_features[image_name] = features.squeeze()
#         except Exception as e:
#             print(f"\nWarning: Could not process image {img_path}. Error: {e}")
#             continue # Skip this image

#     with open(pickled_file_path, "wb") as pickle_f:
#         pickle.dump(encoded_features, pickle_f)
#     return encoded_features

# train_features = encode_images(train_captions, feature_extractor_model, images_base_path, "train_encoded_images.p")
# test_features = encode_images(test_captions, feature_extractor_model, images_base_path, "test_encoded_images.p")
# validation_features = encode_images(validation_captions, feature_extractor_model, images_base_path, "validation_encoded_images.p")


# # ### Setting hyper parameters for vocabulary size and maximum length



# all_train_captions_list=[]
# for img_id in tqdm(train_captions, desc="Gathering all train captions"):
#   for caption_text in train_captions[img_id]:
#     all_train_captions_list.append(caption_text)

# print(f"Total number of training captions: {len(all_train_captions_list)}")

# # Using Keras Tokenizer
# tokenizer = Tokenizer(oov_token="<unk>")  
# tokenizer.fit_on_texts(all_train_captions_list)

# # added section
# tokenizer_save_path = os.path.join(pickled_features_path, "tokenizer.pkl") # pickled_features_path should be defined
# with open(tokenizer_save_path, 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print(f"Tokenizer saved to {tokenizer_save_path}")

# vocab_size = len(tokenizer.word_index) + 1 # +1 for padding token 0
# print(f"Vocabulary Size: {vocab_size}")

# # Create word_to_indices and indices_to_words
# words_to_indices = tokenizer.word_index
# indices_to_words = {idx: word for word, idx in words_to_indices.items()}



# # Calculate max_length from all train captions
# all_sequences = tokenizer.texts_to_sequences(all_train_captions_list)
# max_length = max(len(s) for s in all_sequences)
# print(f"Maximum caption length: {max_length}")



# model_params_save_path = os.path.join(pickled_features_path, "model_params.pkl") # pickled_features_path should be defined
# params_to_save = {
#     'max_length': max_length,
#     'vocab_size': vocab_size,
#     'words_to_indices': words_to_indices, # Good to save for potential future use
#     'indices_to_words': indices_to_words
# }
# try:
#     with open(model_params_save_path, 'wb') as f_params:
#         pickle.dump(params_to_save, f_params, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"Model parameters (max_length, vocab_size, etc.) saved to {model_params_save_path}")
# except Exception as e:
#     print(f"Error saving model_params.pkl: {e}")

# # ### Transforming data into dictonary mapping of image_id to encoded captions



# #forming dictionary having encoded captions
# train_encoded_captions={}
# for img_id in tqdm(train_captions, desc="Encoding train captions"):
#   captions_for_img = train_captions[img_id]
#   sequences = tokenizer.texts_to_sequences(captions_for_img)
#   padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
#   train_encoded_captions[img_id] = padded_sequences

# if train_encoded_captions:
#     sample_id = list(train_encoded_captions.keys())[0]
#     print(f"Sample encoded captions for {sample_id}:\n", train_encoded_captions[sample_id])

#     # Example of decoding a caption
#     print("\nDecoding sample caption:")
#     sample_encoded_caption = train_encoded_captions[sample_id][0]
#     decoded_caption = " ".join([indices_to_words.get(idx, "?") for idx in sample_encoded_caption if idx != 0])
#     print(decoded_caption)


# # ## Data Generator for Modelling



# def data_generator(encoded_captions_dict, image_features_dict, tokenizer_obj, max_len, num_photos_per_batch, vocab_sz):
#     X1_list, X2_list, Y_list = list(), list(), list() 
#     n = 0
#     image_ids = list(encoded_captions_dict.keys())
    
#     while True:
#         np.random.shuffle(image_ids)
#         for img_id in image_ids:
#             if img_id not in image_features_dict:
#                 continue
            
#             n += 1
#             captions = encoded_captions_dict[img_id]
#             img_feature = image_features_dict[img_id]
            
#             for caption_seq in captions:
#                 for i in range(1, len(caption_seq)):
#                     in_seq, out_seq = caption_seq[:i], caption_seq[i]
#                     if out_seq == 0:
#                         break
#                     in_seq_padded = pad_sequences([in_seq], maxlen=max_len, padding='post')[0] 
#                     out_word = to_categorical([out_seq], num_classes=vocab_sz)[0]
                    
#                     X1_list.append(img_feature)
#                     X2_list.append(in_seq_padded) # Use the padded sequence
#                     Y_list.append(out_word)
            
#             if n == num_photos_per_batch:
                
#                 yield ( (array(X1_list), array(X2_list)), array(Y_list) ) 
#                 X1_list, X2_list, Y_list = list(), list(), list()
#                 n = 0
  

# embedding_dim = 256
# lstm_units = 256
# dropout_rate = 0.3

# # Define the model

# input_img_features = Input(shape=(feature_extractor_model.output_shape[1],), name="image_features_input") # (2048,)
# fe1 = Dropout(dropout_rate)(input_img_features)
# fe2 = Dense(embedding_dim, activation='relu')(fe1)

# input_captions = Input(shape=(max_length,), name="caption_sequence_input")
# se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(input_captions)
# se2 = Dropout(dropout_rate)(se1)
# se3 = LSTM(lstm_units)(se2)

# decoder1 = add([fe2, se3])
# decoder2 = Dense(embedding_dim, activation='relu')(decoder1)
# outputs = Dense(vocab_size, activation='softmax', name="caption_output")(decoder2) # This defines 'outputs'

# captioning_model = Model(inputs=[input_img_features, input_captions], outputs=outputs)
# captioning_model.summary()
# captioning_model.compile( # <<< ADDED PARENTHESES HERE
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# # --- Code to load latest checkpoint ---
# latest_checkpoint = None
# start_epoch = 0
# if os.path.exists(model_checkpoint_path):
#     checkpoint_files = [f for f in os.listdir(model_checkpoint_path) if f.endswith(".weights.h5") and f.startswith("caption_model_epoch_")]
#     if checkpoint_files:
#         # Sort by epoch number (assuming format "caption_model_epoch_EE_...")
#         checkpoint_files.sort(key=lambda x: int(x.split('_')[3])) # epoch_EE
#         latest_checkpoint = os.path.join(model_checkpoint_path, checkpoint_files[-1])
#         try:
#             # Extract epoch number from filename
#             start_epoch = int(latest_checkpoint.split('_epoch_')[1].split('_')[0])
#             print(f"Found latest checkpoint: {latest_checkpoint}")
#             print(f"Loading weights and resuming from epoch {start_epoch + 1}") # Training will start at the *next* epoch
#             captioning_model.load_weights(latest_checkpoint)
#         except Exception as e:
#             print(f"Error loading weights from {latest_checkpoint}: {e}. Starting from scratch.")
#             latest_checkpoint = None
#             start_epoch = 0


# # ## Training
# # Training parameters
# total_epochs_to_run = 6 # Define the total number of epochs you want to run in the end
# photos_per_batch = 32
# steps_per_epoch = len(train_encoded_captions) // photos_per_batch
# if len(train_encoded_captions) % photos_per_batch != 0:
#     steps_per_epoch += 1

# train_gen = data_generator(train_encoded_captions, train_features, tokenizer, max_length, photos_per_batch, vocab_size)

# checkpoint_filepath_template = os.path.join(model_checkpoint_path, "caption_model_epoch_{epoch:02d}_loss_{loss:.2f}.weights.h5")
# model_checkpoint_callback = ModelCheckpoint(
#     filepath=checkpoint_filepath_template,
#     save_weights_only=True,
#     monitor='loss',
#     mode='min',
#     save_best_only=False,
#     save_freq='epoch'
# )

# if start_epoch < total_epochs_to_run:
#     print(f"\n--- Starting/Resuming Training ---")
#     print(f"Targeting {total_epochs_to_run} total epochs.")
#     print(f"Starting from epoch {start_epoch}.") # Keras initial_epoch is 0-indexed for the first epoch to run
    
#     captioning_model.fit(train_gen,
#                          epochs=total_epochs_to_run,        # Total epochs to reach
#                          initial_epoch=start_epoch,        # Epoch to start training from
#                          steps_per_epoch=steps_per_epoch,
#                          verbose=1,
#                          callbacks=[model_checkpoint_callback])
    
#     print("\n--- Training Finished ---")
#     captioning_model.save(os.path.join(model_checkpoint_path, "final_caption_model.keras"))
#     captioning_model.save_weights(os.path.join(model_checkpoint_path, "final_caption_model_weights.weights.h5"))
#     print(f"Final model and weights saved in '{model_checkpoint_path}'")
# else:
#     print(f"Training already completed up to {start_epoch} epochs (target was {total_epochs_to_run}). Loading final model for evaluation.")
#     # Optionally load the very final saved model if it exists and matches total_epochs_to_run
#     final_model_path = os.path.join(model_checkpoint_path, "final_caption_model.keras")
#     if os.path.exists(final_model_path) and start_epoch >= total_epochs_to_run:
#         print(f"Loading fully trained model from {final_model_path}")
#         try:
#             #Loads the saved model from disk for testing or predictions.


#             captioning_model = tensorflow.keras.models.load_model(final_model_path) # 
#         except Exception as e:
#             print(f"Could not load final model {final_model_path}: {e}. Using weights from last epoch checkpoint.")
#             # Fallback to loading weights if full model load fails, or if it's preferred.
#             if latest_checkpoint and os.path.exists(latest_checkpoint):
#                  captioning_model.load_weights(latest_checkpoint)
#             else:
#                 print("No suitable model or weights found to load for evaluation.")

# # ... (rest of the script for prediction and evaluation) ...

# def greedy_search_predict(photo_features_encoded, trained_model, tokenizer_obj, max_len_caption, idx_to_word_map):
#     photo_features_encoded = photo_features_encoded.reshape(1, -1) # Reshape for model input
#     in_text = '<start>'
#     for _ in range(max_len_caption):
#         sequence = tokenizer_obj.texts_to_sequences([in_text])[0]
#         sequence_padded = pad_sequences([sequence], maxlen=max_len_caption, padding='post')
        
#         y_pred_probs = trained_model.predict([photo_features_encoded, sequence_padded], verbose=0)
#         y_pred_idx = np.argmax(y_pred_probs[0])
        
#         word = idx_to_word_map.get(y_pred_idx)
        
#         if word is None or word == '<end>':
#             break
#         in_text += ' ' + word
        
#     final_caption_words = in_text.split()
#     # Remove <start> and <end> if present (already handled by break on <end>)
#     final_caption_words = [w for w in final_caption_words if w not in ('<start>', '<end>')]
#     return final_caption_words

# # ## Predicting Captions on Test Set using Greedy Search





# # ## Calculating Average Bleu Score on Test Set using Greedy Search





# # Define a flag file name for Greedy Search
# greedy_eval_flag_file = os.path.join(pickled_features_path, "greedy_eval_completed.flag")
# print(f"Path for Greedy Search flag file: {greedy_eval_flag_file}") # For debugging

# # ## Calculating Average Bleu Score on Test Set using Greedy Search
# print(f"Checking existence of Greedy Search flag file: {os.path.exists(greedy_eval_flag_file)}") # For debugging

# if not os.path.exists(greedy_eval_flag_file):
#     print("\n--- Calculating Average BLEU Score (Greedy Search) ---")
#     total_bleu_score = 0
#     num_test_images = 0
    
#     # Ensure captioning_model, tokenizer, max_length, indices_to_words are loaded and available globally
#     if 'captioning_model' not in globals() or captioning_model is None:
#         print("ERROR: captioning_model is not loaded. Cannot perform evaluation.")
#     elif 'tokenizer' not in globals() or tokenizer is None:
#         print("ERROR: tokenizer is not loaded. Cannot perform evaluation.")
#     # Add similar checks for max_length and indices_to_words if necessary
#     else:
#         for img_id in tqdm(test_features, desc="Greedy Search Evaluation"):
#             photo_feature = test_features.get(img_id) # Use .get for safety
#             if photo_feature is None:
#                 print(f"Warning: Features for image_id {img_id} not found in test_features.")
#                 continue

#             reference_captions_list = []
#             if img_id in test_captions:
#                 for cap_text in test_captions[img_id]:
#                     ref_words = [word for word in cap_text.split(" ") if word not in ('<start>', '<end>')]
#                     reference_captions_list.append(ref_words)
            
#             if not reference_captions_list:
#                 print(f"Warning: No reference captions for image_id {img_id}.")
#                 continue

#             predicted_caption_words = greedy_search_predict(photo_feature, captioning_model, tokenizer, max_length, indices_to_words)
            
#             try:
#                 score = sentence_bleu(reference_captions_list, predicted_caption_words, weights=(0.25, 0.25, 0.25, 0.25))
#                 total_bleu_score += score
#                 num_test_images += 1
#             except ZeroDivisionError:
#                 print(f"Warning: Could not calculate BLEU for {img_id} (Greedy Search) due to short prediction.")
#             except Exception as e_bleu:
#                 print(f"Error calculating BLEU for {img_id} (Greedy Search): {e_bleu}")


#         if num_test_images > 0:
#             avg_bleu_score = total_bleu_score / num_test_images
#             print(f"\nAverage BLEU-4 Score on Test Set (Greedy Search): {avg_bleu_score:.4f}")
#             try:
#                 with open(greedy_eval_flag_file, 'w') as f:
#                     f.write(f"Completed on {pd.Timestamp.now()}\n")
#                     f.write(f"Average BLEU: {avg_bleu_score:.4f}\n")
#                 print(f"Greedy search evaluation completion flagged in {greedy_eval_flag_file}")
#             except Exception as e_flag:
#                 print(f"Error writing flag file {greedy_eval_flag_file}: {e_flag}")
#         else:
#             print("\nNo test images were successfully evaluated for Greedy Search. Flag file not created.")
# else:
#     print("\n--- Greedy Search Evaluation previously completed. Skipping. ---")
#     try:
#         with open(greedy_eval_flag_file, 'r') as f:
#             print("Previous results from Greedy Search flag file:")
#             for line in f:
#                 print(line.strip())
#     except Exception as e:
#         print(f"Could not read from Greedy Search flag file {greedy_eval_flag_file}: {e}")




# # ## Beam Search Function



# def beam_search_predict(photo_features_encoded, trained_model, tokenizer_obj, max_len_caption, idx_to_word_map, beam_width_k):
#     photo_features_encoded = photo_features_encoded.reshape(1, -1)
#     start_token_text = '<start>'
    
#     # Initial step: Expand from <start>
#     start_sequence_indices = tokenizer_obj.texts_to_sequences([start_token_text])[0]
#     start_sequence_padded = pad_sequences([start_sequence_indices], maxlen=max_len_caption, padding='post')
    
#     y_pred_probs = trained_model.predict([photo_features_encoded, start_sequence_padded], verbose=0)[0]
#     top_k_indices = np.argsort(y_pred_probs)[-beam_width_k:] # Get indices of top k predictions
    
#     # current_beams stores tuples of (text_sequence, probability_score)
#     current_beams = []
#     for word_idx in top_k_indices:
#         word = idx_to_word_map.get(word_idx)
#         if word is None: continue
#         current_beams.append( (start_token_text + ' ' + word, np.log(y_pred_probs[word_idx] + 1e-9)) ) # Use log probs for stability

#     completed_beams = []

#     for _ in range(max_len_caption - 1): # Max number of steps
#         if not current_beams: break # Stop if no active beams

#         candidate_beams = []
#         for text_seq, current_log_prob in current_beams:
#             if text_seq.endswith('<end>'):
#                 completed_beams.append((text_seq, current_log_prob))
#                 continue # This beam is finished

#             # Prepare input for the model
#             current_indices = tokenizer_obj.texts_to_sequences([text_seq])[0]
#             current_indices_padded = pad_sequences([current_indices], maxlen=max_len_caption, padding='post')
            
#             # Predict next word probabilities
#             next_word_probs = trained_model.predict([photo_features_encoded, current_indices_padded], verbose=0)[0]
            
#             # Expand this beam with top k next words
#             top_k_next_indices = np.argsort(next_word_probs)[-beam_width_k:]
            
#             for next_word_idx in top_k_next_indices:
#                 next_word = idx_to_word_map.get(next_word_idx)
#                 if next_word is None: continue
                
#                 new_text_seq = text_seq + ' ' + next_word
#                 new_log_prob = current_log_prob + np.log(next_word_probs[next_word_idx] + 1e-9)
#                 candidate_beams.append((new_text_seq, new_log_prob))
        
       
#         all_possible_beams = candidate_beams + completed_beams 
        
      
#         completed_beams = []
#         active_beams_for_next_step = []

#         sorted_beams = sorted(all_possible_beams, key=lambda x: x[1], reverse=True)
        
#         for beam_text, beam_score in sorted_beams:
#             if beam_text.endswith('<end>'):
#                 if len(completed_beams) < beam_width_k : # Keep top k completed beams
#                      completed_beams.append((beam_text, beam_score))
#             else:
#                 if len(active_beams_for_next_step) < beam_width_k:
#                     active_beams_for_next_step.append((beam_text, beam_score))
        
#         current_beams = active_beams_for_next_step
        
#         # Optimization: if all top k current beams are in completed_beams, we can stop early.
#         if len(completed_beams) >= beam_width_k and (not current_beams or completed_beams[0][1] > current_beams[0][1]):
#              break


#     # If no beams completed, use the best active beam
#     if not completed_beams and current_beams:
#         completed_beams.extend(current_beams)
#     elif not completed_beams and not current_beams: # Edge case: no beams at all
#         return ["<error>", "generation", "failed"]


#     # Sort completed beams by score and pick the best one
#     best_beam = sorted(completed_beams, key=lambda x: x[1], reverse=True)[0]
    
#     final_caption_words = best_beam[0].split()
#     final_caption_words = [w for w in final_caption_words if w not in ('<start>', '<end>')]
#     return final_caption_words


# # ## Predicting Captions on Test Set using Beam Search with k=3



# print("\n--- Predicting on Test Set (Beam Search, k=3) ---")
# beam_k = 3
# num_predictions_to_show = 5
# count = 0
# for img_id in list(test_features.keys())[:num_predictions_to_show * 2]: # Iterate more for displayable ones
#     if count >= num_predictions_to_show:
#         break

#     img_path_to_show = os.path.join(images_base_path, img_id)
#     img_display = cv2.imread(img_path_to_show)
    
#     if img_display is None:
#         print(f"Warning: Could not load image {img_id} for display.")
#         continue
        
#     plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
#     # plt.show() # This will block. Save instead.
#     plt.savefig(f"predicted_image_{img_id.replace('.jpg','')}_beam{beam_k}.png")
#     plt.close()
    
#     photo_feature = test_features[img_id]
    
#     reference_captions_list = []
#     if img_id in test_captions:
#         for cap_text in test_captions[img_id]:
#             ref_words = [word for word in cap_text.split(" ") if word not in ('<start>', '<end>')]
#             reference_captions_list.append(ref_words)
            
#     predicted_caption_words = beam_search_predict(photo_feature, captioning_model, tokenizer, max_length, indices_to_words, beam_k)
    
#     bleu_score = 0
#     if reference_captions_list:
#         bleu_score = sentence_bleu(reference_captions_list, predicted_caption_words, weights=(0.25, 0.25, 0.25, 0.25)) # BLEU-4

#     print(f"\nImage: {img_id}")
#     print("Reference Captions:")
#     for ref_cap in reference_captions_list:
#         print(" ".join(ref_cap))
#     print(f"Predicted Caption (Beam Search k={beam_k}):")
#     print(" ".join(predicted_caption_words))
#     print(f"BLEU Score: {bleu_score:.4f}")
#     count += 1


# # ## Calculating Average Bleu Score on Test Set using Beam Search with k=3




# beam_eval_flag_file_template = os.path.join(pickled_features_path, "beam_eval_k{k}_completed.flag")
# beam_k = 3 # Make sure beam_k is defined before using it in the filename
# beam_eval_flag_file = beam_eval_flag_file_template.format(k=beam_k)


# # ## Calculating Average Bleu Score on Test Set using Beam Search with k=3



# if not os.path.exists(beam_eval_flag_file):
#     print(f"\n--- Calculating Average BLEU Score (Beam Search, k={beam_k}) ---")
#     total_bleu_score_beam = 0
#     num_test_images_beam = 0

    
#     for img_id in tqdm(test_features, desc=f"Beam Search (k={beam_k}) Evaluation"):
        

#         photo_feature = test_features[img_id]
        
#         reference_captions_list = []
#         if img_id in test_captions:
#             for cap_text in test_captions[img_id]:
#                 ref_words = [word for word in cap_text.split(" ") if word not in ('<start>', '<end>')]
#                 reference_captions_list.append(ref_words)
        
#         if not reference_captions_list:
#             continue

#         predicted_caption_words = beam_search_predict(photo_feature, captioning_model, tokenizer, max_length, indices_to_words, beam_k)
        
#         try:
#             score = sentence_bleu(reference_captions_list, predicted_caption_words, weights=(0.25, 0.25, 0.25, 0.25))
#             total_bleu_score_beam += score
#             num_test_images_beam += 1
#             # individual_beam_results[img_id] = score # Store for granular resume
#         except ZeroDivisionError:
#             print(f"Warning: Could not calculate BLEU for {img_id} (Beam Search k={beam_k}) due to short prediction.")
#             # individual_beam_results[img_id] = 0 # Or some other placeholder


#     if num_test_images_beam > 0:
#         avg_bleu_score_beam = total_bleu_score_beam / num_test_images_beam
#         print(f"\nAverage BLEU-4 Score on Test Set (Beam Search k={beam_k}): {avg_bleu_score_beam:.4f}")
#         # Create the flag file
#         with open(beam_eval_flag_file, 'w') as f:
#             f.write(f"Completed on {pd.Timestamp.now()} for k={beam_k}\n")
#             f.write(f"Average BLEU: {avg_bleu_score_beam:.4f}\n")
#         print(f"Beam search (k={beam_k}) evaluation completion flagged in {beam_eval_flag_file}")
#     else:
#         print(f"\nNo test images were evaluated for Beam Search (k={beam_k}). Flag file not created.")
# else:
#     print(f"\n--- Beam Search (k={beam_k}) Evaluation previously completed. Skipping. ---")
#     # Optionally, read and print the stored average BLEU score
#     try:
#         with open(beam_eval_flag_file, 'r') as f:
#             print(f"Previous results for Beam Search (k={beam_k}) from flag file:")
#             for line in f:
#                 print(line.strip())
#     except Exception as e:
#         print(f"Could not read from Beam Search (k={beam_k}) flag file: {e}")


#!/usr/bin/env python
# coding: utf-8

# # Image Captioning using LSTM on Flickr8K dataset

# ## Downloading Data and Dependencies

#importing the libraries
# Run this in a Python shell or script
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from numpy import array
import pickle
import numpy as np
import os
import cv2
import datetime # For timestamp without pandas if preferred

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image # Renamed to avoid conflict with cv2.image
from tensorflow.keras.callbacks import ModelCheckpoint

# ## Data Visualization and Preprocessing

# Define paths
captions_base_path = "all_captions/"
images_base_path = "all_images/Flicker8k_Dataset/" # Corrected Flickr spelling
pickled_features_path = "pickled_features/"
model_checkpoint_path = "model_checkpoints/"

# Create directories if they don't exist
os.makedirs(pickled_features_path, exist_ok=True)
os.makedirs(model_checkpoint_path, exist_ok=True)


#converting the text files to pandas dataframe
try:
    image_tokens=pd.read_csv(os.path.join(captions_base_path, "Flickr8k.lemma.token.txt"),sep='\t',names=["img_id","img_caption"])
    train_image_names=pd.read_csv(os.path.join(captions_base_path, "Flickr_8k.trainImages.txt"),names=["img_id"])
    test_image_names=pd.read_csv(os.path.join(captions_base_path, "Flickr_8k.testImages.txt"),names=["img_id"])
    val_image_names=pd.read_csv(os.path.join(captions_base_path, "Flickr_8k.devImages.txt"),names=["img_id"])
except FileNotFoundError as e:
    print(f"Error: Caption file not found. Make sure '{captions_base_path}' directory is correctly populated.")
    print(e)
    exit()

#just checking the number of images in the dataset
try:
    l=os.listdir(images_base_path)
    print(f"Number of images found: {len(l)}")
except FileNotFoundError:
    print(f"Error: Image directory '{images_base_path}' not found. Please check the path.")
    exit()



#visualising one of the train images
if not train_image_names.empty:
    try:
        img_path_to_show = os.path.join(images_base_path, train_image_names.img_id[0])
        img_show = cv2.imread(img_path_to_show)
        if img_show is not None:
            plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
            plt.savefig("sample_train_image.png")
            print("Saved sample_train_image.png")
            plt.close()
        else:
            print(f"Warning: Could not read sample image at {img_path_to_show}")
    except Exception as e:
        print(f"Error displaying sample image: {e}")
else:
    print("No training images to display.")


#removing the #0,#1,#2,#3,#5 from the image ids
image_tokens["img_id"]=image_tokens["img_id"].map(lambda x: x.split('#')[0])


image_tokens["img_caption"]=image_tokens["img_caption"].map(lambda x: "<start> " + x.strip() + " <end>")


print("Image Tokens Head:\n", image_tokens.head())
print("Train Image Names Head:\n", train_image_names.head())


# ### Creating dictionaries to map image_id and their corresponding captions
def create_captions_dictionary(image_names_df, all_image_tokens_df):
    captions_dict = {}
    for i in tqdm(range(len(image_names_df)), desc=f"Creating Captions Dict for {image_names_df.iloc[0,0][:10]}..."): # Show part of first img_id for context
        img_id = image_names_df["img_id"].iloc[i]
        captions_list = list(all_image_tokens_df[all_image_tokens_df["img_id"] == img_id].img_caption)
        captions_dict[img_id] = captions_list
    return captions_dict

train_captions = create_captions_dictionary(train_image_names, image_tokens)
test_captions = create_captions_dictionary(test_image_names, image_tokens)
validation_captions = create_captions_dictionary(val_image_names, image_tokens)


# ### ResNet50 model for encoding images
feature_extractor_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))
# feature_extractor_model.summary() # Uncomment if you want to see it


# ### Encoding images
def encode_images(captions_dict, feature_extractor, image_dir_path, pkl_filename):
    pickled_file_path = os.path.join(pickled_features_path, pkl_filename)
    if os.path.exists(pickled_file_path):
        print(f"Loading encoded features from {pickled_file_path}...")
        with open(pickled_file_path, "rb") as pickle_f:
            encoded_features = pickle.load(pickle_f)
        return encoded_features

    print(f"Encoding images and saving to {pickled_file_path}...")
    encoded_features = {}
    for image_name in tqdm(captions_dict.keys(), desc=f"Encoding {pkl_filename.split('_encoded_images.p')[0]}"):
        img_path = os.path.join(image_dir_path, image_name)
        try:
            img = keras_image.load_img(img_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = feature_extractor.predict(x, verbose=0)
            encoded_features[image_name] = features.squeeze()
        except Exception as e:
            print(f"\nWarning: Could not process image {img_path}. Error: {e}")
            continue

    with open(pickled_file_path, "wb") as pickle_f:
        pickle.dump(encoded_features, pickle_f)
    return encoded_features

train_features = encode_images(train_captions, feature_extractor_model, images_base_path, "train_encoded_images.p")
test_features = encode_images(test_captions, feature_extractor_model, images_base_path, "test_encoded_images.p")
validation_features = encode_images(validation_captions, feature_extractor_model, images_base_path, "validation_encoded_images.p")


# ### Setting hyper parameters for vocabulary size and maximum length
all_train_captions_list=[]
for img_id in tqdm(train_captions, desc="Gathering all train captions"):
  for caption_text in train_captions[img_id]:
    all_train_captions_list.append(caption_text)

print(f"Total number of training captions: {len(all_train_captions_list)}")

tokenizer = Tokenizer(oov_token="<unk>") 
tokenizer.fit_on_texts(all_train_captions_list)

tokenizer_save_path = os.path.join(pickled_features_path, "tokenizer.pkl")
with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved to {tokenizer_save_path}")

vocab_size = len(tokenizer.word_index) + 1 
print(f"Vocabulary Size: {vocab_size}")

words_to_indices = tokenizer.word_index
indices_to_words = {idx: word for word, idx in words_to_indices.items()}
# Keras Tokenizer typically reserves 0 for padding, so we don't need to explicitly add indices_to_words[0] = "<pad>"
# unless your specific <pad> token has a different index or string representation.
# For this script, if 0 is used for padding, it will be skipped during decoding.

all_sequences = tokenizer.texts_to_sequences(all_train_captions_list)
max_length = max(len(s) for s in all_sequences)
print(f"Maximum caption length: {max_length}")

# Determine the string representation for the end token
# This is based on your Flask app logs where index 3 mapped to "end"
# If "<end>" is in vocab, what index does it get? What does that index map back to?
END_TOKEN_STRING = "end" # Default based on your observation; VERIFY THIS
if "<end>" in words_to_indices:
    end_idx_check = words_to_indices["<end>"]
    actual_end_str_from_idx = indices_to_words.get(end_idx_check)
    print(f"DEBUG: Token '<end>' maps to index {end_idx_check}, which maps back to string '{actual_end_str_from_idx}'")
    if actual_end_str_from_idx:
        END_TOKEN_STRING = actual_end_str_from_idx # Use the actual string from tokenizer
else:
    print(f"DEBUG: Token '<end>' not found directly in vocabulary. Will use '{END_TOKEN_STRING}' as end token string.")
    # Verify if "end" (without <>) is your end token
    if "end" in words_to_indices:
         print(f"DEBUG: Token 'end' (no brackets) is in vocab at index {words_to_indices['end']}")
    else:
        print(f"DEBUG: Neither '<end>' nor 'end' found. Critical to identify actual end token string for {indices_to_words.get(3)} (assuming 3 is end index based on prior logs)")
        # If index 3 is 'end' as per logs:
        if indices_to_words.get(3) == "end":
            END_TOKEN_STRING = "end"
        else:
            print("CRITICAL WARNING: Could not auto-determine END_TOKEN_STRING. Defaulting to 'end'. Check your vocabulary!")


model_params_save_path = os.path.join(pickled_features_path, "model_params.pkl")
params_to_save = {
    'max_length': max_length,
    'vocab_size': vocab_size,
    'words_to_indices': words_to_indices,
    'indices_to_words': indices_to_words,
    'end_token_string': END_TOKEN_STRING # Save the determined end token string
}
try:
    with open(model_params_save_path, 'wb') as f_params:
        pickle.dump(params_to_save, f_params, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model parameters saved to {model_params_save_path} (including end_token_string: '{END_TOKEN_STRING}')")
except Exception as e:
    print(f"Error saving model_params.pkl: {e}")

# ### Transforming data
train_encoded_captions={}
for img_id in tqdm(train_captions, desc="Encoding train captions"):
  captions_for_img = train_captions[img_id]
  sequences = tokenizer.texts_to_sequences(captions_for_img)
  padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
  train_encoded_captions[img_id] = padded_sequences

if train_encoded_captions:
    sample_id = list(train_encoded_captions.keys())[0]
    print(f"Sample encoded captions for {sample_id}:\n", train_encoded_captions[sample_id][0])
    decoded_caption = " ".join([indices_to_words.get(idx, "?") for idx in train_encoded_captions[sample_id][0] if idx != 0])
    print(f"\nDecoded sample caption: {decoded_caption}")


# ## Data Generator
def data_generator(encoded_captions_dict, image_features_dict, max_len, num_photos_per_batch, vocab_sz): # Removed tokenizer_obj as it's not used
    X1_list, X2_list, Y_list = list(), list(), list() 
    n = 0
    image_ids = list(encoded_captions_dict.keys())
    
    while True:
        np.random.shuffle(image_ids)
        for img_id in image_ids:
            if img_id not in image_features_dict: continue
            n += 1
            captions = encoded_captions_dict[img_id]
            img_feature = image_features_dict[img_id]
            for caption_seq in captions:
                for i in range(1, len(caption_seq)):
                    in_seq, out_seq = caption_seq[:i], caption_seq[i]
                    if out_seq == 0: break
                    in_seq_padded = pad_sequences([in_seq], maxlen=max_len, padding='post')[0] 
                    out_word = to_categorical([out_seq], num_classes=vocab_sz)[0]
                    X1_list.append(img_feature)
                    X2_list.append(in_seq_padded)
                    Y_list.append(out_word)
            if n == num_photos_per_batch:
                yield ( (array(X1_list), array(X2_list)), array(Y_list) )
                X1_list, X2_list, Y_list = list(), list(), list()
                n = 0

# ## Model Definition
embedding_dim = 256
lstm_units = 256
dropout_rate = 0.3

input_img_features = Input(shape=(feature_extractor_model.output_shape[1],), name="image_features_input")
fe1 = Dropout(dropout_rate)(input_img_features)
fe2 = Dense(embedding_dim, activation='relu')(fe1)

input_captions = Input(shape=(max_length,), name="caption_sequence_input")
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(input_captions)
se2 = Dropout(dropout_rate)(se1)
se3 = LSTM(lstm_units)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(embedding_dim, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax', name="caption_output")(decoder2)

captioning_model = Model(inputs=[input_img_features, input_captions], outputs=outputs)
captioning_model.summary()
captioning_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ## Training
latest_checkpoint = None
start_epoch = 0
if os.path.exists(model_checkpoint_path):
    checkpoint_files = [f for f in os.listdir(model_checkpoint_path) if f.endswith(".weights.h5") and f.startswith("caption_model_epoch_")]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[3]))
        latest_checkpoint = os.path.join(model_checkpoint_path, checkpoint_files[-1])
        try:
            start_epoch = int(latest_checkpoint.split('_epoch_')[1].split('_')[0])
            print(f"Found latest checkpoint: {latest_checkpoint}")
            print(f"Loading weights and resuming from epoch {start_epoch + 1}")
            captioning_model.load_weights(latest_checkpoint)
        except Exception as e:
            print(f"Error loading weights from {latest_checkpoint}: {e}. Starting from scratch.")
            latest_checkpoint = None
            start_epoch = 0

total_epochs_to_run = 6
photos_per_batch = 32
steps_per_epoch = len(train_encoded_captions) // photos_per_batch
if len(train_encoded_captions) % photos_per_batch != 0: steps_per_epoch += 1

train_gen = data_generator(train_encoded_captions, train_features, max_length, photos_per_batch, vocab_size)

checkpoint_filepath_template = os.path.join(model_checkpoint_path, "caption_model_epoch_{epoch:02d}_loss_{loss:.2f}.weights.h5")
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath_template, save_weights_only=True, monitor='loss', mode='min', save_best_only=False, save_freq='epoch'
)

if start_epoch < total_epochs_to_run:
    print(f"\n--- Starting/Resuming Training ---")
    print(f"Targeting {total_epochs_to_run} total epochs. Starting from epoch {start_epoch}.")
    captioning_model.fit(train_gen, epochs=total_epochs_to_run, initial_epoch=start_epoch, steps_per_epoch=steps_per_epoch, verbose=1, callbacks=[model_checkpoint_callback])
    print("\n--- Training Finished ---")
    captioning_model.save(os.path.join(model_checkpoint_path, "final_caption_model.keras"))
    captioning_model.save_weights(os.path.join(model_checkpoint_path, "final_caption_model_weights.weights.h5"))
    print(f"Final model and weights saved in '{model_checkpoint_path}'")
else:
    print(f"Training already completed up to {start_epoch} epochs (target was {total_epochs_to_run}). Loading final model for evaluation.")
    final_model_path = os.path.join(model_checkpoint_path, "final_caption_model.keras")
    if os.path.exists(final_model_path) and start_epoch >= total_epochs_to_run:
        print(f"Loading fully trained model from {final_model_path}")
        try:
            captioning_model = tensorflow.keras.models.load_model(final_model_path)
        except Exception as e:
            print(f"Could not load final model {final_model_path}: {e}. Using weights from last epoch checkpoint if available.")
            if latest_checkpoint and os.path.exists(latest_checkpoint): captioning_model.load_weights(latest_checkpoint)
            else: print("No suitable model or weights found to load for evaluation.")

# --- Caption Generation Functions (with corrected end token handling) ---
def greedy_search_predict(photo_features_encoded, current_model, current_tokenizer, max_len, idx_to_word_map, end_token_str):
    photo_features_encoded = photo_features_encoded.reshape(1, -1)
    in_text = '<start>'
    generated_words = []
    print(f"--- Greedy Search (end token: '{end_token_str}') ---")
    for i_step in range(max_len):
        sequence = current_tokenizer.texts_to_sequences([in_text])[0]
        if not sequence:
            print(f"Greedy Warn: Tokenizer failed for '{in_text}' at step {i_step}")
            break
        sequence_padded = pad_sequences([sequence], maxlen=max_len, padding='post')
        y_pred_probs = current_model.predict([photo_features_encoded, sequence_padded], verbose=0)
        y_pred_idx = np.argmax(y_pred_probs[0])
        word = idx_to_word_map.get(y_pred_idx)
        
        print(f"Greedy Step {i_step+1}: Predicted '{word}' (idx {y_pred_idx})")

        if word is None or word == '<pad>':
            print(f"Greedy: Stopping due to '{word}'")
            break
        if word == end_token_str:
            print(f"Greedy: Stopping due to end token '{word}'")
            break
        generated_words.append(word)
        in_text += ' ' + word
    return " ".join(generated_words)

def beam_search_predict(photo_features_encoded, current_model, current_tokenizer, max_len, idx_to_word_map, beam_k, end_token_str):
    photo_features_encoded = photo_features_encoded.reshape(1, -1)
    start_token_text = '<start>'
    print(f"--- Beam Search k={beam_k} (end token: '{end_token_str}') ---")

    start_sequence_indices = current_tokenizer.texts_to_sequences([start_token_text])[0]
    if not start_sequence_indices:
        print(f"Beam Warn: Tokenizer failed for start token '{start_token_text}'")
        return "error tokenizer failed for start"
        
    start_sequence_padded = pad_sequences([start_sequence_indices], maxlen=max_len, padding='post')
    y_pred_probs = current_model.predict([photo_features_encoded, start_sequence_padded], verbose=0)[0]
    
    # Beam: list of (sequence_text, log_probability)
    current_beams = []
    top_k_indices = np.argsort(y_pred_probs)[-beam_k:]
    for word_idx in top_k_indices:
        word = idx_to_word_map.get(word_idx)
        if word is None or word == '<pad>': continue
        prob = y_pred_probs[word_idx]
        current_beams.append((start_token_text + ' ' + word, np.log(prob + 1e-9)))
    
    completed_beams = []

    for _ in range(max_len - 1): # -1 because we already generated the first word after <start>
        if not current_beams: break
        candidate_beams = []
        for text_seq, current_log_prob in current_beams:
            if text_seq.split()[-1] == end_token_str: # Check if last word was end token
                completed_beams.append((text_seq, current_log_prob))
                continue

            current_indices = current_tokenizer.texts_to_sequences([text_seq])[0]
            if not current_indices: continue
            current_indices_padded = pad_sequences([current_indices], maxlen=max_len, padding='post')
            next_word_probs = current_model.predict([photo_features_encoded, current_indices_padded], verbose=0)[0]
            
            # Expand this beam
            # Consider all possible next words (or a reasonable subset for efficiency if vocab is huge)
            # For simplicity, here we consider top_k from all, can be optimized
            
            # Get top beam_k candidates from this specific beam's expansion
            # To do this properly, we'd normally keep a heap of candidates from *all* current beams
            # and then select overall top k. This simplified version expands each beam by k.
            
            # Simplified: expand each current beam by k, then prune overall
            # More correct: for each current beam, for *every* word in vocab, calculate new score, then pick top k overall
            
            # Let's do a slightly better version: keep all candidates and then sort
            for next_word_idx in range(len(next_word_probs)): # Iterate all vocab
                next_word = idx_to_word_map.get(next_word_idx)
                if next_word is None or next_word == '<pad>': continue
                
                prob = next_word_probs[next_word_idx]
                if prob < 1e-6 and next_word != end_token_str: continue # Prune very unlikely words (unless it's end token)

                new_text_seq = text_seq + ' ' + next_word
                new_log_prob = current_log_prob + np.log(prob + 1e-9)
                candidate_beams.append((new_text_seq, new_log_prob))
        
        all_possible_beams = candidate_beams + completed_beams
        sorted_beams = sorted(all_possible_beams, key=lambda x: x[1], reverse=True)
        
        current_beams = []
        newly_completed_this_step = []
        
        for beam_text, beam_score in sorted_beams:
            if beam_text.split()[-1] == end_token_str:
                if len(newly_completed_this_step) < beam_k:
                    newly_completed_this_step.append((beam_text, beam_score))
            else:
                if len(current_beams) < beam_k:
                    current_beams.append((beam_text, beam_score))
            if len(current_beams) >= beam_k and len(newly_completed_this_step) >= beam_k:
                break # Got enough candidates for both

        completed_beams.extend(newly_completed_this_step)
        completed_beams = sorted(list(set(completed_beams)), key=lambda x: x[1], reverse=True)[:beam_k] # Keep best unique completed

        if len(completed_beams) >= beam_k and (not current_beams or (current_beams and completed_beams[-1][1] > current_beams[0][1])):
             break
        if not current_beams and completed_beams:
            break
            
    final_beams_to_consider = completed_beams if completed_beams else current_beams
    if not final_beams_to_consider: return "beam search failed"
    
    best_beam_text = sorted(final_beams_to_consider, key=lambda x: x[1], reverse=True)[0][0]
    
    final_caption_words = best_beam_text.split()
    clean_caption_words = [w for w in final_caption_words if w not in ('<start>', end_token_str)]
    return " ".join(clean_caption_words)

# ## Evaluation Sections (Using the refined generation functions)
chencherry = SmoothingFunction() # For BLEU score smoothing

# Greedy Search Evaluation
greedy_eval_flag_file = os.path.join(pickled_features_path, "greedy_eval_completed.flag")
print(f"\nPath for Greedy Search flag file: {greedy_eval_flag_file}")
print(f"Checking existence of Greedy Search flag file: {os.path.exists(greedy_eval_flag_file)}")

if not os.path.exists(greedy_eval_flag_file):
    print("\n--- Calculating Average BLEU Score (Greedy Search) ---")
    total_bleu_score_greedy = 0
    num_test_images_greedy = 0
    if 'captioning_model' not in globals() or captioning_model is None: print("ERROR: captioning_model not loaded for Greedy Eval.")
    else:
        for img_id in tqdm(test_features, desc="Greedy Search Evaluation"):
            photo_feature = test_features.get(img_id)
            if photo_feature is None: continue
            reference_captions_list = [[word for word in cap.split(" ") if word not in ('<start>', END_TOKEN_STRING)] for cap in test_captions.get(img_id, [])]
            if not reference_captions_list: continue
            
            predicted_caption_str = greedy_search_predict(photo_feature, captioning_model, tokenizer, max_length, indices_to_words, END_TOKEN_STRING)
            predicted_caption_words_list = predicted_caption_str.split()
            
            try:
                score = sentence_bleu(reference_captions_list, predicted_caption_words_list, smoothing_function=chencherry.method1)
                total_bleu_score_greedy += score
                num_test_images_greedy += 1
            except Exception as e_bleu: print(f"Error BLEU Greedy for {img_id}: {e_bleu}")

        if num_test_images_greedy > 0:
            avg_bleu_score_greedy = total_bleu_score_greedy / num_test_images_greedy
            print(f"\nAverage BLEU Score on Test Set (Greedy Search): {avg_bleu_score_greedy:.4f}")
            try:
                with open(greedy_eval_flag_file, 'w') as f:
                    f.write(f"Completed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nAverage BLEU (Greedy): {avg_bleu_score_greedy:.4f}\n")
                print(f"Greedy search evaluation completion flagged.")
            except Exception as e_flag: print(f"Error writing greedy flag file: {e_flag}")
        else: print("\nNo test images evaluated for Greedy Search. Flag file not created.")
else:
    print("\n--- Greedy Search Evaluation previously completed. Skipping. ---")
    try:
        with open(greedy_eval_flag_file, 'r') as f: print("Previous Greedy Results:\n" + f.read())
    except Exception as e: print(f"Could not read greedy flag file: {e}")

# Beam Search Evaluation
beam_k_eval = 3
beam_eval_flag_file = os.path.join(pickled_features_path, f"beam_eval_k{beam_k_eval}_completed.flag")
print(f"\nPath for Beam Search (k={beam_k_eval}) flag file: {beam_eval_flag_file}")
print(f"Checking existence of Beam Search flag file: {os.path.exists(beam_eval_flag_file)}")

if not os.path.exists(beam_eval_flag_file):
    print(f"\n--- Calculating Average BLEU Score (Beam Search, k={beam_k_eval}) ---")
    total_bleu_score_beam = 0
    num_test_images_beam = 0
    if 'captioning_model' not in globals() or captioning_model is None: print("ERROR: captioning_model not loaded for Beam Eval.")
    else:
        for img_id in tqdm(test_features, desc=f"Beam Search (k={beam_k_eval}) Evaluation"):
            photo_feature = test_features.get(img_id)
            if photo_feature is None: continue
            reference_captions_list = [[word for word in cap.split(" ") if word not in ('<start>', END_TOKEN_STRING)] for cap in test_captions.get(img_id, [])]
            if not reference_captions_list: continue
            
            predicted_caption_str = beam_search_predict(photo_feature, captioning_model, tokenizer, max_length, indices_to_words, beam_k_eval, END_TOKEN_STRING)
            predicted_caption_words_list = predicted_caption_str.split()

            try:
                score = sentence_bleu(reference_captions_list, predicted_caption_words_list, smoothing_function=chencherry.method1)
                total_bleu_score_beam += score
                num_test_images_beam += 1
            except Exception as e_bleu: print(f"Error BLEU Beam for {img_id}: {e_bleu}")

        if num_test_images_beam > 0:
            avg_bleu_score_beam = total_bleu_score_beam / num_test_images_beam
            print(f"\nAverage BLEU Score on Test Set (Beam Search k={beam_k_eval}): {avg_bleu_score_beam:.4f}")
            try:
                with open(beam_eval_flag_file, 'w') as f:
                    f.write(f"Completed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for k={beam_k_eval}\nAverage BLEU (Beam k={beam_k_eval}): {avg_bleu_score_beam:.4f}\n")
                print(f"Beam search (k={beam_k_eval}) evaluation completion flagged.")
            except Exception as e_flag: print(f"Error writing beam flag file: {e_flag}")
        else: print(f"\nNo test images evaluated for Beam Search (k={beam_k_eval}). Flag file not created.")
else:
    print(f"\n--- Beam Search (k={beam_k_eval}) Evaluation previously completed. Skipping. ---")
    try:
        with open(beam_eval_flag_file, 'r') as f: print(f"Previous Beam (k={beam_k_eval}) Results:\n" + f.read())
    except Exception as e: print(f"Could not read beam flag file: {e}")

# Predicting and Displaying a few sample captions (optional, good for visual check)
print("\n--- Generating Sample Predictions for Visual Check ---")
num_display_samples = 3 # How many samples to display
display_count = 0
for img_id in list(test_features.keys()):
    if display_count >= num_display_samples: break
    
    photo_feature = test_features.get(img_id)
    if photo_feature is None: continue

    print(f"\nImage: {img_id}")
    
    # Greedy Prediction
    greedy_caption = greedy_search_predict(photo_feature, captioning_model, tokenizer, max_length, indices_to_words, END_TOKEN_STRING)
    print(f"  Greedy Caption: {greedy_caption}")

    # Beam Search Prediction
    beam_caption = beam_search_predict(photo_feature, captioning_model, tokenizer, max_length, indices_to_words, beam_k_eval, END_TOKEN_STRING)
    print(f"  Beam (k={beam_k_eval}) Caption: {beam_caption}")
    
    # Display image (optional, may not work well in all script environments)
    try:
        img_path_display = os.path.join(images_base_path, img_id)
        img_display_cv = cv2.imread(img_path_display)
        if img_display_cv is not None:
            plt.figure(figsize=(5,5))
            plt.imshow(cv2.cvtColor(img_display_cv, cv2.COLOR_BGR2RGB))
            plt.title(f"Greedy: {greedy_caption}\nBeam(k={beam_k_eval}): {beam_caption}", fontsize=8)
            plt.axis('off')
            plt.savefig(f"sample_prediction_{img_id.replace('.jpg', '')}.png") # Save the plot
            print(f"Saved sample prediction image: sample_prediction_{img_id.replace('.jpg', '')}.png")
            plt.close() # Close the plot to prevent display in non-interactive env
    except Exception as e_disp:
        print(f"Error displaying/saving sample image {img_id}: {e_disp}")
        
    display_count += 1
